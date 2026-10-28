#pragma once
// Native MMA operator using the existing VM worker loops and page lifetimes.
// CUTLASS is used only for descriptors/layouts and instruction atoms, never its
// standalone kernel scheduler, epilogue, allocator, or instruction loop.
#include "megakernel.cuh"
#include "nvfp4_types.cuh"
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include "gmem_barrier.cuh"

namespace megakernel::nvfp4 {
struct Config : megakernel::thor_config {
 static constexpr int NUM_CONSUMER_WARPS=4,NUM_WARPS=8,NUM_THREADS=256;
 static constexpr int PAGE_SIZE=65536,NUM_PAGES=2;
 static constexpr int DYNAMIC_SHARED_MEMORY=NUM_PAGES*PAGE_SIZE+1023;
 static constexpr int NATIVE_TMEM_COLUMNS=512;
};
static_assert(sizeof(Mainloop::TensorStorage)<=Config::PAGE_SIZE);
struct Operand {
 Mainloop::Params mainloop;
 cute::Shape<int,int,int,int> problem_shape;
 const __nv_bfloat16* input;
 __nv_bfloat16* output;
 unsigned char *packed_input,*input_scales;
 int m,k;
 float input_tensor_scale,alpha;
 const __nv_bfloat16 *norm=nullptr,*input_gate=nullptr,*residual=nullptr;
 float norm_eps=1e-5f;
 int transform=0;
 // Optional weight-only mode: prepare activations as BF16, never as FP4.
 __nv_bfloat16* bf16_input=nullptr;
 const unsigned char* weight_scales=nullptr;
};
struct Globals {
 megakernel::instruction_layout<Config> instructions;
 megakernel::timing_layout<Config> timings;
 const Operand* operands;
 int *ready,*done;
 int workers;
 bool dependent,quantize,fused_quant;
 int length;
 int quant_workers;
};
using State=megakernel::state<Config>;
__device__ inline kittens::semaphore& sem(State& s,int i){return s.semaphores()[i];}
__device__ inline Mainloop::TensorStorage& storage(State& s){
 return *reinterpret_cast<Mainloop::TensorStorage*>(s.pages[s.pid(0)].ptr());
}
template<int Opcode=8> struct LinearOp {
 static constexpr int opcode=Opcode;
 struct controller {
  __device__ static int release_lid(const Globals&,Config::instruction_t&,int& q){return 1-q;}
  __device__ static int init_semaphores(const Globals&,State& s){
   // 0,1: TMA arrivals; 2,3: async MMA stage retirement.
   for(int i=0;i<4;++i) kittens::init_semaphore(sem(s,i),1);
   kittens::init_semaphore(sem(s,4),1); // accumulated tile ready
   kittens::init_semaphore(sem(s,5),Config::NUM_CONSUMER_WARPS);
   kittens::init_semaphore(sem(s,6),Config::NUM_CONSUMER_WARPS); // outputs published
   return 7;
  }
 };
 struct loader {
  __device__ static void run(const Globals& g,State& s){
   if(kittens::laneid()==1){int pid=s.pid(1);s.wait_page_ready(pid);s.finish_page(pid,Config::NUM_CONSUMER_WARPS);}
   if(kittens::laneid()!=0)return;
   int index=s.instruction()[1],begin=s.instruction()[2],end=s.instruction()[3];
   auto& o=g.operands[index];int pid=s.pid(0);s.wait_page_ready(pid);
   if(begin==end){s.finish_page(pid,Config::NUM_CONSUMER_WARPS);return;}
   if(g.quantize) {
    int* counter=g.fused_quant && index>0?g.done+index-1:g.ready+index;
    int target=g.fused_quant && index>0?g.workers:g.quant_workers;
    megakernel::gmem_wait<Config>([&]{return megakernel::gmem_load_acquire(counter)<target;});
   }
   Mainloop ml(o.mainloop, NvCluster{},0);
   auto loads=ml.load_init(o.problem_shape,storage(s));
   auto [unused,a,b,sa,sb,sfa,sfb,ssfa,ssfb,ma,mb,msa,msb]=loads;
   int iteration=0;
   for(int tile=begin;tile<end;++tile){
    auto ga=a(cute::_,tile,cute::_,0);auto gb=b(cute::_,0,cute::_,0);
    auto gsa=sfa(cute::_,tile,cute::_,0);auto gsb=sfb(cute::_,0,cute::_,0);
    for(int k=0;k<o.k/256;++k,++iteration){
     int stage=iteration%2;int phase=(iteration/2)&1;
     kittens::wait(sem(s,2+stage),phase^1);
     kittens::tma::expect_bytes(sem(s,stage),uint32_t(Mainloop::TmaTransactionBytes));
     cute::copy(o.mainloop.tma_load_a.with(*reinterpret_cast<uint64_t*>(&sem(s,stage)),ma),ga(cute::_,k),sa(cute::_,stage));
     cute::copy(o.mainloop.tma_load_b.with(*reinterpret_cast<uint64_t*>(&sem(s,stage)),mb),gb(cute::_,k),sb(cute::_,stage));
     cute::copy(o.mainloop.tma_load_sfa.with(*reinterpret_cast<uint64_t*>(&sem(s,stage)),msa),gsa(cute::_,k),ssfa(cute::_,stage));
     cute::copy(o.mainloop.tma_load_sfb.with(*reinterpret_cast<uint64_t*>(&sem(s,stage)),msb),gsb(cute::_,k),ssfb(cute::_,stage));
    }
   }
  }
 };
 struct launcher {
  __device__ static void run(const Globals& g,State& s){
   int begin=s.instruction()[2],end=s.instruction()[3];if(begin==end)return;
   auto& o=g.operands[s.instruction()[1]];
   Mainloop ml(o.mainloop,NvCluster{},0);
   auto tm=Mainloop::init_tmem_tensors(typename Epilogue::EpilogueTile{});
   Mainloop::set_tmem_offsets(tm,s.native_tmem_base);
   auto inputs=ml.mma_init(tm,storage(s));
   auto [mma,a,b,sfa,sfb,ca,ssa,tsa,cb,ssb,tsb]=inputs;
   auto acc=tm.accumulators(cute::_,cute::_,cute::_,0);
   int iteration=0;
   for(int tile=begin;tile<end;++tile){
    kittens::wait(sem(s,5),((tile-begin)&1)^1);
    mma.accumulate_=cute::UMMA::ScaleOut::Zero;
    for(int k=0;k<o.k/256;++k,++iteration){
     int stage=iteration%2;int phase=(iteration/2)&1;
     kittens::wait(sem(s,stage),phase);
     if(cute::elect_one_sync()){
      cute::copy(ca,ssa(cute::_,cute::_,cute::_,cute::_,stage),tsa);
      cute::copy(cb,ssb(cute::_,cute::_,cute::_,cute::_,stage),tsb);
     }
     #pragma unroll
     for(int kb=0;kb<cute::size<2>(a);++kb){
      cute::gemm(mma.with(mma.accumulate_,sfa(cute::_,cute::_,kb),sfb(cute::_,cute::_,kb)),
       a(cute::_,cute::_,kb,stage),b(cute::_,cute::_,kb,stage),acc);
      mma.accumulate_=cute::UMMA::ScaleOut::One;
     }
     cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&sem(s,2+stage)));
    }
    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&sem(s,4)));
   }
   kittens::wait(sem(s,5),(end-begin-1)&1);
   if(kittens::laneid()==0)s.finish_page(s.pid(0),Config::NUM_CONSUMER_WARPS);
  }
 };
 struct consumer {
  __device__ static void run(const Globals& g,State& s){
   int index=s.instruction()[1];int begin=s.instruction()[2],end=s.instruction()[3];auto& o=g.operands[index];
   for(int tile=begin;tile<end;++tile){
    kittens::wait(sem(s,4),(tile-begin)&1);
    uint32_t value;
    cute::SM100_TMEM_LOAD_32dp32b1x::copy(s.native_tmem_base+((kittens::warpid()*32)<<16),value);
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    int row=tile*128+kittens::warpid()*32+kittens::laneid();
    __nv_bfloat16 output=__float2bfloat16(__uint_as_float(value)*o.alpha);
    if(row<o.m){
     if(o.residual)output=__float2bfloat16(__bfloat162float(output)+__bfloat162float(o.residual[row]));
     o.output[row]=output;
    }
    // Producer epilogue quantization: preserve the BF16 rounding boundary,
    // then quantize directly from registers for a compatible next operand.
    // This removes a separate vector pass and its global completion protocol.
    if(g.fused_quant && index+1<g.length){
     auto& next_operand=g.operands[index+1];int lane=kittens::laneid();
     float v=__bfloat162float(output)/next_operand.input_tensor_scale;
     float a=fabsf(v);
     for(int d=8;d;d>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
     __nv_fp8_e4m3 sf(fmaxf(a/6.f,1.f/512.f));
     unsigned char q=__nv_cvt_float_to_fp4(v/float(sf),__NV_E2M1,cudaRoundNearest);
     auto layout=SFConfig::tile_atom_to_shape_SFB(cute::make_shape(next_operand.m,1,next_operand.k,1));
     if(row<o.m && lane%16==0)next_operand.input_scales[layout(cute::make_coord(0,row,0))]=sf.__x;
     unsigned char qnext=__shfl_down_sync(0xffffffff,unsigned(q),1);
     if(row<o.m && lane%2==0)next_operand.packed_input[row/2]=q|(qnext<<4);
     asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    __threadfence();__syncwarp();
    if(kittens::laneid()==0)kittens::arrive(sem(s,5));
   }
   if(kittens::laneid()==0)kittens::arrive(sem(s,6));
  }
 };
 struct storer {
  __device__ static void run(const Globals& g,State& s){
   kittens::wait(sem(s,6),0);
   if(kittens::laneid()==0)megakernel::gmem_add_release(g.done+s.instruction()[1],1);
  }
 };
};
template<int Opcode=9> struct QuantizeOp : megakernel::NoOp<Config> {
 static constexpr int opcode=Opcode;
 struct controller {
  __device__ static int release_lid(const Globals&,Config::instruction_t&,int& q){return q;}
  __device__ static int init_semaphores(const Globals&,State& s){kittens::init_semaphore(sem(s,0),Config::NUM_CONSUMER_WARPS);return 1;}
 };
 struct consumer {
  __device__ static void run(const Globals& g,State& s){
   int index=s.instruction()[1];auto& o=g.operands[index];int lane=kittens::laneid();
   if(megakernel::get_worker_id()>=g.quant_workers)return;
   if(g.dependent && index>0 && lane==0)megakernel::gmem_wait<Config>([&]{return megakernel::gmem_load_acquire(g.done+index-1)<g.workers;});
   __syncwarp();
   float norm_scale=1.f;
   if(o.transform==1){
    float sum=0;for(int c=lane;c<o.k;c+=32){float v=__bfloat162float(o.input[c]);sum=fmaf(v,v,sum);}
    for(int d=16;d;d>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,d);
    norm_scale=rsqrtf(sum/o.k+o.norm_eps);
   }
   auto layout=SFConfig::tile_atom_to_shape_SFB(cute::make_shape(o.m,1,o.k,1));
   for(int group=megakernel::get_worker_id()*Config::NUM_CONSUMER_WARPS+kittens::warpid();group<o.k/16;group+=g.quant_workers*Config::NUM_CONSUMER_WARPS){
    int col=group*16;float v=lane<16?__bfloat162float(o.input[col+lane]):0.f;
    if(lane<16 && o.transform==1){
     v=__bfloat162float(__float2bfloat16(v*norm_scale));
     v=__bfloat162float(__float2bfloat16(v*__bfloat162float(o.norm[col+lane])));
    }
    if(lane<16 && o.transform==2){float gate=__bfloat162float(o.input_gate[col+lane]);v=__bfloat162float(__float2bfloat16(v*(gate/(1.f+expf(-gate)))));}
    if(o.bf16_input){
     if(lane<16)o.bf16_input[col+lane]=__float2bfloat16(v);
     continue;
    }
    v/=o.input_tensor_scale;
    float a=fabsf(v);for(int d=16;d;d>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
    __nv_fp8_e4m3 sf(fmaxf(a/6.f,1.f/512.f));
    unsigned char q=__nv_cvt_float_to_fp4(v/float(sf),__NV_E2M1,cudaRoundNearest);
    if(lane==0)o.input_scales[layout(cute::make_coord(0,col,0))]=sf.__x;
    unsigned char next=__shfl_down_sync(0xffffffff,unsigned(q),1);
    if(lane<16 && lane%2==0)o.packed_input[(col+lane)/2]=q|(next<<4);
   }
   __threadfence();asm volatile("fence.proxy.async.global;" ::: "memory");__syncwarp();
   if(lane==0)kittens::arrive(sem(s,0));
  }
 };
 struct storer {
  __device__ static void run(const Globals& g,State& s){
   if(megakernel::get_worker_id()>=g.quant_workers)return;
   kittens::wait(sem(s,0),0);
   if(kittens::laneid()==0)megakernel::gmem_add_release(g.ready+s.instruction()[1],1);
  }
 };
};
using Linear=LinearOp<>;
using Quantize=QuantizeOp<>;
}
