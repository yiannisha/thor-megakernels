#pragma once
#include "nvfp4_linear.cuh"
namespace megakernel::nvfp4::weight_only {
using BF=cutlass::bfloat16_t;
using Mma=decltype(cute::make_tiled_mma(cute::SM100_MMA_F16BF16_SS<BF,BF,float,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>{}));
using AShape=decltype(cute::partition_shape_A(Mma{},cute::Shape<cute::_128,cute::_128>{}));
using BShape=decltype(cute::partition_shape_B(Mma{},cute::Shape<cute::_8,cute::_128>{}));
using ALayout=decltype(cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW128_Atom<BF>{},cute::append(AShape{},cute::Int<2>{})));
using BLayout=decltype(cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW128_Atom<BF>{},cute::append(BShape{},cute::Int<2>{})));
struct Buffers {
 Mainloop::TensorStorage packed;
 alignas(128) cute::ArrayEngine<BF,cute::cosize_v<ALayout>> expanded;
 alignas(128) cute::ArrayEngine<BF,cute::cosize_v<BLayout>> activation;
};
static_assert(sizeof(Buffers)<=Config::NUM_PAGES*Config::PAGE_SIZE);
__device__ inline Buffers& buffers(State& s){return *reinterpret_cast<Buffers*>(s.pages[0].ptr());}
struct Linear {
 struct controller {
  __device__ static int release_lid(const Globals&,Config::instruction_t&,int& q){return q;}
  __device__ static int init_semaphores(const Globals&,State& s){
   for(int i=0;i<4;++i)kittens::init_semaphore(sem(s,i),1);
   kittens::init_semaphore(sem(s,4),Config::NUM_CONSUMER_WARPS); // BF16 tile produced
   kittens::init_semaphore(sem(s,5),Config::NUM_CONSUMER_WARPS); // accumulator released
   kittens::init_semaphore(sem(s,6),Config::NUM_CONSUMER_WARPS); // outputs published
   kittens::init_semaphore(sem(s,7),1); // MMA completed, BF16 staging reusable
   kittens::init_semaphore(sem(s,8),1); // physical pool ownership acquired
   kittens::init_semaphore(sem(s,9),Config::NUM_CONSUMER_WARPS); // BF16 stage 1 produced
   kittens::init_semaphore(sem(s,10),1); // BF16 stage 1 retired
   kittens::init_semaphore(sem(s,11),1); // complete accumulator available
   return 12;
  }
 };
 struct loader {
  __device__ static void run(const Globals& g,State& s){
   if(kittens::laneid()!=0)return;
   for(int p=0;p<Config::NUM_PAGES;++p)s.wait_page_ready(p);
   int index=s.instruction()[1],begin=s.instruction()[2],end=s.instruction()[3];
   auto& o=g.operands[index];
   if(g.quantize)megakernel::gmem_wait<Config>([&]{return megakernel::gmem_load_acquire(g.ready+index)<g.quant_workers;});
   kittens::arrive(sem(s,8));
   if(begin==end)return;
   Mainloop ml(o.mainloop,NvCluster{},0);auto loads=ml.load_init(o.problem_shape,buffers(s).packed);
   auto [unused,a,b,sa,sb,sfa,sfb,ssfa,ssfb,ma,mb,msa,msb]=loads;
   int iteration=0;
   for(int tile=begin;tile<end;++tile){auto ga=a(cute::_,tile,cute::_,0);auto gsa=sfa(cute::_,tile,cute::_,0);
    for(int k=0;k<o.k/256;++k,++iteration){int stage=iteration%2,phase=(iteration/2)&1;
     kittens::wait(sem(s,2+stage),phase^1);
     kittens::tma::expect_bytes(sem(s,stage),18432);
     cute::copy(o.mainloop.tma_load_a.with(*reinterpret_cast<uint64_t*>(&sem(s,stage)),ma),ga(cute::_,k),sa(cute::_,stage));
     cute::copy(o.mainloop.tma_load_sfa.with(*reinterpret_cast<uint64_t*>(&sem(s,stage)),msa),gsa(cute::_,k),ssfa(cute::_,stage));
    }
   }
  }
 };
 struct launcher {
  __device__ static void run(const Globals& g,State& s){
   int begin=s.instruction()[2],end=s.instruction()[3];auto& o=g.operands[s.instruction()[1]];
   if(begin==end){kittens::wait(sem(s,8),0);if(kittens::laneid()==0)for(int p=0;p<Config::NUM_PAGES;++p)s.finish_page(p,Config::NUM_CONSUMER_WARPS);return;}
   auto& sm=buffers(s);auto mma=Mma{};auto cta=mma.get_slice(0);
   auto sa=cute::make_tensor(cute::make_smem_ptr(sm.expanded.begin()),ALayout{});
   auto sb=cute::make_tensor(cute::make_smem_ptr(sm.activation.begin()),BLayout{});
   auto ra=cta.make_fragment_A(sa);auto rb=cta.make_fragment_B(sb);
   auto gc=cute::make_tensor(cute::make_gmem_ptr((float*)nullptr),cute::make_layout(cute::Shape<cute::_128,cute::_8>{},cute::Stride<cute::_1,cute::_128>{}));
   auto acc=cta.make_fragment_C(cta.partition_C(gc));acc.data()=s.native_tmem_base;
   int step=0;
   for(int tile=begin;tile<end;++tile){
    kittens::wait(sem(s,5),((tile-begin)&1)^1);mma.accumulate_=cute::UMMA::ScaleOut::Zero;
    for(int k=0;k<o.k/128;++k,++step){
     int bfstage=step&1;
     kittens::wait(sem(s,bfstage?9:4),(step/2)&1);
     #pragma unroll
     for(int b=0;b<cute::size<2>(ra);++b){cute::gemm(mma,ra(cute::_,cute::_,b,bfstage),rb(cute::_,cute::_,b,bfstage),acc);mma.accumulate_=cute::UMMA::ScaleOut::One;}
     cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&sem(s,bfstage?10:7)));
    }
    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&sem(s,11)));
   }
   kittens::wait(sem(s,5),(end-begin-1)&1);
   if(kittens::laneid()==0)for(int p=0;p<Config::NUM_PAGES;++p)s.finish_page(p,Config::NUM_CONSUMER_WARPS);
  }
 };
 struct consumer {
  __device__ static void run(const Globals& g,State& s){
   kittens::wait(sem(s,8),0);auto& o=g.operands[s.instruction()[1]];
   int begin=s.instruction()[2],end=s.instruction()[3],lane=kittens::laneid(),warp=kittens::warpid(),tid=warp*32+lane;
   auto& sm=buffers(s);
   auto packed=cute::make_tensor(cute::make_smem_ptr(sm.packed.smem_A.begin()),Mainloop::SmemLayoutA{});
   auto sa=cute::make_tensor(cute::make_smem_ptr(sm.expanded.begin()),ALayout{});
   auto sb=cute::make_tensor(cute::make_smem_ptr(sm.activation.begin()),BLayout{});
   int iteration=0,step=0;
   for(int tile=begin;tile<end;++tile){
    for(int chunk=0;chunk<o.k/256;++chunk,++iteration){int stage=iteration%2;
     kittens::wait(sem(s,stage),(iteration/2)&1);
     for(int half=0;half<2;++half,++step){
      int bfstage=step&1;
      kittens::wait(sem(s,bfstage?10:7),((step/2)&1)^1);
      for(int pair=tid;pair<128*128/2;pair+=128){int row=pair/64,col=(pair%64)*2,pc=half*128+col;
       auto q0=cutlass::float_e2m1_t(packed(cute::make_coord(row,pc%64),0,pc/64,stage));
       auto q1=cutlass::float_e2m1_t(packed(cute::make_coord(row,(pc+1)%64),0,(pc+1)/64,stage));
       unsigned char byte=q0.raw()|(q1.raw()<<4);
       __half2 h=__half2(__nv_cvt_fp4x2_to_halfraw2(byte,__NV_E2M1));
       int gr=tile*128+row;
       // CUTLASS's compact 128-row, block-16 scale atom: four K64
       // blocks, each 512 bytes. Stage all scales with the packed weights.
       int scale_offset=stage*2048+(pc/64)*512+(row%32)*16+(row/32)*4+(pc/16)%4;
       __nv_fp8_e4m3 sf;sf.__x=gr<o.m?reinterpret_cast<unsigned char*>(sm.packed.smem_SFA.begin())[scale_offset]:0;
       sa(cute::make_coord(row,col%16),0,col/16,bfstage)=BF(__low2float(h)*float(sf));
       sa(cute::make_coord(row,(col+1)%16),0,(col+1)/16,bfstage)=BF(__high2float(h)*float(sf));
      }
      for(int i=tid;i<8*128;i+=128){int row=i/128,col=i%128;
       sb(cute::make_coord(row,col%16),0,col/16,bfstage)=BF(row==0?__bfloat162float(o.bf16_input[chunk*256+half*128+col]):0.f);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");__syncwarp();
      if(lane==0)kittens::arrive(sem(s,bfstage?9:4));
     }
     kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);
     if(warp==0 && lane==0)kittens::arrive(sem(s,2+stage));
    }
    kittens::wait(sem(s,11),(tile-begin)&1);
    uint32_t value;cute::SM100_TMEM_LOAD_32dp32b1x::copy(s.native_tmem_base+((warp*32)<<16),value);
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");int row=tile*128+tid;
    if(row<o.m){auto out=__float2bfloat16(__uint_as_float(value)*o.alpha);
     if(o.residual)out=__float2bfloat16(__bfloat162float(out)+__bfloat162float(o.residual[row]));o.output[row]=out;
    }
    __threadfence();__syncwarp();if(lane==0)kittens::arrive(sem(s,5));
   }
   if(lane==0)kittens::arrive(sem(s,6));
  }
 };
};
}
