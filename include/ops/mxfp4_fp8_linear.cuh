#pragma once
#include "nvfp4_linear.cuh"
namespace megakernel::nvfp4::mixed8 {
static_assert(sizeof(Mainloop::TensorStorage)<=Config::NUM_PAGES*Config::PAGE_SIZE);
__device__ inline Mainloop::TensorStorage& mixed_storage(State& s){return *reinterpret_cast<Mainloop::TensorStorage*>(s.pages[0].ptr());}
struct Linear {
 struct controller : nvfp4::Linear::controller {
  __device__ static int release_lid(const Globals&,Config::instruction_t&,int& q){return q;}
 };
 struct loader {
  __device__ static void run(const Globals& g,State& s){
   if(kittens::laneid()!=0)return;
   int index=s.instruction()[1],begin=s.instruction()[2],end=s.instruction()[3];
   auto& o=g.operands[index];int pid=0;for(int p=0;p<Config::NUM_PAGES;++p)s.wait_page_ready(p);
   if(begin==end){for(int p=0;p<Config::NUM_PAGES;++p)s.finish_page(p,Config::NUM_CONSUMER_WARPS);return;}
   if(g.quantize) {
    int* counter=g.fused_quant && index>0?g.done+index-1:g.ready+index;
    int target=g.fused_quant && index>0?g.workers:g.quant_workers;
    megakernel::gmem_wait<Config>([&]{return megakernel::gmem_load_acquire(counter)<target;});
   }
   Mainloop ml(o.mixed_mainloop, NvCluster{},0);
   auto loads=ml.load_init(o.problem_shape,mixed_storage(s));
   auto [unused,a,b,sa,sb,sfa,sfb,ssfa,ssfb,ma,mb,msa,msb]=loads;
   int iteration=0;
   for(int tile=begin;tile<end;++tile){
    auto ga=a(cute::_,tile,cute::_,0);auto gb=b(cute::_,0,cute::_,0);
    auto gsa=sfa(cute::_,tile,cute::_,0);auto gsb=sfb(cute::_,0,cute::_,0);
    for(int k=0;k<o.k/256;++k,++iteration){
     int stage=iteration%2;int phase=(iteration/2)&1;
     kittens::wait(sem(s,2+stage),phase^1);
     kittens::tma::expect_bytes(sem(s,stage),uint32_t(Mainloop::TmaTransactionBytes));
     cute::copy(o.mixed_mainloop.tma_load_a.with(*reinterpret_cast<uint64_t*>(&sem(s,stage)),ma),ga(cute::_,k),sa(cute::_,stage));
     cute::copy(o.mixed_mainloop.tma_load_b.with(*reinterpret_cast<uint64_t*>(&sem(s,stage)),mb),gb(cute::_,k),sb(cute::_,stage));
     cute::copy(o.mixed_mainloop.tma_load_sfa.with(*reinterpret_cast<uint64_t*>(&sem(s,stage)),msa),gsa(cute::_,k),ssfa(cute::_,stage));
     cute::copy(o.mixed_mainloop.tma_load_sfb.with(*reinterpret_cast<uint64_t*>(&sem(s,stage)),msb),gsb(cute::_,k),ssfb(cute::_,stage));
    }
   }
  }
 };
 struct launcher {
  __device__ static void run(const Globals& g,State& s){
   int begin=s.instruction()[2],end=s.instruction()[3];if(begin==end)return;
   auto& o=g.operands[s.instruction()[1]];
   Mainloop ml(o.mixed_mainloop,NvCluster{},0);
   auto tm=Mainloop::init_tmem_tensors(typename Epilogue::EpilogueTile{});
   Mainloop::set_tmem_offsets(tm,s.native_tmem_base);
   auto inputs=ml.mma_init(tm,mixed_storage(s));
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
   if(kittens::laneid()==0)for(int p=0;p<Config::NUM_PAGES;++p)s.finish_page(p,Config::NUM_CONSUMER_WARPS);
  }
 };
 using consumer=nvfp4::Linear::consumer;
 using storer=nvfp4::Linear::storer;
};
}
