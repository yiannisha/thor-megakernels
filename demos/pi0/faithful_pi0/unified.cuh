#pragma once
// Coherent group-2 runtime variant. Vision uses legacy BF16 mma.sync;
// projections retain tcgen05 group 2, with a fixed physical 2x2 cluster.
#include "coherent_down_types.h"
struct CoherentVision {
    void const *q,*k,*v;void *output;float* statistics;
    int nq,nk,batches,ldq,ldkv,out_ld;float scale_log2;
};
__device__ inline void coherent_vision(CoherentVision const& a,char* arena) {
    int tasks=a.batches*16*((a.nq+127)/128);
    if(threadIdx.x<128)for(int task=megakernel::get_worker_id();task<tasks;task+=16)
        native_vision_fa2(a,task,arena);
}
namespace coherent_geglu {
using Tile=faithful_geglu::Tile;using Cluster=faithful_geglu::Cluster;
using T=cutlass::bfloat16_t;
using GateFusion=cutlass::epilogue::fusion::LinCombEltAct<cutlass::epilogue::thread::GELU_taylor,T,float>;
using UpFusion=cutlass::epilogue::fusion::LinCombDeEltAct<cutlass::layout::RowMajor,cutlass::multiplies,T,float,T,T,float>;
template<class Fusion> using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100,cutlass::arch::OpClassTensorOp,Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,
    float,float,T,cutlass::layout::RowMajor,8,T,cutlass::layout::RowMajor,8,
    cutlass::epilogue::collective::EpilogueScheduleAuto,Fusion>::CollectiveOp;
using Gate=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,faithful_geglu::CollectiveMmaGate,Epi<GateFusion>,faithful::StaticSchedulerTag>;
using Up=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,faithful_geglu::CollectiveMmaUp,Epi<UpFusion>,faithful::StaticSchedulerTag>;
struct Params {Gate::Params gate;Up::Params up;};
__device__ inline void run(Params const& p,char* arena,uint32_t tmem) {
    Gate{}(p.gate,arena,tmem);__syncthreads();
    cooperative_groups::this_cluster().sync();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("fence.proxy.async.global;" ::: "memory");
    cooperative_groups::this_grid().sync();
    Up{}(p.up,arena,tmem);
}
}
