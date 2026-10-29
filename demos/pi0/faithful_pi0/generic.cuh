#pragma once
// Device replacements for opaque cuBLAS sites. Source prefix tactics stay separate.
namespace faithful_generic {
#ifndef PI0_GENERIC_M
#define PI0_GENERIC_M 128
#endif
#ifndef PI0_GENERIC_N
#define PI0_GENERIC_N 128
#endif
#ifdef PI0_PHASE_ACTION
using Tile=cute::Shape<cute::_64,cute::_128,cute::_64>;
using Cluster=cute::Shape<cute::_1,cute::_1,cute::_1>;
using Schedule=cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
#else
using Tile=cute::Shape<cute::Int<PI0_GENERIC_M>,cute::Int<PI0_GENERIC_N>,cute::_64>;
using Cluster=cute::Shape<cute::_2,cute::_2,cute::_1>;
using Schedule=cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
#endif
using T=cutlass::bfloat16_t;
using Fusion=cutlass::epilogue::fusion::LinCombEltAct<cutlass::epilogue::thread::Identity,T,float>;
using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm100,cutlass::arch::OpClassTensorOp,
    Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,float,float,T,cutlass::layout::RowMajor,8,
    T,cutlass::layout::RowMajor,8,cutlass::epilogue::collective::EpilogueScheduleAuto,Fusion>::CollectiveOp;
template<class BLayout> struct Types {
    using Main=typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100,cutlass::arch::OpClassTensorOp,
        T,cutlass::layout::RowMajor,8,T,BLayout,8,float,Tile,Cluster,cutlass::gemm::collective::StageCount<3>,
        Schedule>::CollectiveOp;
    using Kernel=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,Main,Epi,faithful::StaticSchedulerTag>;
};
using NT=Types<cutlass::layout::ColumnMajor>::Kernel;
using NN=Types<cutlass::layout::RowMajor>::Kernel;
struct Pack {
    const __nv_bfloat16 *a,*b;__nv_bfloat16 *out,*pa,*pb,*pd;
    int m,n,k,np,kp,lda,ldbn,ldbk,ldo;float beta;
};
template<class Kernel> struct Params {typename Kernel::Params gemm;Pack pack;};
__device__ inline void pack(Pack const& p) {
    int tid=megakernel::get_worker_id()*256+threadIdx.x;
    if(p.pa)for(int i=tid;i<p.m*p.kp;i+=faithful_pointwise::worker_count()*256){int row=i/p.kp,col=i%p.kp;p.pa[i]=col<p.k?p.a[row*p.lda+col]:__float2bfloat16(0.f);}
    if(p.pb)for(int i=tid;i<p.np*p.kp;i+=faithful_pointwise::worker_count()*256){
        int row=p.ldbk==1?i/p.kp:i%p.np,col=p.ldbk==1?i%p.kp:i/p.np;
        p.pb[i]=(row<p.n && col<p.k)?p.b[row*p.ldbn+col*p.ldbk]:__float2bfloat16(0.f);
    }
    if(p.pd && p.beta!=0.f)for(int i=tid;i<p.m*p.np;i+=faithful_pointwise::worker_count()*256){int row=i/p.np,col=i%p.np;p.pd[i]=col<p.n?p.out[row*p.ldo+col]:__float2bfloat16(0.f);}
}
template<class Kernel> __device__ inline void run(Params<Kernel> const& p,char* arena,uint32_t tmem) {
#ifdef PI0_COHERENT_FAST_BARRIERS
    bool prepare=p.pack.pa || p.pack.pb || (p.pack.pd && p.pack.beta!=0.f);
    if(prepare) {
        pack(p.pack);__syncthreads();
        asm volatile("fence.proxy.async.global;" ::: "memory");
        cooperative_groups::this_grid().sync();
    }
    Kernel{}(p.gemm,arena,tmem);
    if(p.pack.pd) {
        __syncthreads();asm volatile("fence.proxy.async.global;" ::: "memory");
        cooperative_groups::this_grid().sync();
    }
#else
    pack(p.pack);__syncthreads();cooperative_groups::this_grid().sync();
    Kernel{}(p.gemm,arena,tmem);
    __syncthreads();cooperative_groups::this_grid().sync();
#endif
    auto const& a=p.pack;
    if(a.pd)for(int i=megakernel::get_worker_id()*256+threadIdx.x;i<a.m*a.n;i+=faithful_pointwise::worker_count()*256){int row=i/a.n,col=i%a.n;a.out[row*a.ldo+col]=a.pd[row*a.np+col];}
}
}
