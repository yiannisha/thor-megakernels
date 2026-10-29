#include "static_scheduler.cuh"
#include "gemm_types_bf16.h"
#include <cooperative_groups.h>
using NativeKernel=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,
    sm100_fp16_k64::Main,sm100_fp16_k64::Epi,faithful::StaticSchedulerTag>;
__global__ void __cluster_dims__(2,2,1) native_collective(NativeKernel::Params const* params) {
    extern __shared__ __align__(128) char arena[];
    NativeKernel{}(*params,arena);
}
extern "C" int faithful_native_probe(void* a,void* b,void* out,int m,int n,int k,void* stream) {
    using G=cutlass::gemm::device::GemmUniversalAdapter<NativeKernel>;
    auto sa=cutlass::make_cute_packed_stride(NativeKernel::StrideA{},{m,k,1});
    auto sb=cutlass::make_cute_packed_stride(NativeKernel::StrideB{},{n,k,1});
    auto sd=cutlass::make_cute_packed_stride(NativeKernel::StrideD{},{m,n,1});
    G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,1},
      {(G::ElementA*)a,sa,(G::ElementB*)b,sb},{{1.f,0.f},(G::ElementC*)out,sd,(G::ElementD*)out,sd}};
    auto params=NativeKernel::to_underlying_arguments(args,nullptr);
    NativeKernel::Params* dp=nullptr;
    auto st=static_cast<cudaStream_t>(stream);
    cudaError_t e=cudaMalloc(&dp,sizeof(params));if(e)return int(e);
    e=cudaMemcpyAsync(dp,&params,sizeof(params),cudaMemcpyHostToDevice,st);if(e){cudaFree(dp);return int(e);}
    e=cudaFuncSetAttribute(native_collective,cudaFuncAttributeMaxDynamicSharedMemorySize,NativeKernel::SharedStorageSize);
    if(e){cudaFree(dp);return int(e);}
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(4,4,1);cfg.blockDim=dim3(NativeKernel::MaxThreadsPerBlock,1,1);
    cfg.dynamicSmemBytes=NativeKernel::SharedStorageSize;cfg.stream=st;
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeCooperative;attr.val.cooperative=1;
    cfg.attrs=&attr;cfg.numAttrs=1;
    e=cudaLaunchKernelEx(&cfg,native_collective,dp);
    if(!e)e=cudaStreamSynchronize(st);
    cudaFree(dp);return int(e);
}
