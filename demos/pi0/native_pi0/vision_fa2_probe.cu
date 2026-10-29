#include <cuda_runtime.h>
#include "vision_fa2.cuh"
struct VisionProbe {
    const void *q,*k,*v;void* output;float* statistics;
    int batches,nq=256,nk=256,ldq=1152,ldkv=1152,out_ld=1152;
    float scale_log2=float(double(float(1./sqrt(72.)))*1.4426950408889634);
};
__global__ void vision_fa2_kernel(VisionProbe a){
    extern __shared__ char pool[];
    native_vision_fa2(a,blockIdx.x,pool);
}
extern "C" int probe_vision_fa2(const void* q,const void* k,const void* v,void* out,float* lse,int views,void* stream){
    auto error=cudaFuncSetAttribute(vision_fa2_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,VisionTraits::kSmemSize);
    if(error!=cudaSuccess)return error;
    VisionProbe a{q,k,v,out,lse,views};
    vision_fa2_kernel<<<views*16*2,128,VisionTraits::kSmemSize,static_cast<cudaStream_t>(stream)>>>(a);
    return cudaGetLastError();
}
