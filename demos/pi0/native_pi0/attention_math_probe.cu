// Diagnose BF16 rounding against actual first-layer OpenPI Q/K/V fixtures.
#define PI0_MATH_PROBE
#define NATIVE_ATTENTION_SCORE_SCALING 1
#include "attention.cu"
#include "model_attention.cuh"
struct ProbePage {
    char bytes[16384];
    __device__ void* ptr(){return bytes;}
};
struct ProbeState {
    ProbePage* pages;
    __device__ int pid(int i){return i;}
};
__global__ void probe_vision_kernel(const kittens::bf16* q,const kittens::bf16* k,
                                   const kittens::bf16* v,kittens::bf16* out,int mode){
    extern __shared__ char memory[];
    auto aligned=(reinterpret_cast<uintptr_t>(memory)+1023)&~uintptr_t(1023);
    ProbeState state{reinterpret_cast<ProbePage*>(aligned)};
    float scale=mode&8?float(double(float(1./sqrt(72.)))*1.4426950408889634):float(1./sqrt(72.))*1.4426950408889634f;
    native_model_attention::flash_attend<kittens::bf16,72>(state,blockIdx.x,q,k,v,1152,1152,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,out,1152,256,256,16,16,
        scale,-1,nullptr,0,256,
        nullptr,nullptr,nullptr,nullptr,0,0,false,mode);
}
extern "C" int probe_vision_math(const void* q,const void* k,const void* v,void* out,int views,int mode,void* stream){
    constexpr int bytes=4*16384+1023;
    auto error=cudaFuncSetAttribute(probe_vision_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(error!=cudaSuccess)return error;
    probe_vision_kernel<<<views*16*4,128,bytes,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const kittens::bf16*>(q),static_cast<const kittens::bf16*>(k),
        static_cast<const kittens::bf16*>(v),static_cast<kittens::bf16*>(out),mode);
    return cudaGetLastError();
}
