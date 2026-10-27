#include "kittens.cuh"
#include <cstdio>
// Compare 1 arrival/warp with 32 arrivals/warp under racecheck. The same
// shared-buffer reuse protocol is used in both builds; do not infer timing.
#ifndef MK_LITMUS_PARTIES
#define MK_LITMUS_PARTIES 32
#endif
static_assert(MK_LITMUS_PARTIES == 1 || MK_LITMUS_PARTIES == 32);
__global__ void litmus(int *out) {
    __shared__ kittens::semaphore ready, done;
    __shared__ int values[32];
    int lane=threadIdx.x%32;
    if(threadIdx.x==0) { kittens::init_semaphore(ready,MK_LITMUS_PARTIES); kittens::init_semaphore(done,MK_LITMUS_PARTIES); }
    __syncthreads();
    int sum=0;
    for(int i=0;i<100;++i) {
        if(threadIdx.x<32) {
            if(i) kittens::wait(done,(i-1)%2);
            values[lane]=i;
            __syncwarp();
            if(lane<MK_LITMUS_PARTIES) kittens::arrive(ready);
        } else {
            kittens::wait(ready,i%2);
            sum+=values[lane];
            __syncwarp();
            if(lane<MK_LITMUS_PARTIES) kittens::arrive(done);
        }
    }
    if(threadIdx.x==32) *out=sum;
}
int main(){int *out;cudaMallocManaged(&out,4);litmus<<<1,64>>>(out);auto e=cudaDeviceSynchronize();printf("%s %d\n",cudaGetErrorString(e),*out);int r=*out!=4950;cudaFree(out);return r;}
