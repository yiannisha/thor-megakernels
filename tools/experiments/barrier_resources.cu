#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#define CHECK(x) do {auto s=(x);if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}while(0)
template<int ID> __global__ void barrier_probe(){asm volatile("bar.sync %0;"::"n"(ID):"memory");}
int main(){
    std::printf("{\"scope\":\"occupancy query only; kernels not launched\",\"threads\":384,\"dynamic_smem_bytes\":85103,\"variants\":[");
    int index=0;
    for(auto kernel:{barrier_probe<15>,barrier_probe<7>}){
        CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,85103));
        int resident;CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,384,85103));
        std::printf("%s{\"barrier_id\":%d,\"resident_ctas_per_sm\":%d}",index?",":"",index?7:15,resident);++index;
    }
    std::printf("]}\n");
}
