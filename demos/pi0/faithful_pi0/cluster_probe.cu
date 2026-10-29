#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace cg=cooperative_groups;

__global__ __cluster_dims__(2,2,1) void probe(unsigned* output){
    auto cluster=cg::this_cluster();
    __shared__ unsigned value;
    extern __shared__ unsigned scratch[];
    unsigned rank=cluster.block_rank();
    if(threadIdx.x==0){value=rank;scratch[0]=rank+100;}
    cluster.sync();
    unsigned id=blockIdx.x+gridDim.x*blockIdx.y;
    if(threadIdx.x==0){
        output[id*4]=rank;
        output[id*4+1]=*cluster.map_shared_rank(&value,rank^1);
        output[id*4+2]=cg::this_grid().is_valid();
        output[id*4+3]=scratch[0];
    }
    cluster.sync();
    cg::this_grid().sync();
}
int main(int argc,char** argv){
    int smem=argc>1?atoi(argv[1]):213*1024;
    auto check=[](cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(1);}};
    int device;check(cudaGetDevice(&device));cudaDeviceProp p;check(cudaGetDeviceProperties(&p,device));
    check(cudaFuncSetAttribute(probe,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeCooperative;attr.val.cooperative=1;
    cudaLaunchConfig_t config{};config.blockDim=dim3(256);config.gridDim=dim3(2,2);
    config.dynamicSmemBytes=smem;config.attrs=&attr;config.numAttrs=1;
    int capacity;check(cudaOccupancyMaxActiveClusters(&capacity,probe,&config));
    int clusters=capacity<p.multiProcessorCount/4?capacity:p.multiProcessorCount/4;
    if(clusters<1)return 2;
    config.gridDim=dim3(2*clusters,2);
    unsigned* out;check(cudaMalloc(&out,clusters*4*4*sizeof(unsigned)));
    check(cudaLaunchKernelEx(&config,probe,out));check(cudaDeviceSynchronize());
    std::vector<unsigned> values(clusters*16);check(cudaMemcpy(values.data(),out,values.size()*4,cudaMemcpyDeviceToHost));
    for(int i=0;i<clusters*4;++i){auto* v=values.data()+i*4;if(v[1]!=(v[0]^1)||v[2]!=1||v[3]!=100+v[0])return 3;}
    check(cudaFree(out));
    printf("{\"sms\":%d,\"cluster\":[2,2,1],\"resident_clusters\":%d,\"workers\":%d,\"dynamic_smem\":%d,\"cooperative_grid_sync\":true,\"cluster_shared_peer_read\":true}\n",p.multiProcessorCount,clusters,clusters*4,smem);
}
