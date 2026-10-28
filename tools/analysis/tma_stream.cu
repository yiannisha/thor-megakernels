// Diagnostic TMA reader: native 16x512 BF16 tiles, four tiles per 64KiB
// iteration, fixed 13-page allocation and 20 resident CTAs. No model math.
#include "kittens.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
using Tile=kittens::st_bf<16,512>;
using GL=kittens::gl<kittens::bf16,1,1,-1,2048,Tile>;
#ifndef TMA_DIAG_TILES
#define TMA_DIAG_TILES 37712
#endif
#ifndef TMA_DIAG_WARMUPS
#define TMA_DIAG_WARMUPS 5
#endif
#ifndef TMA_DIAG_REPEATS
#define TMA_DIAG_REPEATS 30
#endif
constexpr int TILES=TMA_DIAG_TILES,THREADS=640,CONSUMERS=512,ALLOC=13*16384+1023;
void check(cudaError_t r){if(r!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(r));std::exit(1);}}
__device__ uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
__global__ void initialize(uint4* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]=make_uint4(mix(4*i),mix(4*i+1),mix(4*i+2),mix(4*i+3));}
__global__ void reference(const uint4* p,size_t n,uint32_t* out){uint32_t x=0;for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){auto a=p[i];x^=a.x^a.y^a.z^a.w;}out[blockIdx.x*blockDim.x+threadIdx.x]=x;}
template<int STAGES,int ISSUERS,bool READ> __maxnreg__(96) __global__ void stream(const __grid_constant__ GL g,uint32_t* output){
 __shared__ kittens::semaphore arrived[STAGES],finished[STAGES];
 extern __shared__ unsigned char raw[];
 auto* pages=reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(raw)+1023)&~uintptr_t(1023));
 if(threadIdx.x<STAGES){kittens::init_semaphore(arrived[threadIdx.x],1);kittens::init_semaphore(finished[threadIdx.x],CONSUMERS);}
 asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();
 int count=(TILES+gridDim.x-1-blockIdx.x)/gridDim.x;
 if(threadIdx.x>=512&&threadIdx.x<544){
  int lane=threadIdx.x-512;
  for(int i=0;i<count;i++){
   int stage=i%STAGES;auto& sem=arrived[stage];
   if(lane==0)kittens::wait(finished[stage],!((i/STAGES)&1));
   __syncwarp();
   if(lane==0)kittens::tma::expect_bytes(sem,65536);
   __syncwarp();
   if(lane<ISSUERS)for(int chunk=lane;chunk<4;chunk+=ISSUERS){
    auto& tile=*reinterpret_cast<Tile*>(pages+stage*65536+chunk*16384);
    kittens::tma::load_async<kittens::dim::ROW,kittens::cache_policy::EVICT_FIRST>(tile,g,{0,int(blockIdx.x+i*gridDim.x),chunk},sem);
   }
   __syncwarp();
  }
 }else if(threadIdx.x<CONSUMERS){
  uint32_t sum=0;
  for(int i=0;i<count;i++){
   int stage=i%STAGES;kittens::wait(arrived[stage],(i/STAGES)&1);
   if constexpr(READ){
    const auto* p=reinterpret_cast<const uint4*>(pages+stage*65536);
    for(int j=threadIdx.x;j<65536/16;j+=CONSUMERS){auto a=p[j];sum^=a.x^a.y^a.z^a.w;}
   }
   kittens::arrive(finished[stage]);
  }
  output[blockIdx.x*CONSUMERS+threadIdx.x]=sum;
 }
}
template<int S,int I,bool R> void run(GL g,uint32_t* out,uint32_t expected){
 auto kernel=stream<S,I,R>;check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ALLOC));cudaFuncAttributes attr;check(cudaFuncGetAttributes(&attr,kernel));int resident;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,THREADS,ALLOC));if(resident<1){puts("no resident CTA");std::exit(3);}
 auto launch=[&]{kernel<<<20,THREADS,ALLOC>>>(g,out);check(cudaGetLastError());};
 for(int n=0;n<TMA_DIAG_WARMUPS;n++)launch();check(cudaDeviceSynchronize());cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));std::vector<float> times;
 for(int n=0;n<TMA_DIAG_REPEATS;n++){check(cudaEventRecord(a));launch();check(cudaEventRecord(b));check(cudaEventSynchronize(b));float ms;check(cudaEventElapsedTime(&ms,a,b));times.push_back(ms);}
 std::vector<uint32_t> result(20*CONSUMERS);check(cudaMemcpy(result.data(),out,result.size()*4,cudaMemcpyDeviceToHost));uint32_t checksum=0;for(auto x:result)checksum^=x;if(R&&checksum!=expected){fprintf(stderr,"checksum %u != %u\n",checksum,expected);std::exit(2);}
 std::sort(times.begin(),times.end());float median=(times[(times.size()-1)/2]+times[times.size()/2])/2;size_t bytes=size_t(TILES)*65536;
 printf("{\"stages\":%d,\"issuers\":%d,\"consumer_reads\":%s,\"bytes\":%zu,\"median_ms\":%.6f,\"GB_s\":%.6f,\"checksum\":%u,\"checksum_checked\":%s,\"resident_ctas_per_sm\":%d,\"registers\":%d,\"dynamic_smem\":%d,\"samples_ms\":[",S,I,R?"true":"false",bytes,median,bytes/(median*1e6),checksum,R?"true":"false",resident,attr.numRegs,ALLOC);
 for(size_t n=0;n<times.size();n++)printf("%s%.6f",n?",":"",times[n]);puts("]}");check(cudaEventDestroy(a));check(cudaEventDestroy(b));
}
int main(int argc,char** argv){
 size_t bytes=size_t(TILES)*65536;uint4* data;uint32_t* out;check(cudaMalloc(&data,bytes));check(cudaMalloc(&out,20*CONSUMERS*4));initialize<<<320,256>>>(data,bytes/16);reference<<<20,512>>>(data,bytes/16,out);check(cudaDeviceSynchronize());std::vector<uint32_t> ref(20*CONSUMERS);check(cudaMemcpy(ref.data(),out,ref.size()*4,cudaMemcpyDeviceToHost));uint32_t expected=0;for(auto x:ref)expected^=x;
 auto g=kittens::make_gl<GL>((uint64_t)data,1,1,TILES*16,2048);
 using Fn=void(*)(GL,uint32_t*,uint32_t);
 std::vector<Fn> cases={run<3,1,true>,run<1,1,true>,run<2,1,true>,run<3,2,true>,run<3,4,true>,run<3,1,false>,run<1,1,false>,run<2,1,false>,run<3,2,false>,run<3,4,false>};
 std::mt19937 rng(argc>1?std::strtoul(argv[1],nullptr,10):71);std::shuffle(cases.begin(),cases.end(),rng);
 for(auto fn:cases)fn(g,out,expected);
 check(cudaFree(out));check(cudaFree(data));
}
