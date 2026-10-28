// Independent read-only LPDDR throughput probe; random data, larger than L2.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
void check(cudaError_t r){if(r!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(r));std::exit(1);}}
__device__ uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);}
__global__ void initialize(uint4* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]=make_uint4(mix(4*i),mix(4*i+1),mix(4*i+2),mix(4*i+3));}
__global__ void stream_read(const uint4* p,size_t n,uint32_t* out){
 uint32_t x=0,y=0,z=0,w=0;
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){uint4 a=p[i];x^=a.x;y^=a.y;z^=a.z;w^=a.w;}
 x^=y^z^w;
 for(int o=16;o;o/=2)x^=__shfl_down_sync(0xffffffff,x,o);
 __shared__ uint32_t partials[16];
 if((threadIdx.x&31)==0)partials[threadIdx.x/32]=x;
 __syncthreads();
 if(threadIdx.x==0){uint32_t sum=0;for(int i=0;i<16;i++)sum^=partials[i];out[blockIdx.x]=sum;}
}
int main(int argc,char** argv){
 const size_t bytes=argc>1?std::strtoull(argv[1],nullptr,10):2471628800ULL;
 if(!bytes||bytes%16){fprintf(stderr,"byte count must be positive and divisible by 16\n");return 3;}
 const size_t n=bytes/sizeof(uint4);uint4* data;uint32_t* out;check(cudaMalloc(&data,bytes));check(cudaMalloc(&out,320*sizeof(uint32_t)));
 initialize<<<320,256>>>(data,n);check(cudaGetLastError());check(cudaDeviceSynchronize());cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
 bool first=true;uint32_t expected=0;
 for(int blocks: {20,80,320}){
  for(int i=0;i<5;i++)stream_read<<<blocks,512>>>(data,n,out);check(cudaDeviceSynchronize());std::vector<float> samples;
  for(int i=0;i<30;i++){check(cudaEventRecord(start));stream_read<<<blocks,512>>>(data,n,out);check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,start,end));samples.push_back(ms);}
  std::vector<uint32_t> values(blocks);check(cudaMemcpy(values.data(),out,blocks*sizeof(uint32_t),cudaMemcpyDeviceToHost));uint32_t checksum=0;for(auto v:values)checksum^=v;
  if(!first&&checksum!=expected){fprintf(stderr,"checksum mismatch\n");return 2;}first=false;expected=checksum;
  std::sort(samples.begin(),samples.end());float median=(samples[14]+samples[15])/2;printf("{\"bytes\":%zu,\"blocks\":%d,\"threads\":512,\"median_ms\":%.6f,\"GB_s\":%.6f,\"checksum\":%u,\"samples_ms\":[",bytes,blocks,median,bytes/(median*1e6),checksum);
  for(size_t i=0;i<samples.size();i++)printf("%s%.6f",i?",":"",samples[i]);puts("]}");
 }
 check(cudaFree(out));check(cudaFree(data));return 0;
}
