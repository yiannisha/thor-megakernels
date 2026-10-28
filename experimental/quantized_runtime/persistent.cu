// Experimental generic cooperative instruction loop using CUTLASS collectives.
// Separate from the frozen BF16 VM: no model-specific math or shape constants.
#define NV_STATIC_RUNTIME
#include "../../tools/analysis/nvfp4/primitive.cu"
#include <cooperative_groups.h>
#include <vector>

struct QuantInstruction {
 const __nv_bfloat16* input;
 unsigned char* packed;
 unsigned char* native;
 int m,k;
 float tensor_scale;
};
__device__ void quantize_vector(const QuantInstruction& q) {
 int lane=threadIdx.x%32;
 auto layout=SFConfig::tile_atom_to_shape_SFB(make_shape(q.m,1,q.k,1));
 for(int group=threadIdx.x/32;group<q.k/16;group+=blockDim.x/32){
  int col=group*16;
  float v=lane<16?__bfloat162float(q.input[col+lane])/q.tensor_scale:0.f;
  float a=fabsf(v);
  for(int d=16;d;d>>=1) a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
  __nv_fp8_e4m3 sf(fmaxf(a/6.f,1.f/512.f));
  unsigned char value=__nv_cvt_float_to_fp4(v/float(sf),__NV_E2M1,cudaRoundNearest);
  if(lane==0) q.native[layout(make_coord(0,col,0))]=sf.__x;
  unsigned char next=__shfl_down_sync(0xffffffff,unsigned(value),1);
  if(lane<16 && lane%2==0) q.packed[(col+lane)/2]=value|(next<<4);
 }
}
__global__ __launch_bounds__(Kernel::MaxThreadsPerBlock,1)
void persistent_linear(const Kernel::Params* instructions,const QuantInstruction* quant,
 int length,int loops,bool quantize_inputs) {
 extern __shared__ __align__(128) char smem[];
 auto grid=cooperative_groups::this_grid();
 for(int l=0;l<loops;++l) for(int i=0;i<length;++i){
  if(quantize_inputs){
   if(blockIdx.x==0) quantize_vector(quant[i]);
   asm volatile("fence.proxy.async.global;" ::: "memory");
   grid.sync();
   // Publish scalar shared-memory staging writes to the async proxy happens
   // within CUTLASS, after its global-to-shared TMA loads.
  }
  Kernel{}(instructions[i],smem);
  // A kernel boundary normally drains global TMA stores. A thread/grid
  // barrier alone cannot replace that completion guarantee between ops.
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  grid.sync();
 }
 // MMA is warp zero in the pinned collective. Repeated allocations must use
 // the same warp; relinquishing the permit is terminal for this CTA.
 if(threadIdx.x<32) cute::TMEM::Allocator1Sm{}.release_allocation_lock();
}
struct RuntimePlan {
 Kernel::Params* instructions=nullptr;
 QuantInstruction* quant=nullptr;
 int length=0,workers=0,allocated_smem=0;
};
extern "C" void* nv_runtime_prepare(void** plans,void** inputs,void** packed,
 void** scales,float* tensor_scales,int length) {
 auto* runtime=new RuntimePlan;
 std::vector<Kernel::Params> parameters;
 std::vector<QuantInstruction> quant;
 for(int i=0;i<length;++i){
  auto* p=(Plan*)plans[i];
  parameters.push_back(p->gemm.params());
  parameters.back().retain_tmem_permit=true;
  quant.push_back({(__nv_bfloat16*)inputs[i],(unsigned char*)packed[i],
   (unsigned char*)scales[i],p->m,p->k,tensor_scales[i]});
 }
 cudaDeviceProp prop;cudaGetDeviceProperties(&prop,0);
 runtime->workers=prop.multiProcessorCount;runtime->length=length;
 // The collective assumes every admitted CTA owns an initial MMA tile. The
 // static scheduler's ordinary launch caps the grid in the same way.
 for(int i=0;i<length;++i) runtime->workers=std::min(runtime->workers,(((Plan*)plans[i])->m+127)/128);
 // Retained TMEM permits require one resident CTA per SM. Limit the grid
 // and reserve more than half the per-SM shared memory even when reducing
 // pipeline depth; occupancy admission must prove this, not assume spreading.
 runtime->allocated_smem=std::max(int(Kernel::SharedStorageSize),
   int((prop.sharedMemPerMultiprocessor/2+128)/128*128));
 auto kernel=persistent_linear;
 auto error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,runtime->allocated_smem);
 int occupancy=0;
 if(error==cudaSuccess) error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,
   Kernel::MaxThreadsPerBlock,runtime->allocated_smem);
 if(error!=cudaSuccess || occupancy!=1 || !prop.cooperativeLaunch){
  snprintf(error_text,512,"runtime resource admission failed: %s, occupancy=%d",cudaGetErrorString(error),occupancy);
  delete runtime;return nullptr;
 }
 error=cudaMalloc(&runtime->instructions,length*sizeof(Kernel::Params));
 if(error==cudaSuccess) error=cudaMalloc(&runtime->quant,length*sizeof(QuantInstruction));
 if(error==cudaSuccess) error=cudaMemcpy(runtime->instructions,parameters.data(),length*sizeof(Kernel::Params),cudaMemcpyHostToDevice);
 if(error==cudaSuccess) error=cudaMemcpy(runtime->quant,quant.data(),length*sizeof(QuantInstruction),cudaMemcpyHostToDevice);
 if(error!=cudaSuccess){
  snprintf(error_text,512,"runtime setup: %s",cudaGetErrorString(error));
  cudaFree(runtime->instructions);cudaFree(runtime->quant);delete runtime;return nullptr;
 }
 return runtime;
}
extern "C" int nv_runtime_run(void* handle,int loops,int quantize_inputs,void* stream){
 auto* r=(RuntimePlan*)handle; bool q=quantize_inputs;
 void* args[]={&r->instructions,&r->quant,&r->length,&loops,&q};
 return cudaLaunchCooperativeKernel((const void*)persistent_linear,r->workers,
  Kernel::MaxThreadsPerBlock,args,r->allocated_smem,(cudaStream_t)stream);
}
extern "C" void nv_runtime_destroy(void* handle){
 auto* r=(RuntimePlan*)handle;if(r){cudaFree(r->instructions);cudaFree(r->quant);delete r;}
}
extern "C" int nv_runtime_smem(){return Kernel::SharedStorageSize;}
extern "C" int nv_runtime_threads(){return Kernel::MaxThreadsPerBlock;}

extern "C" int nv_runtime_allocated_smem(void* handle){return ((RuntimePlan*)handle)->allocated_smem;}
