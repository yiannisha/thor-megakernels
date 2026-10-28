#define NV_STAGES 2
#include "../../tools/analysis/nvfp4/primitive.cu"
#include "ops/nvfp4_linear.cuh"
#include <vector>
namespace fvm=megakernel::nvfp4;
struct VMPlan {
 fvm::Globals g;
 VMPlan():g{kittens::make_gl<megakernel::instruction_layout<fvm::Config>>(0,1,1,1,fvm::Config::INSTRUCTION_WIDTH),
 kittens::make_gl<megakernel::timing_layout<fvm::Config>>(0,1,1,1,fvm::Config::TIMING_WIDTH),nullptr,nullptr,nullptr,0,false,false,false,0,0}{}
 int length,slots;
 fvm::Operand* operands=nullptr;
 int *instructions=nullptr,*timings=nullptr,*ready=nullptr,*done=nullptr;
};
extern "C" void* vm_prepare(void** handles,void** input,void** packed,void** scales,void** output,
 float* tensor_scales,float* alpha,int length,int workers,int dependent,int quantize,int insert_noops,int quant_workers,int fused_quant){
 if(length<=0 || workers<=0 || workers>20 || quant_workers<0 || (fused_quant && (!quantize || !dependent))){snprintf(error_text,512,"invalid VM dimensions");return nullptr;}
 auto* p=new VMPlan;p->length=length;p->slots=length*(1+bool(insert_noops))+(quantize?(fused_quant?1:length):0);
 std::vector<fvm::Operand> ops;
 for(int i=0;i<length;++i){auto* h=(Plan*)handles[i];ops.push_back({h->gemm.params().mainloop,h->gemm.params().problem_shape,(__nv_bfloat16*)input[i],(__nv_bfloat16*)output[i],
  (unsigned char*)packed[i],(unsigned char*)scales[i],h->m,h->k,tensor_scales[i],alpha[i]});}
 std::vector<int> instructions(workers*p->slots*fvm::Config::INSTRUCTION_WIDTH,0);
 for(auto& o:ops)if(o.m<=0 || o.m%8 || o.k<=0 || o.k%256 ||
    !std::isfinite(o.input_tensor_scale) || o.input_tensor_scale<=0 || !std::isfinite(o.alpha)){
  snprintf(error_text,512,"invalid aligned native FP4 operand dimensions/scales");delete p;return nullptr;
 }
 for(int i=0;i+1<length;++i)if(fused_quant && (ops[i].m!=ops[i+1].k || ops[i].m%16 || ops[i].output!=ops[i+1].input)){
  snprintf(error_text,512,"fused quantization requires aligned matching adjacent dimensions");delete p;return nullptr;
 }
 for(int w=0;w<workers;++w){int slot=0;for(int i=0;i<length;++i){
  auto at=[&](int slot_index){return instructions.data()+(w*p->slots+slot_index)*fvm::Config::INSTRUCTION_WIDTH;};
  if(quantize && (!fused_quant || i==0)){at(slot)[0]=fvm::Quantize::opcode;at(slot)[1]=i;++slot;}
  int tiles=(ops[i].m+127)/128;int* inst=at(slot++);inst[0]=fvm::Linear::opcode;inst[1]=i;
  inst[2]=tiles*w/workers;inst[3]=tiles*(w+1)/workers;
  if(insert_noops)++slot;
 }}
 auto check=[&](cudaError_t e){if(e!=cudaSuccess){snprintf(error_text,512,"VM setup: %s",cudaGetErrorString(e));throw e;}};
 try{
  check(cudaMalloc(&p->operands,ops.size()*sizeof(fvm::Operand)));check(cudaMemcpy(p->operands,ops.data(),ops.size()*sizeof(fvm::Operand),cudaMemcpyHostToDevice));
  check(cudaMalloc(&p->instructions,instructions.size()*4));check(cudaMemcpy(p->instructions,instructions.data(),instructions.size()*4,cudaMemcpyHostToDevice));
  check(cudaMalloc(&p->timings,workers*p->slots*fvm::Config::TIMING_WIDTH*4));
  check(cudaMalloc(&p->ready,length*4));check(cudaMalloc(&p->done,length*4));
  p->g={kittens::make_gl<megakernel::instruction_layout<fvm::Config>>((uint64_t)p->instructions,1,workers,p->slots,fvm::Config::INSTRUCTION_WIDTH),
   kittens::make_gl<megakernel::timing_layout<fvm::Config>>((uint64_t)p->timings,1,workers,p->slots,fvm::Config::TIMING_WIDTH),
   p->operands,p->ready,p->done,workers,bool(dependent),bool(quantize),bool(fused_quant),length,quant_workers?std::min(quant_workers,workers):workers};
  auto kernel=megakernel::mk<fvm::Config,fvm::Globals,fvm::Linear,fvm::Quantize>;
  cudaDeviceProp prop;int device;check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));
  if(prop.major!=11 || prop.minor!=0 || !prop.cooperativeLaunch || workers>prop.multiProcessorCount){
   snprintf(error_text,512,"native FP4 VM requires Thor cooperative launch and workers <= physical SM count");throw cudaErrorInvalidDevice;
  }
  check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,fvm::Config::DYNAMIC_SHARED_MEMORY));
  int occupancy;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,fvm::Config::NUM_THREADS,fvm::Config::DYNAMIC_SHARED_MEMORY));
  if(occupancy!=1){snprintf(error_text,512,"native TMEM VM requires exactly one CTA/SM");throw cudaErrorInvalidConfiguration;}
 }catch(...){cudaFree(p->operands);cudaFree(p->instructions);cudaFree(p->timings);cudaFree(p->ready);cudaFree(p->done);delete p;return nullptr;}
 return p;
}
extern "C" int vm_run(void* handle,void* stream){
 auto* p=(VMPlan*)handle;auto st=(cudaStream_t)stream;
 auto e=cudaMemsetAsync(p->ready,0,p->length*4,st);if(e!=cudaSuccess)return e;
 e=cudaMemsetAsync(p->done,0,p->length*4,st);if(e!=cudaSuccess)return e;
 auto kernel=megakernel::mk<fvm::Config,fvm::Globals,fvm::Linear,fvm::Quantize>;void* args[]={&p->g};
 return cudaLaunchCooperativeKernel((const void*)kernel,p->g.workers,fvm::Config::NUM_THREADS,args,fvm::Config::DYNAMIC_SHARED_MEMORY,st);
}
extern "C" void vm_destroy(void* handle){auto* p=(VMPlan*)handle;if(p){cudaFree(p->operands);cudaFree(p->instructions);cudaFree(p->timings);cudaFree(p->ready);cudaFree(p->done);delete p;}}
