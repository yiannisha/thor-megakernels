#include "../../demos/low-latency-llama/llama.cuh"
#include "../../demos/low-latency-llama/attention_partial.cu"
#include "vm.cu"
#include "pyutils/pyutils.cuh"
#include <pybind11/stl.h>

using Legacy=llama_1b_globals;
struct Trace {__nv_bfloat16 *input,*gate,*residual,*output;};
struct ModelGlobals : fvm::Globals { Legacy legacy; Trace* trace; };
using ModelState=megakernel::state<fvm::Config>;
using OldAttention=attention_partial<fvm::Config,Legacy>;
struct ModelAttention {
 static constexpr int opcode=2;
 struct controller {
  __device__ static int release_lid(const ModelGlobals&,fvm::Config::instruction_t&,int& q){return (q+2)%fvm::Config::NUM_PAGES;}
  __device__ static int init_semaphores(const ModelGlobals& g,ModelState& s){return OldAttention::controller::init_semaphores(g.legacy,s);}
 };
 struct loader {__device__ static void run(const ModelGlobals& g,ModelState& s){OldAttention::loader::run(g.legacy,s);}};
 struct launcher {__device__ static void run(const ModelGlobals& g,ModelState& s){OldAttention::launcher::run(g.legacy,s);}};
 struct consumer {__device__ static void run(const ModelGlobals& g,ModelState& s){OldAttention::consumer::run(g.legacy,s);}};
 struct storer {__device__ static void run(const ModelGlobals& g,ModelState& s){OldAttention::storer::run(g.legacy,s);}};
};
struct ModelQuantize : fvm::Quantize {
 struct consumer {
  __device__ static void run(const ModelGlobals& g,ModelState& s){
   int index=s.instruction()[1];
   if(index<64 && index%4==1){
    if(kittens::warpid()==0 && kittens::laneid()==0)
     megakernel::gmem_wait<fvm::Config>([&]{return megakernel::gmem_load_acquire(&g.legacy.Bar[{index/4,2,0}])<32;});
    kittens::group<4>::sync(4);
   }
   fvm::Quantize::consumer::run(g,s);
  }
 };
 struct storer {
  __device__ static void run(const ModelGlobals& g,ModelState& s){
   if(g.trace && megakernel::get_worker_id()==0){
    kittens::wait(fvm::sem(s,0),0);int index=s.instruction()[1];auto& o=g.operands[index];
    for(int c=kittens::laneid();c<o.k;c+=32){
     g.trace[index].input[c]=o.input[c];
     if(o.input_gate)g.trace[index].gate[c]=o.input_gate[c];
    }
    if(o.residual)for(int c=kittens::laneid();c<o.m;c+=32)g.trace[index].residual[c]=o.residual[c];
    __threadfence();__syncwarp();
   }
   fvm::Quantize::storer::run(g,s);
  }
 };

};
struct ModelLinear : fvm::Linear {
 struct storer {
  __device__ static void run(const ModelGlobals& g,ModelState& s){
   if(g.trace){
    kittens::wait(fvm::sem(s,6),0);int index=s.instruction()[1];auto& o=g.operands[index];
    int begin=s.instruction()[2]*128,end=min(s.instruction()[3]*128,o.m);
    for(int c=begin+kittens::laneid();c<end;c+=32)g.trace[index].output[c]=o.output[c];
    __threadfence();__syncwarp();
   }
   fvm::Linear::storer::run(g,s);
  }
 };
};
struct RopeAppend : megakernel::NoOp<fvm::Config> {
 static constexpr int opcode=10;
 struct consumer {
  __device__ static void run(const ModelGlobals& g,ModelState& s){
   int layer=s.instruction()[1];
   if(kittens::laneid()==0)megakernel::gmem_wait<fvm::Config>([&]{return megakernel::gmem_load_acquire(g.done+layer*4)<g.workers;});
   __syncwarp();
   int head=megakernel::get_worker_id()*4+kittens::warpid();int lane=kittens::laneid();
   if(head>=48)return;
   const auto* input=g.operands[layer*4].output;
   for(int c=lane;c<64;c+=32){
    float v=__bfloat162float(input[head*64+c]);
    if(head<40){
     float pair=__shfl_xor_sync(0xffffffff,v,1);
     v=v*g.legacy.rope_cos[{g.legacy.pos_id,c}]+((c&1)?pair:-pair)*g.legacy.rope_sin[{g.legacy.pos_id,c}];
    }
    auto bf=__float2bfloat16(v);
    if(head<32)g.legacy.q_post_rope[{head*64+c}]=kittens::bf16(bf);
    else if(head<40)g.legacy.k_cache[{layer,int(g.legacy.pos_id),head-32,c}]=kittens::bf16(bf);
    else g.legacy.v_cache[{layer,int(g.legacy.pos_id),head-40,c}]=kittens::bf16(bf);
   }
   __threadfence();__syncwarp();
   if(lane==0)megakernel::gmem_add_release(&g.legacy.Bar[{layer,0,head}],4);
  }
 };
};
struct ModelPlan {VMPlan* vm;Legacy legacy;Trace* trace=nullptr;int* attention_codes=nullptr;};
static auto model_kernel=megakernel::mk<fvm::Config,ModelGlobals,ModelLinear,ModelQuantize,RopeAppend,ModelAttention>;

template<class... Members>void bind_model(pybind11::module_& m,Members... members){
 m.def("configure",[](uintptr_t handle,std::vector<uintptr_t> norms,kittens::py::object<Members>... args){
  auto* vm=(VMPlan*)handle;
  if(vm->length!=65 || norms.size()!=33)throw std::runtime_error("expected the 16-layer Llama operand plan");
  Legacy legacy{kittens::py::from_object<typename kittens::py::trait<Members>::member_type>::make(args)...};
  auto* p=new ModelPlan{vm,legacy};
  std::vector<fvm::Operand> ops(65);cudaMemcpy(ops.data(),vm->operands,sizeof(fvm::Operand)*65,cudaMemcpyDeviceToHost);
  for(int i=0;i<65;++i){
   if(i==64 || i%4==0 || i%4==2){ops[i].transform=1;ops[i].norm=(__nv_bfloat16*)norms[i==64?32:(i/4)*2+(i%4==2)];ops[i].norm_eps=legacy.rms_norm_eps;}
   if(i<64 && i%4==3){ops[i].transform=2;ops[i].input_gate=ops[i].input+8192;}
   if(i<64 && (i%4==1 || i%4==3))ops[i].residual=ops[i].output;
  }
  cudaMemcpy(vm->operands,ops.data(),sizeof(fvm::Operand)*65,cudaMemcpyHostToDevice);
  int slots=16*10+2;std::vector<int> code(20*slots*32,0);
  for(int w=0;w<20;++w){int slot=0;auto ins=[&](){return code.data()+(w*slots+slot++)*32;};
   auto quant=[&](int i){int* a=ins();a[0]=fvm::Quantize::opcode;a[1]=i;};
   auto linear=[&](int i){int* a=ins();a[0]=fvm::Linear::opcode;a[1]=i;int tiles=(ops[i].m+127)/128;a[2]=tiles*w/20;a[3]=tiles*(w+1)/20;};
   for(int l=0;l<16;++l){quant(l*4);linear(l*4);int* rope=ins();rope[0]=10;rope[1]=l;
    int* attn=ins();if(w<8){attn[0]=2;attn[1]=l;attn[2]=w;attn[3]=1;attn[4]=0;}
    quant(l*4+1);linear(l*4+1);quant(l*4+2);linear(l*4+2);quant(l*4+3);linear(l*4+3);
   }quant(64);linear(64);
   if(slot!=slots)throw std::runtime_error("model instruction count mismatch");
  }
  cudaFree(vm->instructions);cudaFree(vm->timings);
  cudaMalloc(&vm->instructions,code.size()*4);cudaMemcpy(vm->instructions,code.data(),code.size()*4,cudaMemcpyHostToDevice);
  cudaMalloc(&vm->timings,20*slots*128*4);
  vm->g.instructions=kittens::make_gl<megakernel::instruction_layout<fvm::Config>>((uint64_t)vm->instructions,1,20,slots,32);
  vm->g.timings=kittens::make_gl<megakernel::timing_layout<fvm::Config>>((uint64_t)vm->timings,1,20,slots,128);
  cudaFuncSetAttribute(model_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,fvm::Config::DYNAMIC_SHARED_MEMORY);
  int occupancy;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,model_kernel,256,fvm::Config::DYNAMIC_SHARED_MEMORY);
  if(occupancy!=1)throw std::runtime_error("model TMEM residency admission failed");
  std::vector<int> attention_codes(16*20*32,0);
  for(int layer=0;layer<16;++layer)for(int w=0;w<8;++w){int* ins=attention_codes.data()+(layer*20+w)*32;
   ins[0]=2;ins[1]=layer;ins[2]=w;ins[3]=1;ins[4]=0;}
  cudaMalloc(&p->attention_codes,attention_codes.size()*4);
  cudaMemcpy(p->attention_codes,attention_codes.data(),attention_codes.size()*4,cudaMemcpyHostToDevice);
  return (uintptr_t)p;
 });
 m.def("attention",[](uintptr_t handle,int layer,int position,uintptr_t stream){
  if(layer<0 || layer>=16)throw std::runtime_error("invalid attention layer");
  auto* p=(ModelPlan*)handle;p->legacy.pos_id=position;
  auto base=p->vm->g;
  base.instructions=kittens::make_gl<megakernel::instruction_layout<fvm::Config>>((uint64_t)(p->attention_codes+layer*20*32),1,20,1,32);
  base.timings=kittens::make_gl<megakernel::timing_layout<fvm::Config>>((uint64_t)p->vm->timings,1,20,1,128);
  ModelGlobals g{base,p->legacy,nullptr};void* args[]={&g};
  auto kernel=megakernel::mk<fvm::Config,ModelGlobals,ModelAttention>;
  cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,fvm::Config::DYNAMIC_SHARED_MEMORY);
  auto error=cudaLaunchCooperativeKernel((const void*)kernel,20,256,args,fvm::Config::DYNAMIC_SHARED_MEMORY,(cudaStream_t)stream);
  if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
 });
 m.def("set_trace",[](uintptr_t handle,std::vector<std::vector<uintptr_t>> pointers){
  auto* p=(ModelPlan*)handle;std::vector<Trace> trace;
  if(pointers.size()!=65)throw std::runtime_error("expected 65 operand traces");
  for(auto& v:pointers){if(v.size()!=4)throw std::runtime_error("expected four trace pointers");
   trace.push_back({(__nv_bfloat16*)v[0],(__nv_bfloat16*)v[1],(__nv_bfloat16*)v[2],(__nv_bfloat16*)v[3]});}
  if(!p->trace)cudaMalloc(&p->trace,65*sizeof(Trace));
  auto e=cudaMemcpy(p->trace,trace.data(),65*sizeof(Trace),cudaMemcpyHostToDevice);
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
 });
 m.def("run",[](uintptr_t handle,int position,uintptr_t stream){auto* p=(ModelPlan*)handle;auto st=(cudaStream_t)stream;
  cudaMemsetAsync(p->vm->ready,0,65*4,st);cudaMemsetAsync(p->vm->done,0,65*4,st);
  p->legacy.pos_id=position;ModelGlobals g{p->vm->g,p->legacy,p->trace};void* args[]={&g};
  auto error=cudaLaunchCooperativeKernel((const void*)model_kernel,20,256,args,fvm::Config::DYNAMIC_SHARED_MEMORY,st);
  if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
 });
}
PYBIND11_MODULE(mk_fp4_model,m){
 bind_model(m,&Legacy::Bar,&Legacy::instructions,&Legacy::timings,
 &Legacy::qkv_weights,&Legacy::attn_norm_weights,&Legacy::o_weights,&Legacy::mlp_norm_weights,
 &Legacy::up_weights,&Legacy::gate_weights,&Legacy::down_weights,&Legacy::lm_head_norm_weights,&Legacy::lm_head_weights,
 &Legacy::k_cache,&Legacy::v_cache,&Legacy::rope_cos,&Legacy::rope_sin,
 &Legacy::hidden_states,&Legacy::q_post_rope,&Legacy::attn_out,&Legacy::attn_lse_intermediates,&Legacy::attn_out_intermediates,
 &Legacy::silu_out,&Legacy::logits,&Legacy::pos_id,&Legacy::attn_scale,&Legacy::rms_norm_eps,&Legacy::skip_attn_reduction);
}
