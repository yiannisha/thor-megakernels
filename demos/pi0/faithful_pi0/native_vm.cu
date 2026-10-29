// Full-CTA source collective opcode in the original static megakernel VM.
#ifdef PI0_UNIFIED
#include "../native_pi0/vision_fa2.cuh"
#endif
#include "static_scheduler.cuh"
#include "gemm_types_bf16.h"
#include "geglu_native_types.h"

#include "kittens.cuh"
#include "megakernel.cuh"
#include "pointwise.cuh"
#include "generic.cuh"
#include <vector>
using NativeKernel=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,
    sm100_fp16_k64::Main,sm100_fp16_k64::Epi,faithful::StaticSchedulerTag>;
using NativeOKernel=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,
    sm100_fp16_plain::Main,sm100_fp16_plain::Epi,faithful::StaticSchedulerTag>;
#ifdef PI0_UNIFIED
#include "unified.cuh"
using NativeDownKernel=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,
    coherent_down::Main,coherent_down::Epi,faithful::StaticSchedulerTag>;
#else
using NativeDownKernel=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,
    sm100_fp16_2sm21::Main,sm100_fp16_2sm21::Epi,faithful::StaticSchedulerTag>;
#endif
struct FaithfulConfig:megakernel::thor_config {
    #ifdef FAITHFUL_FULL_CTA
    static constexpr int NUM_CONSUMER_WARPS=12,NUM_WARPS=16,NUM_THREADS=512;
#else
    static constexpr int NUM_CONSUMER_WARPS=4,NUM_WARPS=8,NUM_THREADS=256;
#endif
#ifdef PI0_PHASE_ACTION
    static constexpr int WORKERS=20;
    static constexpr int CLUSTER_X=1,CLUSTER_Y=1,CLUSTER_Z=1,CLUSTER_BLOCKS=1;
    // Full TMEM ownership is a runtime resource that CUDA occupancy does not
    // account for. Reserve > half the SMEM capacity to force one CTA per SM.
    static constexpr int GENERIC_STORAGE=sizeof(faithful_generic::NT::SharedStorage)>sizeof(faithful_generic::NN::SharedStorage)?sizeof(faithful_generic::NT::SharedStorage):sizeof(faithful_generic::NN::SharedStorage);
    static constexpr int NUM_PAGES=1,PAGE_SIZE=GENERIC_STORAGE>131072?GENERIC_STORAGE:131072;
#else
    static constexpr int WORKERS=16;
    static constexpr int CLUSTER_X=2,CLUSTER_Y=2,CLUSTER_Z=1,CLUSTER_BLOCKS=4;
#ifdef PI0_PHASE_VISION
    static constexpr int GENERIC_STORAGE=sizeof(faithful_generic::NT::SharedStorage)>sizeof(faithful_generic::NN::SharedStorage)?sizeof(faithful_generic::NT::SharedStorage):sizeof(faithful_generic::NN::SharedStorage);
    static constexpr int PHASE_STORAGE=GENERIC_STORAGE>VisionTraits::kSmemSize?GENERIC_STORAGE:VisionTraits::kSmemSize;
    static constexpr int NUM_PAGES=1,PAGE_SIZE=PHASE_STORAGE>131072?PHASE_STORAGE:131072;
#else
    static constexpr int NUM_PAGES=1,PAGE_SIZE=NativeKernel::SharedStorageSize;
#endif
#endif
    static constexpr int ARENA_ALIGNMENT=1024,DYNAMIC_SMEM_ALIGNMENT=1024;
    static constexpr int DYNAMIC_SHARED_MEMORY=PAGE_SIZE;
    static constexpr int TIMING_WIDTH=1,DYNAMIC_SEMAPHORES=0,SCRATCH_BYTES=0;
    using timing_t=int[1];
    static constexpr int NATIVE_TMEM_COLUMNS=512;
#ifdef PI0_PHASE_ACTION
    static constexpr int TMEM_CTA_GROUP=1;
#else
    static constexpr int TMEM_CTA_GROUP=2;
#endif
    template<class G,class S> static __device__ void collective_controller(const G& g,S& s);
};
using State=megakernel::state<FaithfulConfig>;
struct Instructions {
    int* data;int length;
    __device__ int rows()const{return length;}
    __device__ int& operator[](const kittens::coord<>& c)const {
        return data[(size_t(megakernel::get_worker_id())*length+c.r)*32+c.c];
    }
};
struct Timings {
    int* unused=nullptr;
    __device__ int& operator[](const kittens::coord<>&)const { asm("trap;");return *unused; }
};
struct NativeStage {void const* params;int tactic;};
struct Globals {
    static constexpr bool DYNAMIC_INSTRUCTIONS=false;
    Instructions instructions;Timings timings;
    void const* params;int tactic;NativeStage const* stages=nullptr;
};
struct CollectiveOp {
    static constexpr int opcode=1;
    using controller=megakernel::NoOp<FaithfulConfig>::controller;
    static __device__ void run(const Globals& g,State& s) {
        // All VM roles, including the controller, enter the physical CUTLASS
        // warp roles together. The source kernel determines each warp's work.
        auto stage=g.stages?g.stages[s.instruction()[1]]:NativeStage{g.params,g.tactic};
        auto* params=stage.params;int tactic=stage.tactic;
        if(threadIdx.x==0){s.wait_page_ready(0);cute::faithful_pair_mode()=
#ifdef PI0_UNIFIED
            0;
#else
            (tactic==2);
#endif
        }
        __syncthreads();
#ifndef PI0_COHERENT_FAST_BARRIERS
        cooperative_groups::this_grid().sync();
#endif
        auto* arena=reinterpret_cast<char*>(s.pages[0].ptr());
        if(tactic>=10)faithful_pointwise::run(*static_cast<faithful_pointwise::Spec const*>(params),arena);
#if !defined(PI0_PHASE_ACTION) && !defined(PI0_PHASE_VISION)
        else if(tactic==0)NativeKernel{}(static_cast<NativeKernel::Params const*>(params)[0],arena,s.native_tmem_base);
        else if(tactic==1)NativeOKernel{}(static_cast<NativeOKernel::Params const*>(params)[0],arena,s.native_tmem_base);
        else if(tactic==2)NativeDownKernel{}(static_cast<NativeDownKernel::Params const*>(params)[0],arena,s.native_tmem_base);
#ifdef PI0_UNIFIED
        else if(tactic==7)GemmKernel{}(*static_cast<GemmKernel::Params const*>(params),arena,s.native_tmem_base);
#endif
#ifdef PI0_UNIFIED
        else if(tactic==3)coherent_geglu::run(*static_cast<coherent_geglu::Params const*>(params),arena,s.native_tmem_base);
#else
        else if(tactic==3)GemmKernel{}(static_cast<GemmKernel::Params const*>(params)[0],arena,s.native_tmem_base);
#endif
#endif
#if defined(PI0_UNIFIED) && !defined(PI0_PHASE_ACTION) && !defined(PI0_PHASE_VLM)
        else if(tactic==4)coherent_vision(*static_cast<CoherentVision const*>(params),arena);
#endif
        else if(tactic==5)faithful_generic::run(*static_cast<faithful_generic::Params<faithful_generic::NT> const*>(params),arena,s.native_tmem_base);
        else if(tactic==6)faithful_generic::run(*static_cast<faithful_generic::Params<faithful_generic::NN> const*>(params),arena,s.native_tmem_base);

        __syncthreads();
#ifdef PI0_UNIFIED
        // A normal kernel boundary retires distributed shared-memory access.
        // Persistent opcodes must establish that lifetime boundary explicitly.
        cooperative_groups::this_cluster().sync();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("fence.proxy.async.global;" ::: "memory");
#endif
        cooperative_groups::this_grid().sync();
        if(threadIdx.x==0)s.finish_page(0,FaithfulConfig::NUM_CONSUMER_WARPS);
        __syncthreads();
    }
    struct Role { static __device__ void run(const Globals& g,State& s){CollectiveOp::run(g,s);} };
    using loader=Role;using launcher=Role;using consumer=Role;using storer=Role;
};
template<class G,class S> __device__ void FaithfulConfig::collective_controller(const G& g,S& s) {
    // Zero is a whole-cluster idle instruction; other opcodes join all roles.
    if(s.instruction()[0]==1)CollectiveOp::run(g,s);
}
struct VMPlan {void* params=nullptr;int* instructions=nullptr;int count=0,tactic=0;NativeStage* stages=nullptr;std::vector<void*> allocations;
    ~VMPlan(){cudaFree(params);cudaFree(instructions);cudaFree(stages);for(auto* a:allocations)cudaFree(a);}};
template<class Kernel> int create_plan(void** handle,int tactic,void* a,void* b,void* out,int m,int n,int k,int count,void* stream,float alpha=1.f,float beta=0.f) {
    *handle=nullptr;if(count<=0)return -1;
    using G=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    auto sa=cutlass::make_cute_packed_stride(typename Kernel::StrideA{},{m,k,1});
    auto sb=cutlass::make_cute_packed_stride(typename Kernel::StrideB{},{n,k,1});
    auto sd=cutlass::make_cute_packed_stride(typename Kernel::StrideD{},{m,n,1});
    typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,1},
        {(typename G::ElementA*)a,sa,(typename G::ElementB*)b,sb},{{alpha,beta},(typename G::ElementC*)out,sd,(typename G::ElementD*)out,sd}};
    auto params=Kernel::to_underlying_arguments(args,nullptr);
    auto* p=new VMPlan;p->count=count;p->tactic=tactic;
    auto st=static_cast<cudaStream_t>(stream);
    cudaError_t e=cudaMalloc(&p->params,sizeof(params));
    if(!e)e=cudaMemcpyAsync(p->params,&params,sizeof(params),cudaMemcpyHostToDevice,st);
    std::vector<int> instructions(size_t(FaithfulConfig::WORKERS)*count*32,0);
    for(int worker=0;worker<FaithfulConfig::WORKERS;++worker)for(int i=0;i<count;++i)instructions[(worker*count+i)*32]=1;
    if(!e)e=cudaMalloc(&p->instructions,instructions.size()*sizeof(int));
    if(!e)e=cudaMemcpyAsync(p->instructions,instructions.data(),instructions.size()*sizeof(int),cudaMemcpyHostToDevice,st);
    if(!e)e=cudaStreamSynchronize(st);
    auto kernel=megakernel::mk<FaithfulConfig,Globals,CollectiveOp>;
    if(!e)e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,FaithfulConfig::DYNAMIC_SHARED_MEMORY);
    cudaFuncAttributes attr{};if(!e)e=cudaFuncGetAttributes(&attr,kernel);
    if(!e && attr.sharedSizeBytes+FaithfulConfig::DYNAMIC_SHARED_MEMORY>232448)e=cudaErrorInvalidConfiguration;
    if(e){delete p;return int(e);}
    *handle=p;return 0;
}
extern "C" int faithful_vm_run(void* handle,void* stream) {
    auto* p=static_cast<VMPlan*>(handle);
    Globals g{{p->instructions,p->count},{},p->params,p->tactic,p->stages};
    cudaLaunchConfig_t cfg{};
#ifdef PI0_PHASE_ACTION
    cfg.gridDim=dim3(FaithfulConfig::WORKERS,1,1);
#else
    cfg.gridDim=dim3(4,4,1);
#endif
    cfg.blockDim=dim3(FaithfulConfig::NUM_THREADS,1,1);
    cfg.dynamicSmemBytes=FaithfulConfig::DYNAMIC_SHARED_MEMORY;cfg.stream=static_cast<cudaStream_t>(stream);
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeCooperative;attr.val.cooperative=1;
    cfg.attrs=&attr;cfg.numAttrs=1;
    return int(cudaLaunchKernelEx(&cfg,megakernel::mk<FaithfulConfig,Globals,CollectiveOp>,g));
}
extern "C" void faithful_vm_destroy(void* handle){delete static_cast<VMPlan*>(handle);}

extern "C" int faithful_vm_create_tactic(void** handle,int tactic,void* a,void* b,void* out,int m,int n,int k,int count,void* stream) {
    if(tactic==0)return create_plan<NativeKernel>(handle,tactic,a,b,out,m,n,k,count,stream);
    if(tactic==1)return create_plan<NativeOKernel>(handle,tactic,a,b,out,m,n,k,count,stream);
    if(tactic==2)return create_plan<NativeDownKernel>(handle,tactic,a,b,out,m,n,k,count,stream);
    return -1;
}
extern "C" int faithful_vm_create(void** handle,void* a,void* b,void* out,int m,int n,int k,int count,void* stream) {
    return faithful_vm_create_tactic(handle,0,a,b,out,m,n,k,count,stream);
}

extern "C" int faithful_vm_geglu_create(void** handle,void* x,void* gate,void* up,void* hidden,int m,int h,int d,int count,void* stream) {
    *handle=nullptr;if(count<=0)return -1;
    #if !defined(PI0_PHASE_ACTION) && !defined(PI0_PHASE_VISION)
    static_assert(sizeof(GemmKernel::SharedStorage)<=FaithfulConfig::PAGE_SIZE);
#endif
    using A=GemmOp::ElementA;using B=GemmOp::ElementB;using D=GemmOp::ElementD;
    auto sa=cutlass::make_cute_packed_stride(GemmKernel::StrideA{},{m,d,1});
    auto sb=cutlass::make_cute_packed_stride(GemmKernel::StrideB{},{h,d,1});
    auto sd=cutlass::make_cute_packed_stride(GemmKernel::StrideD{},{m,h,1});
    GemmOp::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{m,h,d,1},
        {(A*)x,sa,(B*)gate,sb},{{1.f,0.f,nullptr,nullptr,{},{},{}},nullptr,{},(D*)hidden,sd},
        {(A*)x,sa,(B*)up,sb},{{1.f,0.f,nullptr,nullptr,{},{},{}},nullptr,{},(D*)hidden,sd}};
    auto params=GemmKernel::to_underlying_arguments(args,nullptr);
    int tactic=
#ifdef PI0_UNIFIED
    7;
#else
    3;
#endif
    auto* p=new VMPlan;p->count=count;p->tactic=tactic;
    auto st=static_cast<cudaStream_t>(stream);
    cudaError_t e=cudaMalloc(&p->params,sizeof(params));
    if(!e)e=cudaMemcpyAsync(p->params,&params,sizeof(params),cudaMemcpyHostToDevice,st);
    std::vector<int> instructions(size_t(FaithfulConfig::WORKERS)*count*32,0);
    for(int worker=0;worker<FaithfulConfig::WORKERS;++worker)for(int i=0;i<count;++i)instructions[(worker*count+i)*32]=1;
    if(!e)e=cudaMalloc(&p->instructions,instructions.size()*sizeof(int));
    if(!e)e=cudaMemcpyAsync(p->instructions,instructions.data(),instructions.size()*sizeof(int),cudaMemcpyHostToDevice,st);
    if(!e)e=cudaStreamSynchronize(st);
    auto kernel=megakernel::mk<FaithfulConfig,Globals,CollectiveOp>;
    if(!e)e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,FaithfulConfig::DYNAMIC_SHARED_MEMORY);
    cudaFuncAttributes attr{};if(!e)e=cudaFuncGetAttributes(&attr,kernel);
    if(!e && attr.sharedSizeBytes+FaithfulConfig::DYNAMIC_SHARED_MEMORY>232448)e=cudaErrorInvalidConfiguration;
    if(e){delete p;return int(e);}
    *handle=p;return 0;
}

extern "C" int faithful_vm_pointwise_create(void** handle,faithful_pointwise::Spec const* spec,int count,void* stream) {
    *handle=nullptr;if(count<=0 || spec->kind<10)return -1;
    auto params=*spec;int tactic=spec->kind;
    auto* p=new VMPlan;p->count=count;p->tactic=tactic;
    auto st=static_cast<cudaStream_t>(stream);
    cudaError_t e=cudaMalloc(&p->params,sizeof(params));
    if(!e)e=cudaMemcpyAsync(p->params,&params,sizeof(params),cudaMemcpyHostToDevice,st);
    std::vector<int> instructions(size_t(FaithfulConfig::WORKERS)*count*32,0);
    for(int worker=0;worker<FaithfulConfig::WORKERS;++worker)for(int i=0;i<count;++i)instructions[(worker*count+i)*32]=1;
    if(!e)e=cudaMalloc(&p->instructions,instructions.size()*sizeof(int));
    if(!e)e=cudaMemcpyAsync(p->instructions,instructions.data(),instructions.size()*sizeof(int),cudaMemcpyHostToDevice,st);
    if(!e)e=cudaStreamSynchronize(st);
    auto kernel=megakernel::mk<FaithfulConfig,Globals,CollectiveOp>;
    if(!e)e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,FaithfulConfig::DYNAMIC_SHARED_MEMORY);
    cudaFuncAttributes attr{};if(!e)e=cudaFuncGetAttributes(&attr,kernel);
    if(!e && attr.sharedSizeBytes+FaithfulConfig::DYNAMIC_SHARED_MEMORY>232448)e=cudaErrorInvalidConfiguration;
    if(e){delete p;return int(e);}
    *handle=p;return 0;
}
extern "C" int faithful_vm_merge(void** handle,void** parts,int count,void* stream) {
    *handle=nullptr;if(count<=0)return -1;
    std::vector<NativeStage> stages;
    for(int i=0;i<count;++i){auto* part=static_cast<VMPlan*>(parts[i]);if(part->count!=1 || part->stages)return -1;stages.push_back({part->params,part->tactic});}
    auto* p=new VMPlan;p->count=count;
    auto st=static_cast<cudaStream_t>(stream);
    auto e=cudaMalloc(&p->stages,stages.size()*sizeof(NativeStage));
    if(!e)e=cudaMemcpyAsync(p->stages,stages.data(),stages.size()*sizeof(NativeStage),cudaMemcpyHostToDevice,st);
    std::vector<int> instructions(size_t(FaithfulConfig::WORKERS)*count*32,0);
    for(int worker=0;worker<FaithfulConfig::WORKERS;++worker)for(int i=0;i<count;++i){instructions[(worker*count+i)*32]=1;instructions[(worker*count+i)*32+1]=i;}
    if(!e)e=cudaMalloc(&p->instructions,instructions.size()*sizeof(int));
    if(!e)e=cudaMemcpyAsync(p->instructions,instructions.data(),instructions.size()*sizeof(int),cudaMemcpyHostToDevice,st);
    if(!e)e=cudaStreamSynchronize(st);
    if(e){delete p;return int(e);}*handle=p;return 0;
}

template<class K> int generic_create(void** handle,void* a,void* b,void* out,int m,int n,int k,int lda,int ldbn,int ldbk,int ldo,float alpha,float beta,void* stream) {
    using G=cutlass::gemm::device::GemmUniversalAdapter<K>;
    int np=(n+7)/8*8,kp=(k+7)/8*8;
    faithful_generic::Pack pack{static_cast<__nv_bfloat16 const*>(a),static_cast<__nv_bfloat16 const*>(b),static_cast<__nv_bfloat16*>(out),nullptr,nullptr,nullptr,m,n,k,np,kp,lda,ldbn,ldbk,ldo,beta};
    std::vector<void*> allocations;
    auto alloc=[&](auto** ptr,size_t count){void* buffer=nullptr;auto e=cudaMalloc(&buffer,count*2);if(!e){*ptr=static_cast<__nv_bfloat16*>(buffer);allocations.push_back(buffer);}return e;};
    cudaError_t e=cudaSuccess;
    if(kp!=k || lda%8)e=alloc(&pack.pa,size_t(m)*kp);
    if(!e && (np!=n || kp!=k || (ldbk==1?ldbn%8:ldbk%8)))e=alloc(&pack.pb,size_t(np)*kp);
    if(!e && (np!=n || ldo%8))e=alloc(&pack.pd,size_t(m)*np);
    if(e){for(auto* p:allocations)cudaFree(p);return int(e);}
    auto sa=cutlass::make_cute_packed_stride(typename K::StrideA{},{m,pack.pa?kp:lda,1});
    auto sb=[&]{
        if constexpr(std::is_same_v<K,faithful_generic::NT>)return cutlass::make_cute_packed_stride(typename K::StrideB{},{np,pack.pb?kp:ldbn,1});
        else return cutlass::make_cute_packed_stride(typename K::StrideB{},{pack.pb?np:ldbk,kp,1});
    }();
    auto sd=cutlass::make_cute_packed_stride(typename K::StrideD{},{m,pack.pd?np:ldo,1});
    typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{m,np,kp,1},
        {reinterpret_cast<typename G::ElementA*>(pack.pa?pack.pa:const_cast<__nv_bfloat16*>(pack.a)),sa,
         reinterpret_cast<typename G::ElementB*>(pack.pb?pack.pb:const_cast<__nv_bfloat16*>(pack.b)),sb},
        {{alpha,beta},reinterpret_cast<typename G::ElementC*>(pack.pd?pack.pd:pack.out),sd,reinterpret_cast<typename G::ElementD*>(pack.pd?pack.pd:pack.out),sd}};
    faithful_generic::Params<K> params{K::to_underlying_arguments(args,nullptr),pack};
    int tactic=std::is_same_v<K,faithful_generic::NT>?5:6;int count=1;
    auto* p=new VMPlan;p->count=count;p->tactic=tactic;
    auto st=static_cast<cudaStream_t>(stream);
    e=cudaMalloc(&p->params,sizeof(params));
    if(!e)e=cudaMemcpyAsync(p->params,&params,sizeof(params),cudaMemcpyHostToDevice,st);
    std::vector<int> instructions(size_t(FaithfulConfig::WORKERS)*count*32,0);
    for(int worker=0;worker<FaithfulConfig::WORKERS;++worker)for(int i=0;i<count;++i)instructions[(worker*count+i)*32]=1;
    if(!e)e=cudaMalloc(&p->instructions,instructions.size()*sizeof(int));
    if(!e)e=cudaMemcpyAsync(p->instructions,instructions.data(),instructions.size()*sizeof(int),cudaMemcpyHostToDevice,st);
    if(!e)e=cudaStreamSynchronize(st);
    auto kernel=megakernel::mk<FaithfulConfig,Globals,CollectiveOp>;
    if(!e)e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,FaithfulConfig::DYNAMIC_SHARED_MEMORY);
    cudaFuncAttributes attr{};if(!e)e=cudaFuncGetAttributes(&attr,kernel);
    if(!e && attr.sharedSizeBytes+FaithfulConfig::DYNAMIC_SHARED_MEMORY>232448)e=cudaErrorInvalidConfiguration;
    if(e){delete p;for(auto* a:allocations)cudaFree(a);return int(e);}
    p->allocations=std::move(allocations);*handle=p;return 0;
}
extern "C" int faithful_vm_generic_create(void** handle,void* a,void* b,void* out,int m,int n,int k,int lda,int ldbn,int ldbk,int ldo,float alpha,float beta,void* stream) {
    *handle=nullptr;
    if(ldbk==1)return generic_create<faithful_generic::NT>(handle,a,b,out,m,n,k,lda,ldbn,ldbk,ldo,alpha,beta,stream);
    if(ldbn==1)return generic_create<faithful_generic::NN>(handle,a,b,out,m,n,k,lda,ldbn,ldbk,ldo,alpha,beta,stream);
    return -1;
}

extern "C" int faithful_vm_matrix_create(void** handle,int tactic,void* a,void* b,void* out,int m,int n,int k,float alpha,float beta,void* stream) {
    if(tactic==0)return create_plan<NativeKernel>(handle,tactic,a,b,out,m,n,k,1,stream,alpha,beta);
    if(tactic==1)return create_plan<NativeOKernel>(handle,tactic,a,b,out,m,n,k,1,stream,alpha,beta);
    if(tactic==2)return create_plan<NativeDownKernel>(handle,tactic,a,b,out,m,n,k,1,stream,alpha,beta);
    return -1;
}

#ifdef PI0_UNIFIED
extern "C" int faithful_vm_vision_create(void** handle,void* qkv,void* out,int batch,int seq,void* stream) {
    *handle=nullptr;if(batch<1 || seq!=256)return -1;
    static_assert(VisionTraits::kSmemSize<=FaithfulConfig::PAGE_SIZE);
    float* statistics=nullptr;auto alloc_error=cudaMalloc(&statistics,size_t(batch)*16*seq*sizeof(float));
    if(alloc_error)return int(alloc_error);
    CoherentVision params{qkv,static_cast<char*>(qkv)+1152*2,static_cast<char*>(qkv)+2304*2,out,statistics,
        seq,seq,batch,3456,3456,1152,float(1./sqrt(72.)*log2(exp(1.)))};
    int tactic=4,count=1;
    auto* p=new VMPlan;p->count=count;p->tactic=tactic;
    auto st=static_cast<cudaStream_t>(stream);
    cudaError_t e=cudaMalloc(&p->params,sizeof(params));
    if(!e)e=cudaMemcpyAsync(p->params,&params,sizeof(params),cudaMemcpyHostToDevice,st);
    std::vector<int> instructions(size_t(FaithfulConfig::WORKERS)*count*32,0);
    for(int worker=0;worker<FaithfulConfig::WORKERS;++worker)for(int i=0;i<count;++i)instructions[(worker*count+i)*32]=1;
    if(!e)e=cudaMalloc(&p->instructions,instructions.size()*sizeof(int));
    if(!e)e=cudaMemcpyAsync(p->instructions,instructions.data(),instructions.size()*sizeof(int),cudaMemcpyHostToDevice,st);
    if(!e)e=cudaStreamSynchronize(st);
    auto kernel=megakernel::mk<FaithfulConfig,Globals,CollectiveOp>;
    if(!e)e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,FaithfulConfig::DYNAMIC_SHARED_MEMORY);
    cudaFuncAttributes attr{};if(!e)e=cudaFuncGetAttributes(&attr,kernel);
    if(!e && attr.sharedSizeBytes+FaithfulConfig::DYNAMIC_SHARED_MEMORY>232448)e=cudaErrorInvalidConfiguration;
    if(e){delete p;cudaFree(statistics);return int(e);}
    p->allocations.push_back(statistics);*handle=p;return 0;
}
#endif

#ifdef PI0_UNIFIED
extern "C" int faithful_vm_coherent_geglu_create(void** handle,void* x,void* gate,void* up,void* hidden,int m,int h,int d,void* stream) {
    using GK=coherent_geglu::Gate;using UK=coherent_geglu::Up;
    using G=cutlass::gemm::device::GemmUniversalAdapter<GK>;using U=cutlass::gemm::device::GemmUniversalAdapter<UK>;
    *handle=nullptr;
#if !defined(PI0_PHASE_ACTION) && !defined(PI0_PHASE_VISION)
    static_assert(sizeof(GK::SharedStorage)<=FaithfulConfig::PAGE_SIZE);static_assert(sizeof(UK::SharedStorage)<=FaithfulConfig::PAGE_SIZE);
#endif
    auto sa=cutlass::make_cute_packed_stride(GK::StrideA{},{m,d,1});
    auto sb=cutlass::make_cute_packed_stride(GK::StrideB{},{h,d,1});
    auto sd=cutlass::make_cute_packed_stride(GK::StrideD{},{m,h,1});
    G::Arguments ga{cutlass::gemm::GemmUniversalMode::kGemm,{m,h,d,1},
        {(G::ElementA*)x,sa,(G::ElementB*)gate,sb},{{1.f,0.f},nullptr,sd,(G::ElementD*)hidden,sd}};
    U::Arguments ua{cutlass::gemm::GemmUniversalMode::kGemm,{m,h,d,1},
        {(U::ElementA*)x,sa,(U::ElementB*)up,sb},
        {{1.f,0.f,nullptr,nullptr,{},{},{},(coherent_geglu::T const*)hidden,sd},nullptr,sd,(U::ElementD*)hidden,sd}};
    coherent_geglu::Params params{GK::to_underlying_arguments(ga,nullptr),UK::to_underlying_arguments(ua,nullptr)};
    int tactic=3,count=1;
    auto* p=new VMPlan;p->count=count;p->tactic=tactic;
    auto st=static_cast<cudaStream_t>(stream);
    cudaError_t e=cudaMalloc(&p->params,sizeof(params));
    if(!e)e=cudaMemcpyAsync(p->params,&params,sizeof(params),cudaMemcpyHostToDevice,st);
    std::vector<int> instructions(size_t(FaithfulConfig::WORKERS)*count*32,0);
    for(int worker=0;worker<FaithfulConfig::WORKERS;++worker)for(int i=0;i<count;++i)instructions[(worker*count+i)*32]=1;
    if(!e)e=cudaMalloc(&p->instructions,instructions.size()*sizeof(int));
    if(!e)e=cudaMemcpyAsync(p->instructions,instructions.data(),instructions.size()*sizeof(int),cudaMemcpyHostToDevice,st);
    if(!e)e=cudaStreamSynchronize(st);
    auto kernel=megakernel::mk<FaithfulConfig,Globals,CollectiveOp>;
    if(!e)e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,FaithfulConfig::DYNAMIC_SHARED_MEMORY);
    cudaFuncAttributes attr{};if(!e)e=cudaFuncGetAttributes(&attr,kernel);
    if(!e && attr.sharedSizeBytes+FaithfulConfig::DYNAMIC_SHARED_MEMORY>232448)e=cudaErrorInvalidConfiguration;
    if(e){delete p;return int(e);}
    *handle=p;return 0;
}
#endif

extern "C" int faithful_vm_resource_info(int* registers,int* active_blocks,int* smem_bytes) {
    auto kernel=megakernel::mk<FaithfulConfig,Globals,CollectiveOp>;
    cudaFuncAttributes attr{};auto e=cudaFuncGetAttributes(&attr,kernel);if(e)return int(e);
    e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,FaithfulConfig::DYNAMIC_SHARED_MEMORY);if(e)return int(e);
    *registers=attr.numRegs;*smem_bytes=attr.sharedSizeBytes+FaithfulConfig::DYNAMIC_SHARED_MEMORY;
    return int(cudaOccupancyMaxActiveBlocksPerMultiprocessor(active_blocks,kernel,FaithfulConfig::NUM_THREADS,FaithfulConfig::DYNAMIC_SHARED_MEMORY));
}
