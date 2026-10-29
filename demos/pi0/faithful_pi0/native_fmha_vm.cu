// Source SigLIP FMHA in a group-1 native VM. See PI0_FAITHFUL_STATUS.md.
#include "fmha_native_types.h"
#include "kittens.cuh"
#include "megakernel.cuh"
#include <vector>
struct FmhaConfig:megakernel::thor_config {
    #ifdef FAITHFUL_FULL_CTA
    static constexpr int NUM_CONSUMER_WARPS=12,NUM_WARPS=16,NUM_THREADS=512;
#else
    static constexpr int NUM_CONSUMER_WARPS=12,NUM_WARPS=16,NUM_THREADS=512;
#endif
    static constexpr int CLUSTER_X=2,CLUSTER_Y=2,CLUSTER_Z=1,CLUSTER_BLOCKS=4;
    static constexpr int NUM_PAGES=1,PAGE_SIZE=Kernel::SharedStorageSize;
    static constexpr int ARENA_ALIGNMENT=1024,DYNAMIC_SMEM_ALIGNMENT=1024;
    static constexpr int DYNAMIC_SHARED_MEMORY=PAGE_SIZE;
    static constexpr int TIMING_WIDTH=1,DYNAMIC_SEMAPHORES=0,SCRATCH_BYTES=0;
    using timing_t=int[1];
    static constexpr int NATIVE_TMEM_COLUMNS=512,TMEM_CTA_GROUP=1;
    template<class G,class S> static __device__ void collective_controller(const G& g,S& s);
};
using State=megakernel::state<FmhaConfig>;
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
struct Globals {
    static constexpr bool DYNAMIC_INSTRUCTIONS=false;
    Instructions instructions;Timings timings;
    Kernel::Params const* params;
};
struct CollectiveOp {
    static constexpr int opcode=1;
    using controller=megakernel::NoOp<FmhaConfig>::controller;
    static __device__ void run(const Globals& g,State& s) {
        // All VM roles, including the controller, enter the physical CUTLASS
        // warp roles together. The source kernel determines each warp's work.
        if(threadIdx.x==0){s.wait_page_ready(0);cute::faithful_pair_mode()=2;}
        __syncthreads();
        cooperative_groups::this_grid().sync();
        auto* arena=reinterpret_cast<char*>(s.pages[0].ptr());
        auto const& p=g.params[s.instruction()[1]];
        int tiles=p.tile_scheduler.grid.x*p.tile_scheduler.grid.y*p.tile_scheduler.grid.z;
        for(int first=0;first<tiles;first+=16) {
            Kernel{}(p,arena,s.native_tmem_base,first+megakernel::get_worker_id());
            __syncthreads();
        }
        __syncthreads();
        cooperative_groups::this_grid().sync();
        if(threadIdx.x==0)s.finish_page(0,FmhaConfig::NUM_CONSUMER_WARPS);
        __syncthreads();
    }
    struct Role { static __device__ void run(const Globals& g,State& s){CollectiveOp::run(g,s);} };
    using loader=Role;using launcher=Role;using consumer=Role;using storer=Role;
};
template<class G,class S> __device__ void FmhaConfig::collective_controller(const G& g,S& s) {
    // Zero is a whole-cluster idle instruction; other opcodes join all roles.
    if(s.instruction()[0]==1)CollectiveOp::run(g,s);
}
struct VMPlan {Kernel::Params* params=nullptr;int* instructions=nullptr;int count=0,tactic=0;
    ~VMPlan(){cudaFree(params);cudaFree(instructions);}};
extern "C" int faithful_vm_run(void* handle,void* stream) {
    auto* p=static_cast<VMPlan*>(handle);
    Globals g{{p->instructions,p->count},{},p->params};
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(4,4,1);cfg.blockDim=dim3(FmhaConfig::NUM_THREADS,1,1);
    cfg.dynamicSmemBytes=FmhaConfig::DYNAMIC_SHARED_MEMORY;cfg.stream=static_cast<cudaStream_t>(stream);
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeCooperative;attr.val.cooperative=1;
    cfg.attrs=&attr;cfg.numAttrs=1;
    return int(cudaLaunchKernelEx(&cfg,megakernel::mk<FmhaConfig,Globals,CollectiveOp>,g));
}
extern "C" void faithful_vm_destroy(void* handle){delete static_cast<VMPlan*>(handle);}


extern "C" int faithful_vm_fmha_create(void** handle,void* qkv,void* out,int batch,int seq,int heads,int hd,int count,void* stream) {
    *handle=nullptr;if(count<=0)return -1;

    int d=heads*hd;int seq_r=(seq+127)/128*128;
    auto shape=cute::make_tuple(seq,seq,hd,cute::make_tuple(cute::make_tuple(1,heads),batch));
    StrideQ sq=cute::make_stride(3*d,cute::_1{},cute::make_stride(cute::make_stride(hd,hd),3*d*seq));
    StrideO so=cute::make_stride(d,cute::_1{},cute::make_stride(cute::make_stride(hd,hd),d*seq));
    StrideK sk=cute::make_stride(3*d,cute::_1{},cute::make_stride(cute::make_stride(cute::_0{},hd),3*d*seq));
    StrideLSE sl=cute::make_stride(cute::_1{},cute::make_stride(cute::make_stride(seq_r,seq_r),seq_r*heads));
    // LSE is optional in the source epilogue; PiZero only consumes O.
    auto* q=static_cast<Element const*>(qkv);
    Kernel::Arguments args{shape,{{q,sq,q+d,sk,q+2*d,sk},0.f,1.f,1.f,1.f,1.f},
        {static_cast<ElementOut*>(out),so,nullptr,sl},{0,20}};
    auto params=Kernel::to_underlying_arguments(args,nullptr);int tactic=4;
    auto* p=new VMPlan;p->count=count;p->tactic=tactic;
    auto st=static_cast<cudaStream_t>(stream);
    cudaError_t e=cudaMalloc(&p->params,sizeof(params));
    if(!e)e=cudaMemcpyAsync(p->params,&params,sizeof(params),cudaMemcpyHostToDevice,st);
    std::vector<int> instructions(size_t(16)*count*32,0);
    for(int worker=0;worker<16;++worker)for(int i=0;i<count;++i)instructions[(worker*count+i)*32]=1;
    if(!e)e=cudaMalloc(&p->instructions,instructions.size()*sizeof(int));
    if(!e)e=cudaMemcpyAsync(p->instructions,instructions.data(),instructions.size()*sizeof(int),cudaMemcpyHostToDevice,st);
    if(!e)e=cudaStreamSynchronize(st);
    auto kernel=megakernel::mk<FmhaConfig,Globals,CollectiveOp>;
    if(!e)e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,FmhaConfig::DYNAMIC_SHARED_MEMORY);
    cudaFuncAttributes attr{};if(!e)e=cudaFuncGetAttributes(&attr,kernel);
    if(!e && attr.sharedSizeBytes+FmhaConfig::DYNAMIC_SHARED_MEMORY>232448)e=cudaErrorInvalidConfiguration;
    if(e){delete p;return int(e);}
    *handle=p;return 0;
}
