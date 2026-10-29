// Standalone BF16 shadow of FlashRT's actual CUTLASS collective families.
// Host scheduling is retained here solely as the comparison target.
#include "gemm_types_bf16.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

static thread_local std::string last_error;
static void cuda_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct GemmPlan {virtual int run(cudaStream_t)=0;virtual ~GemmPlan()=default;};
template<class G> struct TypedGemmPlan:GemmPlan {
    G gemm;void* workspace=nullptr;
    TypedGemmPlan(void* a,void* b,void* c,void* d,int m,int n,int k,float alpha,float beta,cudaStream_t stream){
        auto sa=cutlass::make_cute_packed_stride(typename G::GemmKernel::StrideA{},{m,k,1});
        auto sb=cutlass::make_cute_packed_stride(typename G::GemmKernel::StrideB{},{n,k,1});
        auto sd=cutlass::make_cute_packed_stride(typename G::GemmKernel::StrideD{},{m,n,1});
        typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,1},
            {(typename G::ElementA*)a,sa,(typename G::ElementB*)b,sb},
            {{alpha,beta},(typename G::ElementC*)c,sd,(typename G::ElementD*)d,sd}};
        if(gemm.can_implement(args)!=cutlass::Status::kSuccess)throw std::runtime_error("CUTLASS cannot implement shape/layout");
        size_t bytes=G::get_workspace_size(args);
        if(bytes)cuda_check(cudaMalloc(&workspace,bytes));
        if(gemm.initialize(args,workspace,stream)!=cutlass::Status::kSuccess){cudaFree(workspace);throw std::runtime_error("CUTLASS initialize failed");}
    }
    int run(cudaStream_t stream)override{return int(gemm.run(stream));}
    ~TypedGemmPlan(){cudaFree(workspace);}
};
template<class G> static const char* traits(const char* name){
    using K=typename G::GemmKernel;using M=typename K::CollectiveMainloop;
    using T=typename K::TileShape;using C=typename K::ClusterShape;
    static char text[1024];
    snprintf(text,sizeof(text),"{\"name\":\"%s\",\"dtype\":\"bf16\",\"accumulator\":\"fp32\",\"tile\":[%d,%d,%d],\"cluster\":[%d,%d,%d],\"stages\":%d,\"threads\":%u,\"shared_bytes\":%zu,\"tmem_columns\":%d}",
        name,int(cute::size<0>(T{})),int(cute::size<1>(T{})),int(cute::size<2>(T{})),
        int(cute::size<0>(C{})),int(cute::size<1>(C{})),int(cute::size<2>(C{})),
        int(M::DispatchPolicy::Stages),K::MaxThreadsPerBlock,sizeof(typename K::SharedStorage),int(K::TmemAllocator::Sm100TmemCapacityColumns));
    return text;
}
extern "C" const char* faithful_last_error(){return last_error.c_str();}
extern "C" const char* faithful_gemm_traits(int tactic){
    switch(tactic){
        case 0:return traits<sm100_fp16_k64::Gemm>("prefix_qkv_k64");
        case 1:return traits<sm100_fp16_plain::Gemm>("prefix_o_plain");
        case 2:return traits<sm100_fp16_2sm21::Gemm>("prefix_down_2sm21");
        default:return nullptr;
    }
}
extern "C" int faithful_gemm_create(void** handle,int tactic,void* a,void* b,void* c,void* d,int m,int n,int k,float alpha,float beta,void* stream){
    if(!handle)return -1;*handle=nullptr;
    try{
        auto st=static_cast<cudaStream_t>(stream);
        switch(tactic){
            case 0:*handle=new TypedGemmPlan<sm100_fp16_k64::Gemm>(a,b,c,d,m,n,k,alpha,beta,st);break;
            case 1:*handle=new TypedGemmPlan<sm100_fp16_plain::Gemm>(a,b,c,d,m,n,k,alpha,beta,st);break;
            case 2:*handle=new TypedGemmPlan<sm100_fp16_2sm21::Gemm>(a,b,c,d,m,n,k,alpha,beta,st);break;
            default:throw std::runtime_error("Unknown selected tactic");
        }
        return 0;
    }catch(const std::exception& e){last_error=e.what();return -2;}
}
extern "C" int faithful_gemm_run(void* handle,void* stream){
    if(!handle)return -1;return static_cast<GemmPlan*>(handle)->run(static_cast<cudaStream_t>(stream));
}
extern "C" void faithful_gemm_destroy(void* handle){delete static_cast<GemmPlan*>(handle);}

// Symbols used by the unchanged two-phase GeGLU bundle's host entry points.
template<class G> static int bundled(void* a,void* b,void* d,int m,int n,int k,float alpha,float beta,cudaStream_t stream){
    try{TypedGemmPlan<G> plan(a,b,d,d,m,n,k,alpha,beta,stream);return plan.run(stream);}
    catch(const std::exception& e){last_error=e.what();return -2;}
}
extern "C" int faithful_bf16_k64(void* a,void* b,void* d,int m,int n,int k,float alpha,float beta,cudaStream_t stream){return bundled<sm100_fp16_k64::Gemm>(a,b,d,m,n,k,alpha,beta,stream);}
extern "C" int faithful_bf16_2sm21(void* a,void* b,void* d,int m,int n,int k,float alpha,float beta,cudaStream_t stream){return bundled<sm100_fp16_2sm21::Gemm>(a,b,d,m,n,k,alpha,beta,stream);}
