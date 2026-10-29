// Source GeGLU collectives, with allocation and initialization moved to setup.
#include "geglu_bf16.cu"
#include <exception>

struct FaithfulGeGLUPlan {
    GemmOp op;
    void* workspace = nullptr;
    ~FaithfulGeGLUPlan() { cudaFree(workspace); }
};
extern "C" int faithful_geglu_create(void** handle, void* x, void* gate, void* up,
    void* hidden, int m, int h, int d, cudaStream_t stream) {
    *handle = nullptr;
    auto* plan = new FaithfulGeGLUPlan;
    using A = GemmOp::ElementA; using B = GemmOp::ElementB; using D = GemmOp::ElementD;
    auto sa = cutlass::make_cute_packed_stride(GemmKernel::StrideA{}, {m,d,1});
    auto sb = cutlass::make_cute_packed_stride(GemmKernel::StrideB{}, {h,d,1});
    auto sd = cutlass::make_cute_packed_stride(GemmKernel::StrideD{}, {m,h,1});
    GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {m,h,d,1},
        {(A*)x,sa,(B*)gate,sb},
        {{1.f,0.f,nullptr,nullptr,{},{},{}},nullptr,{},(D*)hidden,sd},
        {(A*)x,sa,(B*)up,sb},
        {{1.f,0.f,nullptr,nullptr,{},{},{}},nullptr,{},(D*)hidden,sd}};
    if (plan->op.can_implement(args) != cutlass::Status::kSuccess) { delete plan; return -1; }
    auto bytes = GemmOp::get_workspace_size(args);
    if (bytes && cudaMalloc(&plan->workspace,bytes) != cudaSuccess) { delete plan; return -2; }
    if (plan->op.initialize(args,plan->workspace,stream) != cutlass::Status::kSuccess) { delete plan; return -3; }
    *handle = plan; return 0;
}
extern "C" int faithful_geglu_run(void* handle, cudaStream_t stream) {
    return int(static_cast<FaithfulGeGLUPlan*>(handle)->op.run(stream));
}
extern "C" void faithful_geglu_destroy(void* handle) { delete static_cast<FaithfulGeGLUPlan*>(handle); }
