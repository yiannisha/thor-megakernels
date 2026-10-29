#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "norm.cuh"
#include "softmax_bf16.cu"
#include "rope_bf16.cu"
#include "patch_bf16.cu"
extern "C" void faithful_rms(const __nv_bfloat16* x,const __nv_bfloat16* w,__nv_bfloat16* y,int rows,int dim,float eps,cudaStream_t stream) {
    rms_norm(x,w,y,rows,dim,eps,stream);
}
extern "C" void faithful_layernorm(const __nv_bfloat16* x,const __nv_bfloat16* w,const __nv_bfloat16* b,__nv_bfloat16* y,int rows,int dim,float eps,cudaStream_t stream) {
    layer_norm(x,w,b,y,rows,dim,eps,stream);
}
extern "C" void faithful_softmax(__nv_bfloat16* x,int rows,int cols,int mask_rows,int mask_start,int pad_start,cudaStream_t stream) {
    softmax_state_masked_bf16(x,rows,cols,mask_rows,mask_start,pad_start,stream);
}
extern "C" void faithful_rope(const __nv_bfloat16* qkv,const __nv_bfloat16* rope,__nv_bfloat16* q,__nv_bfloat16* k,__nv_bfloat16* v,int rows,int qdim,int kdim,int hd,int stride,int offset,int cache_stride,cudaStream_t stream) {
    qkv_split_rope_kvcache_bf16(qkv,rope,q,k,v,rows,qdim,kdim,hd,stride,offset,cache_stride,stream);
}
extern "C" void faithful_patch(const __nv_bfloat16* x,__nv_bfloat16* y,int views,cudaStream_t stream) { faithful_patch_im2col(x,y,views,stream); }
// Strict source vision boundary: BF16 GEMM -> BF16 bias add -> BF16 residual add.
__global__ void strict_bias_residual(__nv_bfloat16* x,const __nv_bfloat16* bias,const __nv_bfloat16* residual,int n,int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){auto biased=__float2bfloat16(__bfloat162float(x[i])+__bfloat162float(bias[i%dim]));
        x[i]=__float2bfloat16(__bfloat162float(biased)+__bfloat162float(residual[i]));}
}
extern "C" void faithful_bias_residual(__nv_bfloat16* x,const __nv_bfloat16* bias,const __nv_bfloat16* residual,int rows,int dim,cudaStream_t stream) {
    strict_bias_residual<<<(rows*dim+255)/256,256,0,stream>>>(x,bias,residual,rows*dim,dim);
}
#include "activation.cuh"
void fused_add_silu_bf16(__nv_bfloat16*,const __nv_bfloat16*,int,cudaStream_t);
extern "C" void faithful_gelu(__nv_bfloat16* x,int n,cudaStream_t s){gelu_inplace(x,n,s);}
extern "C" void faithful_gate(const __nv_bfloat16* g,const __nv_bfloat16* u,__nv_bfloat16* out,int n,cudaStream_t s){gate_geglu(g,u,out,n,s);}
extern "C" void faithful_add_silu(__nv_bfloat16* x,const __nv_bfloat16* bias,int n,cudaStream_t s){fused_add_silu_bf16(x,bias,n,s);}
__global__ void faithful_bias_kernel(__nv_bfloat16* x,const __nv_bfloat16* b,int n,int dim){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=__float2bfloat16(__bfloat162float(x[i])+__bfloat162float(b[i%dim]));
}
extern "C" void faithful_bias(__nv_bfloat16* x,const __nv_bfloat16* b,int rows,int dim,cudaStream_t s){
    faithful_bias_kernel<<<(rows*dim+255)/256,256,0,s>>>(x,b,rows*dim,dim);
}

#include "patch_bias_bf16.cu"
extern "C" void faithful_patch_bias(__nv_bfloat16* x,const __nv_bfloat16* b,const __nv_bfloat16* pos,int rows,int dim,cudaStream_t stream){
    faithful_patch_bias_pos_impl(x,b,pos,rows,dim,256,stream);
}
