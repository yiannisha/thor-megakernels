#pragma once
#include "native_pointwise_source.cuh"
namespace faithful_pointwise {
__device__ inline int worker_count(){return gridDim.x*gridDim.y*gridDim.z;}
struct Spec {
    const void *x,*a,*b,*c;void *out,*k,*v;
    int kind,m,n,kdim,ldx,ldo,p0,p1,p2,p3;float scale;
};
enum {Rms=10,LayerNorm,Softmax,Patch,Rope,Bias,PatchBias,BiasResidual,Gelu,Gate,AddSilu,Copy,Zero,Embedding};
__device__ inline void run(Spec const& s,char* arena) {
    using T=__nv_bfloat16;
    auto* x=static_cast<T const*>(s.x);auto* a=static_cast<T const*>(s.a);
    auto* b=static_cast<T const*>(s.b);auto* c=static_cast<T const*>(s.c);auto* out=static_cast<T*>(s.out);
    int worker=megakernel::get_worker_id();
    if(s.kind==Rms || s.kind==LayerNorm) {
        for(int row=worker;row<s.m;row+=worker_count()) {
            if(s.kind==Rms)rms_norm_kernel<T>(x,a,out,s.n,s.scale,row,reinterpret_cast<float*>(arena));
            else layer_norm_kernel<T>(x,a,b,out,s.n,s.scale,row,reinterpret_cast<float*>(arena));
            __syncthreads();
        }
        return;
    }
    if(s.kind==Softmax) {
        for(int row=worker*8+int(threadIdx.x)/32;row<s.m;row+=8*worker_count())
            softmax_state_masked_bf16_kernel(out,s.m,s.n,s.p0,s.p1,s.p2,row);
        return;
    }
    for(int i=worker*256+threadIdx.x;i<s.m*s.n;i+=worker_count()*256) {
        int row=i/s.n,col=i%s.n;
        switch(s.kind) {
        case Patch:faithful_patch_im2col_kernel(x,out,s.p0,i);break;
        case Rope:qkv_split_rope_kvcache_bf16_kernel(x,a,out,static_cast<T*>(s.k),static_cast<T*>(s.v),s.m,s.p0,s.p1,s.p2,s.n,s.p3,s.ldo,i);break;
        case PatchBias:faithful_patch_bias_pos_impl_kernel(out,a,b,s.m,s.n,256,i);break;
        case Bias:out[i]=__float2bfloat16(__bfloat162float(out[i])+__bfloat162float(a[col]));break;
        case BiasResidual:{
            auto biased=__float2bfloat16(__bfloat162float(out[i])+__bfloat162float(a[col]));
            out[i]=__float2bfloat16(__bfloat162float(biased)+__bfloat162float(b[i]));break;}
        case Gelu:{float v=__bfloat162float(out[i]);float t=tanhf(.7978845608f*(v+.044715f*v*v*v));out[i]=__float2bfloat16(v*.5f*(1.f+t));break;}
        case Gate:{float g=__bfloat162float(x[i]),u=__bfloat162float(a[i]);float gelu=g/(1.f+expf(-1.5957691216057308f*g*(1.f+.044715f*g*g)));out[i]=__float2bfloat16(gelu*u);break;}
        case AddSilu:{float v=__bfloat162float(out[i])+__bfloat162float(a[i]);out[i]=__float2bfloat16(v/(1.f+expf(-v)));break;}
        case Copy:out[row*s.ldo+col]=x[row*s.ldx+col];break;
        case Zero:out[row*s.ldo+col]=__float2bfloat16(0.f);break;
        case Embedding:{auto token=static_cast<long long const*>(s.a)[row];out[i]=__float2bfloat16(__bfloat162float(x[token*s.n+col])*s.scale);break;}
        }
    }
}
}
