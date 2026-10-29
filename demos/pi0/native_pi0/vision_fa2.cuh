#pragma once
// Pinned FlashAttention arithmetic executes as an ordinary VM range operator.
// No external kernel or host attention call is made by this implementation.
#define FLASH_NAMESPACE pi0_vision_fa2
#define UNFUSE_FMA
#include "flash.h"
#include "flash_fwd_kernel.h"
using VisionTraits=Flash_fwd_kernel_traits<96,128,64,4,false,false,cutlass::bfloat16_t>;
struct VisionParams:pi0_vision_fa2::Flash_fwd_params {char* pi0_smem;};
template<class Attention>
__device__ inline void native_vision_fa2(const Attention& a,int task,char* pool){
    VisionParams p{};
    p.pi0_smem=pool;
    p.q_ptr=const_cast<void*>(a.q);p.k_ptr=const_cast<void*>(a.k);p.v_ptr=const_cast<void*>(a.v);p.o_ptr=a.output;
    p.q_row_stride=a.ldq;p.k_row_stride=a.ldkv;p.v_row_stride=a.ldkv;p.o_row_stride=a.out_ld;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=72;
    p.q_batch_stride=a.nq*a.ldq;p.k_batch_stride=p.v_batch_stride=a.nk*a.ldkv;p.o_batch_stride=a.nq*a.out_ld;
    p.b=a.batches;p.h=p.h_k=16;p.h_h_k_ratio=1;
    p.seqlen_q=p.seqlen_q_rounded=a.nq;p.seqlen_k=p.seqlen_k_rounded=a.nk;
    p.d=72;p.d_rounded=96;p.is_bf16=true;
    p.scale_softmax=float(1./sqrt(72.));p.scale_softmax_log2=a.scale_log2;
    p.scale_softmax_rp_dropout=p.scale_softmax;p.p_dropout=p.rp_dropout=1.f;
    p.window_size_left=p.window_size_right=-1;p.is_seqlens_k_cumulative=true;
    p.softmax_lse_ptr=a.statistics;
    int blocks=(a.nq+127)/128,m=task%blocks,h=(task/blocks)%16,b=task/(blocks*16);
    pi0_vision_fa2::compute_attn_1rowblock<VisionTraits,false,false,false,false,false,false,false,false>(p,b,h,m);
}
