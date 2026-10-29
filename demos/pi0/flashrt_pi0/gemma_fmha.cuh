#pragma once
// FlashRT's fused-attention execution pattern, implemented with the pinned
// BF16 FlashAttention collective inside our VM consumer group. A separate
// state-query tile preserves PiZero's prefix+state-only visibility rule.
using GemmaFlashTraits=Flash_fwd_kernel_traits<256,64,64,4,false,false,cutlass::bfloat16_t>;
static_assert(GemmaFlashTraits::kSmemSize<=Cfg::NUM_PAGES*Cfg::PAGE_SIZE);
template<class Attention>
__device__ inline void flashrt_gemma_fmha(const Attention& a,int task,char* pool){
    VisionParams p{};p.pi0_smem=pool;
    const bool suffix=a.cached_prefix_rows>0;
    const int per=suffix?1+(a.nq-1+63)/64:(a.nq+63)/64;
    int tile=task%per,head=(task/per)%a.heads,batch=task/(per*a.heads);
    const bool state=suffix&&tile==0;
    const int first=suffix&&!state?1:0;
    p.q_ptr=const_cast<kittens::bf16*>(static_cast<const kittens::bf16*>(a.q))+size_t(first)*a.ldq;
    p.o_ptr=static_cast<kittens::bf16*>(a.output)+size_t(first)*a.out_ld;
    p.k_ptr=a.cache_k;p.v_ptr=a.cache_v;
    p.q_row_stride=a.ldq;p.o_row_stride=a.out_ld;
    p.k_row_stride=p.v_row_stride=a.kv_heads*256;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=256;
    p.q_batch_stride=a.nq*a.ldq;p.o_batch_stride=a.nq*a.out_ld;
    p.k_batch_stride=p.v_batch_stride=a.cache_rows*a.kv_heads*256;
    p.b=a.batches;p.h=a.heads;p.h_k=a.kv_heads;p.h_h_k_ratio=a.heads/a.kv_heads;
    p.seqlen_q=state?1:a.nq-first;p.seqlen_q_rounded=(p.seqlen_q+127)/128*128;
    p.seqlen_k=state?a.cached_prefix_rows+1:a.nk;
    p.seqlen_k_rounded=(p.seqlen_k+127)/128*128;
    p.d=p.d_rounded=256;p.is_bf16=true;
    p.scale_softmax=0.0625f;p.scale_softmax_log2=a.scale_log2;
    p.scale_softmax_rp_dropout=p.scale_softmax;p.p_dropout=p.rp_dropout=1.f;
    p.window_size_left=p.window_size_right=-1;p.is_seqlens_k_cumulative=true;
    // Ordinary pi0 has batch one. State and action LSE regions are disjoint.
    p.softmax_lse_ptr=a.statistics+(suffix&&!state?a.heads:0);
    const int block=suffix?(state?0:tile-1):tile;
    pi0_vision_fa2::compute_attn_1rowblock<GemmaFlashTraits,false,false,false,false,false,false,false,false>(p,batch,head,block);
}
