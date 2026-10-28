"""Untimed oracle using exactly the packed FP4 weights and activation format.

FP32 matmuls on dequantized values separate implementation error from BF16 ->
FP4 quantization loss. Original weights and embedding are never modified.
"""
import types
import torch
import torch.nn.functional as F
from tools.thor.model_check import errors

@torch.inference_mode()
def check_quantized(adapter, native_attention=False):
 native_logits=adapter.globs.logits.clone();native_hidden=adapter.globs.hidden_states.clone()
 native_cache=[x[:,0,adapter.context].clone() for x in adapter.model.stacked_kv_cache]
 p=adapter.p;levels=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
 restored=[];retained=[]
 import megakernels.llama as llama_module
 original_attention=llama_module.attention
 if native_attention:
  def shared_attention(query_states,key_states,value_states,kv_cache,position_ids,seq_len):
   k_cache,v_cache=kv_cache
   base=adapter.model.stacked_kv_cache[0]
   layer=(k_cache.data_ptr()-base.data_ptr())//(base.stride(0)*base.element_size())
   k_cache[:,position_ids]=key_states;v_cache[:,position_ids]=value_states
   adapter.globs.post_ln_rope_q.copy_(query_states.flatten())
   adapter.globs.barriers[layer].zero_();adapter.globs.barriers[layer,0].fill_(4)
   adapter.module.attention(adapter.handle,layer,adapter.context,torch.cuda.current_stream().cuda_stream)
   return adapter.globs.attn_out.reshape(query_states.shape).clone()
  llama_module.attention=shared_attention
 def decode(w):
  m,k=w['packed'].shape;mapped=torch.empty((m,k*2),device='cuda',dtype=torch.float32)
  for start in range(0,m,1024):
   packed=w['packed'][start:start+1024]
   codes=torch.stack((packed&15,packed>>4),-1).flatten(-2).long()
   if adapter.mixed_fp8:scales=torch.exp2(w['scales'][start:start+1024].float()-127).repeat_interleave(32,-1)
   else:scales=w['scales'][start:start+1024].view(torch.float8_e4m3fn).float().repeat_interleave(16,-1)
   mapped[start:start+1024]=levels[codes]*scales*w['tensor_scale']
  return mapped
 def patch(module,weight):
  restored.append((module,module.forward));retained.append(weight)
  def forward(x):
   if adapter.weight_only:return F.linear(x.float(),weight).bfloat16()
   if adapter.mixed_fp8:
    from .mixed_format import quantize_activation
    return F.linear(quantize_activation(x),weight).bfloat16()
   q=p.storage(x.reshape(1,-1).contiguous());q['tensor_scale']=1/64;p.quantize(q,reference=True)
   return F.linear(q['decoded'].reshape(x.shape),weight).bfloat16()
  module.forward=forward
 old_precision=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
 try:
  for i,block in enumerate(adapter.model.model.layers):
   qkv=decode(adapter.weights[4*i]);patch(block.self_attn.q_proj,qkv[:2048]);patch(block.self_attn.k_proj,qkv[2048:2560]);patch(block.self_attn.v_proj,qkv[2560:])
   patch(block.self_attn.o_proj,decode(adapter.weights[4*i+1]))
   ug=decode(adapter.weights[4*i+2]);patch(block.mlp.up_proj,ug[:8192]);patch(block.mlp.gate_proj,ug[8192:]);patch(block.mlp.down_proj,decode(adapter.weights[4*i+3]))
   restored.append((block.mlp,block.mlp.forward))
   def mlp_forward(this,state):
    residual=state.hidden_states;x=this.input_layernorm(residual)
    up=this.up_proj(x);gate=this.gate_proj(x)
    product=(F.silu(gate.float())*up.float()).bfloat16()
    state.hidden_states=residual+this.down_proj(product);return state
   block.mlp.forward=types.MethodType(mlp_forward,block.mlp)
  patch(adapter.model.lm_head.lm_head,decode(adapter.weights[64]))
  adapter.reset();output=adapter._run_torch();torch.cuda.synchronize()
  result={'reference_attention':'same VM attention' if native_attention else 'PyTorch flash attention','logits':errors(native_logits,output,nrmse_limit=.03),'hidden':errors(native_hidden,adapter.last_hidden,nrmse_limit=.03)}
  for name,native,reference in zip(('k_cache','v_cache'),native_cache,adapter.model.stacked_kv_cache):
   result[name]=errors(native,reference[:,0,adapter.context],nrmse_limit=.03)
   result[name+'_by_layer']=[errors(native[i],reference[i,0,adapter.context],nrmse_limit=.03) for i in range(native.shape[0])]
  return result
 finally:
  llama_module.attention=original_attention
  for module,forward in reversed(restored):module.forward=forward
  torch.set_float32_matmul_precision(old_precision)
  adapter.reset()
