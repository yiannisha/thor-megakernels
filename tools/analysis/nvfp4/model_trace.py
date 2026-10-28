"""Frozen-input checks: separate quantization, native MMA, and attention errors."""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from .model_bench import FP4ModelAdapter
from tools.thor.model_check import errors,attention_context

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--context',type=int,default=1024);ap.add_argument('--weight-only',action='store_true');args=ap.parse_args()
 a=FP4ModelAdapter(args.context,weight_only=args.weight_only);trace=[]
 for w in a.weights:
  m,halfk=w['packed'].shape;k=halfk*2
  trace.append({name:torch.full((size,),float('nan'),device='cuda',dtype=torch.bfloat16)
                for name,size in [('input',k),('gate',k),('residual',m),('output',m)]})
 a.module.set_trace(a.handle,[[t[name].data_ptr() for name in ('input','gate','residual','output')] for t in trace])
 a.reset();a.run();torch.cuda.synchronize()
 levels=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
 def unpack(packed):return torch.stack((packed&15,packed>>4),-1).flatten(-2).long()
 def weight(w):
  m,hk=w['packed'].shape;out=torch.empty((m,hk*2),device='cuda',dtype=torch.float32)
  for start in range(0,m,1024):
   out[start:start+1024]=levels[unpack(w['packed'][start:start+1024])]*w['scales'][start:start+1024].view(torch.float8_e4m3fn).float().repeat_interleave(16,-1)*w['tensor_scale']
  return out
 result={'precision':'W4A16' if args.weight_only else 'W4A4','scope':'one decode; each operator checked on its actual native input','operations':[],'attention':[]}
 old=torch.get_float32_matmul_precision();torch.set_float32_matmul_precision('highest')
 try:
  for i,(w,x,t) in enumerate(zip(a.weights,a.inputs,trace)):
   source=t['input'];kind=i%4 if i<64 else 0
   if kind in (0,2):
    norm=a.globs.lm_head_norm_weights if i==64 else a.globs.attn_ln_weights[i//4] if kind==0 else a.globs.mlp_ln_weights[i//4]
    transformed=(source.float()*torch.rsqrt(source.float().square().mean()+a.globs.rms_norm_eps)).bfloat16()
    transformed=(transformed*norm).bfloat16()
   elif kind==3:transformed=(F.silu(t['gate'].float())*source.float()).bfloat16()
   else:transformed=source
   if args.weight_only:
    actual_x=a.bf16_activations[i].float()
    decoded=weight(w)
    y=F.linear(actual_x,decoded).bfloat16()
    y_expected=F.linear(transformed.float(),decoded).bfloat16()
    if i<64 and kind in (1,3):y=y+t['residual'];y_expected=y_expected+t['residual']
    row={'index':i,'layer':i//4 if i<64 else None,'kind':('qkv','o','upgate','down')[kind] if i<64 else 'head',
     'activation':errors(actual_x,transformed.float(),nrmse_limit=.001),
     'mma_on_actual_activation':errors(t['output'],y,nrmse_limit=.001),
     'whole_op_on_frozen_input':errors(t['output'],y_expected,nrmse_limit=.003)}
   else:
    q=a.p.storage(transformed.reshape(1,-1).contiguous());q['tensor_scale']=1/64;a.p.quantize(q,reference=True)
    k=source.numel();groups=torch.arange(k//16,device='cuda');offsets=(groups//4)*512+groups%4
    scales=x['native'][offsets].view(torch.float8_e4m3fn).float()
    actual_x=levels[unpack(x['packed'])].flatten()*scales.repeat_interleave(16)/64
    decoded=weight(w)
    y=F.linear(actual_x,decoded).bfloat16()
    y_expected=F.linear(q['decoded'].flatten(),decoded).bfloat16()
    if i<64 and kind in (1,3):y=y+t['residual'];y_expected=y_expected+t['residual']
    row={'index':i,'layer':i//4 if i<64 else None,'kind':('qkv','o','upgate','down')[kind] if i<64 else 'head',
     'activation':errors(actual_x,q['decoded'].flatten(),nrmse_limit=.001),
     'packed_byte_mismatches':int((x['packed']!=q['packed']).sum()),
     'scale_mismatches':int((x['native'][offsets]!=q['scales'].flatten()).sum()),
     'mma_on_actual_activation':errors(t['output'],y,nrmse_limit=.001),
     'whole_op_on_frozen_input':errors(t['output'],y_expected,nrmse_limit=.003)}
   result['operations'].append(row)
   print(i,row['kind'],'activation',row['activation']['normalized_rmse'],'MMA',row['mma_on_actual_activation']['normalized_rmse'],'whole',row['whole_op_on_frozen_input']['normalized_rmse'],flush=True)
   del decoded
  for layer in range(16):
   raw=trace[layer*4]['output'][:2048].reshape(32,64).float()
   pair=raw.reshape(32,32,2).flip(-1).reshape(32,64)*torch.tensor([-1,1],device='cuda').repeat(32)
   cos=a.globs.rope_cos[args.context].flatten();sin=a.globs.rope_sin[args.context].flatten()
   q=(raw*cos+pair*sin).bfloat16().reshape(1,32,1,64)
   keys=a.model.stacked_kv_cache[0][layer,0,:args.context+1].transpose(0,1).unsqueeze(0)
   values=a.model.stacked_kv_cache[1][layer,0,:args.context+1].transpose(0,1).unsqueeze(0)
   with attention_context('flash'):y=F.scaled_dot_product_attention(q,keys,values,enable_gqa=True).flatten()
   result['attention'].append(errors(trace[layer*4+1]['input'],y,nrmse_limit=.003))
 finally:torch.set_float32_matmul_precision(old)
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
 print('max attention error',max(x['normalized_rmse'] for x in result['attention']),flush=True)
if __name__=='__main__':main()
