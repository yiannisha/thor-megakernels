"""Attribute reference-cache decode error to decoder inputs versus native head."""
import argparse,json
from pathlib import Path
import torch
from tools.thor.model_bench import ModelAdapter
from tools.thor.model_check import attention_context,errors

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=45);a=p.parse_args()
 rows=[]
 with torch.inference_mode():
  m=ModelAdapter('native',1024,1,cache_capacity=1024+a.steps+1,reference_attention='math');m.reset()
  for step in range(a.steps):
   m.context=1024+step;m.position.fill_(m.context);m.state.seq_len=m.context+1
   native=m.run().clone();hidden=m.last_hidden.clone()
   norm=m.model.lm_head.input_norm(hidden.reshape(1,-1));oracle_head=m.model.lm_head.lm_head(norm).flatten().clone()
   m.engine='torch'
   with attention_context('math'):reference=m.run().clone()
   m.engine='native'
   rows.append({'step':step,'position':m.context,'input_token':m.case.token.item(),
    'native_vs_reference':errors(native,reference,nrmse_limit=m.logit_limit),
    'torch_head_native_hidden_vs_reference':errors(oracle_head,reference,nrmse_limit=m.logit_limit),
    'native_vs_torch_head_same_hidden':errors(native,oracle_head,nrmse_limit=m.logit_limit),
    'hidden':errors(hidden,m.last_hidden)})
   m.case.token.copy_(reference.argmax().reshape(1,1))
  result={'status':'COMPLETE','protocol':{'context':1024,'seed':1,'steps':a.steps,'past':'reference','reference_attention':'math','timing_measured':False},'metadata':m.metadata,'rows':rows}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':'COMPLETE','output':str(a.output)}))
if __name__=='__main__':main()
