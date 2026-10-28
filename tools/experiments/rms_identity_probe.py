"""Expose native RMS outputs using identity Q projection and identity RoPE."""
import argparse,hashlib,json
from pathlib import Path
import torch
from tools.thor.model_bench import ModelAdapter
from tools.thor.model_check import attention_context,errors
from megakernels.scheduler import Schedule,DAG_Node,assign_to_sms,tensorize_instructions
from megakernels.instructions import NoOp
from megakernels.demos.latency.scheduler import make_dag_layer

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 result={'scope':'native RMS versus PyTorch on reference layer inputs; identity Q weights and RoPE','rows':[],'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 with torch.inference_mode():
  a=ModelAdapter('native',1024,1,cache_capacity=1070,reference_attention='math');a.reset();a.engine='torch';captured={};hooks=[]
  with attention_context('math'):
   for step in range(44):
    a.context=1024+step;a.position.fill_(a.context);a.state.seq_len=a.context+1
    a.case.token.copy_(a.run().argmax().reshape(1,1))
   for i,layer in enumerate(a.model.model.layers):
    def capture(_,inputs,output,i=i):captured[i]=(inputs[0].flatten().clone(),output.flatten().clone())
    hooks.append(layer.self_attn.input_layernorm.register_forward_hook(capture))
   a.context=1068;a.position.fill_(a.context);a.state.seq_len=a.context+1;a.run()
  for h in hooks:h.remove()
  g=a.globs;g.qkv_proj_weights=torch.zeros_like(g.qkv_proj_weights)
  g.qkv_proj_weights[:,:2048,:]=torch.eye(2048,device='cuda',dtype=torch.bfloat16)
  g.rope_cos=torch.ones_like(g.rope_cos);g.rope_sin=torch.zeros_like(g.rope_sin);g.pos_id=a.context
  for layer,(inp,ref)in captured.items():
   g.hidden_states.copy_(inp);g.barriers.zero_()
   if layer:g.barriers[layer-1,5,0]=512
   nodes,outputs=make_dag_layer(g,layer,[],'qkv');schedule=Schedule(g,nodes,DAG_Node(NoOp(),outputs));tensorize_instructions(g,assign_to_sms(a.config.scheduler,schedule=schedule));a.interpret(g,a.native)
   result['rows'].append({'layer':layer,'bitwise_identical':torch.equal(g.post_ln_rope_q,ref),'changed_elements':int((g.post_ln_rope_q!=ref).sum()),'metrics':errors(g.post_ln_rope_q,ref)})
  result['metadata']=a.metadata;result['status']='COMPLETE';result['all_bitwise_identical']=all(r['bitwise_identical']for r in result['rows'])
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'COMPLETE','all_bitwise_identical':result['all_bitwise_identical']}))
if __name__=='__main__':main()
