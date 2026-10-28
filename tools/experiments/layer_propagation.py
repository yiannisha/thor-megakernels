"""Measure cumulative native layer-prefix drift at a reference-cache decode step."""
import argparse,hashlib,json
from pathlib import Path
import torch
from tools.thor.model_bench import ModelAdapter
from tools.thor.model_check import attention_context,errors
from megakernels.scheduler import Schedule,DAG_Node,assign_to_sms,tensorize_instructions
from megakernels.instructions import NoOp
from megakernels.demos.latency.scheduler import make_dag_layer

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--step',type=int,default=44);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 if args.step<0:p.error('negative step')
 result={'scope':'cumulative native prefixes with reference past KV','protocol':{'context':1024,'step':args.step,'reference_attention':'math','seed':1},'rows':[],'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 with torch.inference_mode():
  a=ModelAdapter('native',1024,1,cache_capacity=1024+args.step+2,reference_attention='math');a.reset();a.engine='torch'
  with attention_context('math'):
   for step in range(args.step):
    a.context=1024+step;a.position.fill_(a.context);a.state.seq_len=a.context+1
    token=a.run().argmax();a.case.token.copy_(token.reshape(1,1))
   past=[t.clone()for t in a.model.stacked_kv_cache];captured={};hooks=[]
   for i,layer in enumerate(a.model.model.layers):
    def capture(_,inputs,output,i=i):captured[i]=output.hidden_states.flatten().clone()
    hooks.append(layer.register_forward_hook(capture))
   a.context=1024+args.step;a.position.fill_(a.context);a.state.seq_len=a.context+1
   reference=a.run().clone()
   for hook in hooks:hook.remove()
  a.engine='native'
  for t,c in zip(a.model.stacked_kv_cache,past):t.copy_(c)
  a.run();full_hidden=a.last_hidden.clone()
  from megakernels.model_types import BatchState
  initial=a.model.model.embed_tokens(BatchState(input_ids=a.case.token)).hidden_states.flatten().clone()
  nodes=[];outputs=[];g=a.globs
  for i in range(g.num_hidden_layers):
   new,outputs=make_dag_layer(g,i,outputs);nodes.extend(new)
   for t,c in zip(a.model.stacked_kv_cache,past):t.copy_(c)
   g.hidden_states.copy_(initial);g.barriers.zero_();g.pos_id=a.context
   schedule=Schedule(g,nodes,DAG_Node(NoOp(),outputs));tensorize_instructions(g,assign_to_sms(a.config.scheduler,schedule=schedule));a.interpret(g,a.native)
   result['rows'].append({'layer':i,'hidden':errors(g.hidden_states,captured[i]),'changed_elements':int((g.hidden_states!=captured[i]).sum())})
  result['prefix_vs_full_bitwise']=torch.equal(full_hidden,g.hidden_states)
  result['prefix_vs_full']=errors(g.hidden_states,full_hidden)
  result['metadata']=a.metadata;result['status']='COMPLETE'if result['prefix_vs_full_bitwise']else'INCONSISTENT'
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'output':str(args.output)}))
if __name__=='__main__':main()
