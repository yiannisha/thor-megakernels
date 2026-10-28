"""Diagnostic native decoder with optional PyTorch RoPE on native pre-rotation Q/K.

This splits native execution at QKV and layer boundaries; no latency claims.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
from tools.thor.model_bench import ModelAdapter
from tools.thor.model_check import attention_context,errors
from megakernels.scheduler import Schedule,DAG_Node,assign_to_sms,tensorize_instructions
from megakernels.instructions import NoOp
from megakernels.demos.latency.scheduler import make_dag_layer,schedule_lm_head

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=100);p.add_argument('--torch-rope',action='store_true');args=p.parse_args()
 result={'status':'RUNNING','protocol':{'context':1024,'steps':args.steps,'seed':1,'past':'independent native/reference','torch_rope':args.torch_rope,'timing_measured':False},'rows':[],'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
 with torch.inference_mode():
  a=ModelAdapter('native',1024,1,cache_capacity=1024+args.steps+1,reference_attention='math');a.reset();g=a.globs;result['metadata']=a.metadata
  if a.config.ready_queue:raise ValueError('static diagnostic only')
  def pack(nodes):
   kept=set(nodes)
   for n in nodes:n.dependencies=[d for d in n.dependencies if d in kept]
   schedule=Schedule(g,nodes,DAG_Node(NoOp(),nodes));tensorize_instructions(g,assign_to_sms(a.config.scheduler,schedule=schedule))
   return g.instructions.clone(),g.timings.clone()
  if args.torch_rope:
   g.rope_cos=g.rope_cos.clone();g.rope_sin=g.rope_sin.clone()
  streams=[]
  for layer in range(g.num_hidden_layers):
   q,_=make_dag_layer(g,layer,[],'qkv');q=pack(q)
   nodes,_=make_dag_layer(g,layer,[]);minimum=2
   rest=pack([n for n in nodes if n.instruction.opcode()>=minimum]);streams.append((q,rest))
  head=pack([DAG_Node(i,[])for i in schedule_lm_head(g)])
  native_cache=[t.clone()for t in a.model.stacked_kv_cache];reference_cache=[t.clone()for t in native_cache]
  from megakernels.model_types import BatchState
  for step in range(args.steps):
   a.context=1024+step;a.position.fill_(a.context);a.state.seq_len=a.context+1;g.pos_id=a.context
   for t,c in zip(a.model.stacked_kv_cache,native_cache):t.copy_(c)
   g.hidden_states.copy_(a.model.model.embed_tokens(BatchState(input_ids=a.case.token)).hidden_states.flatten());g.barriers.zero_()
   for layer,(q,rest)in enumerate(streams):
    if args.torch_rope:
     g.rope_cos[a.context].fill_(1);g.rope_sin[a.context].zero_()
    g.instructions,g.timings=q;a.interpret(g,a.native)
    if args.torch_rope:
     from megakernels.llama import apply_rotary_pos_emb_interleaved
     query=g.post_ln_rope_q.reshape(1,1,g.num_attention_heads,g.head_dim).clone()
     key=g.k_cache[layer,0,a.context].reshape(1,1,g.num_kv_heads,g.head_dim).clone()
     cos=a.model.model.rope_cos[a.context].reshape(1,1,g.head_dim)
     sin=a.model.model.rope_sin[a.context].reshape(1,1,g.head_dim)
     query,key=apply_rotary_pos_emb_interleaved(query,key,cos,sin,unsqueeze_dim=-2)
     g.post_ln_rope_q.copy_(query.to(torch.bfloat16).flatten())
     g.k_cache[layer,0,a.context].copy_(key.to(torch.bfloat16).reshape(g.num_kv_heads,g.head_dim))
    g.instructions,g.timings=rest;a.interpret(g,a.native)
   g.instructions,g.timings=head;a.interpret(g,a.native);logits=g.logits.clone();hidden=g.hidden_states.clone();kv=[t[:,0,a.context].clone()for t in a.model.stacked_kv_cache]
   for c,t in zip(native_cache,kv):c[:,0,a.context].copy_(t)
   for t,c in zip(a.model.stacked_kv_cache,reference_cache):t.copy_(c)
   a.engine='torch'
   with attention_context('math'):reference=a.run().clone()
   a.engine='native'
   for c,t in zip(reference_cache,a.model.stacked_kv_cache):c[:,0,a.context].copy_(t[:,0,a.context])
   metrics={'logits':errors(logits,reference,nrmse_limit=a.logit_limit),'hidden':errors(hidden,a.last_hidden),'k_cache':errors(kv[0],a.model.stacked_kv_cache[0][:,0,a.context]),'v_cache':errors(kv[1],a.model.stacked_kv_cache[1][:,0,a.context])}
   result['rows'].append({'step':step,'passed':all(m['passed']for m in metrics.values()),'greedy_match':logits.argmax().item()==reference.argmax().item(),'metrics':metrics});save();a.case.token.copy_(reference.argmax().reshape(1,1))
  result['status']='PASS'if all(r['passed']for r in result['rows'])else'FAIL';result['summary']={'failed_steps':[r['step']for r in result['rows']if not r['passed']],'greedy_matches':sum(r['greedy_match']for r in result['rows']),'worst_logit_nrmse':max(r['metrics']['logits']['normalized_rmse']for r in result['rows'])};save()
 print(json.dumps({'status':result['status'],'output':str(args.output)}))
if __name__=='__main__':main()
