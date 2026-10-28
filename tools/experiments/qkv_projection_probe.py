"""Compare native Q/K/V projection reductions before RoPE on matched inputs."""
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
 result={'scope':'native QKV versus PyTorch on reference inputs; identity RoPE; FP64 dot diagnostic','rows':[],'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 with torch.inference_mode():
  a=ModelAdapter('native',1024,1,cache_capacity=1070,reference_attention='math');a.reset();a.engine='torch';captured={};hooks=[]
  with attention_context('math'):
   for step in range(44):
    a.context=1024+step;a.position.fill_(a.context);a.state.seq_len=a.context+1
    a.case.token.copy_(a.run().argmax().reshape(1,1))
   for i,layer in enumerate(a.model.model.layers):
    def capture(_,inputs,output,i=i):captured[i]={'input':inputs[0].flatten().clone(),'normalized':output.flatten().clone()}
    hooks.append(layer.self_attn.input_layernorm.register_forward_hook(capture))
    for name in ['q','k','v']:
     def projection(_,inputs,output,i=i,name=name):captured[i][name]=output.flatten().clone()
     hooks.append(getattr(layer.self_attn,name+'_proj').register_forward_hook(projection))
   a.context=1068;a.position.fill_(a.context);a.state.seq_len=a.context+1;a.run()
  for h in hooks:h.remove()
  g=a.globs
  g.rope_cos=torch.ones_like(g.rope_cos);g.rope_sin=torch.zeros_like(g.rope_sin);g.pos_id=a.context
  for layer,refs in captured.items():
   inp=refs['input']
   g.hidden_states.copy_(inp);g.barriers.zero_()
   if layer:g.barriers[layer-1,5,0]=512
   nodes,outputs=make_dag_layer(g,layer,[],'qkv');schedule=Schedule(g,nodes,DAG_Node(NoOp(),outputs));tensorize_instructions(g,assign_to_sms(a.config.scheduler,schedule=schedule));a.interpret(g,a.native)
   pairs={'q':g.post_ln_rope_q,'k':g.k_cache[layer,0,a.context].flatten(),'v':g.v_cache[layer,0,a.context].flatten()}
   for name,native in pairs.items():
    ref=refs[name]
    weights=getattr(a.model.model.layers[layer].self_attn,name+'_proj').weight
    exact=(weights.double()@refs['normalized'].double()).to(torch.bfloat16)
    result['rows'].append({'layer':layer,'projection':name,'bitwise_identical':torch.equal(native,ref),'changed_elements':int((native!=ref).sum()),'metrics':errors(native,ref),'native_vs_fp64_changed':int((native!=exact).sum()),'reference_vs_fp64_changed':int((ref!=exact).sum())})
  result['metadata']=a.metadata;result['status']='COMPLETE';result['all_bitwise_identical']=all(r['bitwise_identical']for r in result['rows'])
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'COMPLETE','all_bitwise_identical':result['all_bitwise_identical']}))
if __name__=='__main__':main()
