"""Localize one teacher-forced temporal failure with independent KV histories."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',type=int,default=1024)
    parser.add_argument('--step',type=int,default=10)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--horizon',type=int,default=100)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.context<1 or not 0<=args.step<args.horizon<=100: parser.error('invalid context/step')
    import torch
    from tools.thor.model_bench import ModelAdapter
    from tools.thor.model_check import attention_context,errors
    from megakernels.scheduler import Schedule,DAG_Node,assign_to_sms,tensorize_instructions
    from megakernels.instructions import NoOp
    from megakernels.demos.latency.scheduler import make_dag_layer
    result={'scope':'diagnostic only; reference layer inputs with native/reference past KV',
            'protocol':vars(args)|{'output':str(args.output)},'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'trajectory':[],'rows':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save(): args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with torch.inference_mode():
        a=ModelAdapter('native',args.context,args.seed,cache_capacity=args.context+args.horizon+1,reference_attention='math')
        result['metadata']=a.metadata
        a.reset()
        native_cache=[t.clone() for t in a.model.stacked_kv_cache]
        reference_cache=[t.clone() for t in a.model.stacked_kv_cache]
        captured=[{} for _ in a.model.model.layers]
        hooks=[]
        for step in range(args.step+1):
            a.context=args.context+step;a.position.fill_(a.context);a.state.seq_len=a.context+1
            for t,c in zip(a.model.stacked_kv_cache,native_cache):t.copy_(c)
            if step==args.step:past_native=[t.clone() for t in native_cache]
            native_logits=a.run().clone();native_hidden=a.last_hidden.clone()
            for c,t in zip(native_cache,a.model.stacked_kv_cache):c[:,0,a.context].copy_(t[:,0,a.context])
            for t,c in zip(a.model.stacked_kv_cache,reference_cache):t.copy_(c)
            if step==args.step:
                past_reference=[t.clone() for t in reference_cache]
                for i,layer in enumerate(a.model.model.layers):
                    def pre(_,inputs,i=i):captured[i]['input']=inputs[0].hidden_states.flatten().clone()
                    def attn(_,inputs,output,i=i):captured[i]['oproj']=output.hidden_states.flatten().clone()
                    def full(_,inputs,output,i=i):captured[i]['downproj']=output.hidden_states.flatten().clone()
                    def partial(_,inputs,i=i):captured[i]['partial']=inputs[0].flatten().clone()
                    def upgate(_,inputs,i=i):captured[i]['upgate']=inputs[0].flatten().clone()
                    hooks.extend([layer.register_forward_pre_hook(pre),layer.self_attn.register_forward_hook(attn),
                                  layer.register_forward_hook(full),layer.self_attn.o_proj.register_forward_pre_hook(partial),
                                  layer.mlp.down_proj.register_forward_pre_hook(upgate)])
            a.engine='torch'
            with attention_context('math'):reference=a.run().clone()
            a.engine='native'
            for c,t in zip(reference_cache,a.model.stacked_kv_cache):c[:,0,a.context].copy_(t[:,0,a.context])
            result['trajectory'].append({'step':step,'input_token':a.case.token.item(),
                'logits':errors(native_logits,reference,nrmse_limit=a.logit_limit),'hidden':errors(native_hidden,a.last_hidden),
                'native_token':native_logits.argmax().item(),'reference_token':reference.argmax().item()})
            a.case.token.copy_(reference.argmax().reshape(1,1));save()
        for hook in hooks:hook.remove()
        g=a.globs
        for past_name,past in [('reference',past_reference),('native',past_native)]:
            for layer,refs in enumerate(captured):
                for op in ['partial','oproj','upgate','downproj']:
                    for t,c in zip(a.model.stacked_kv_cache,past):t.copy_(c)
                    g.hidden_states.copy_(refs['input']);g.barriers.zero_();g.pos_id=a.context
                    if layer:g.barriers[layer-1,5,0]=512
                    nodes,outputs=make_dag_layer(g,layer,[],op)
                    schedule=Schedule(g,nodes,DAG_Node(NoOp(),outputs))
                    tensorize_instructions(g,assign_to_sms(a.config.scheduler,schedule=schedule))
                    a.interpret(g,a.native)
                    output={'partial':g.attn_out,'oproj':g.hidden_states,'upgate':g.silu_out,'downproj':g.hidden_states}[op]
                    result['rows'].append({'past':past_name,'layer':layer,'op':op,'output':errors(output,refs[op])});save()
    print(json.dumps({'output':str(args.output),'comparisons':len(result['rows'])}))


if __name__=='__main__':main()
