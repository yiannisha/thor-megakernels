"""Compare grouped/ungrouped O projection using the same model and native binary."""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contexts',type=int,nargs='+',default=[128,1024,8192])
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    from tools.thor.model_bench import ModelAdapter
    from tools.thor.model_check import errors
    from megakernels.demos.latency.scheduler import LatencyScheduleBuilder
    from megakernels.scheduler import Schedule,assign_to_sms,tensorize_instructions
    from megakernels.demos.latency.ready_queue import tensorize_ready
    result={'status':'RUNNING','scope':'same binary/weights/inputs; grouping is the only host schedule change',
            'rows':[],'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with torch.inference_mode():
        for context in args.contexts:
            for seed in args.seeds:
                a=ModelAdapter('native',context,seed)
                if a.config.o_proj_grouped:raise ValueError('start from an ungrouped config')
                a.reset();base_logits=a.run().clone();torch.cuda.synchronize()
                base_hidden=a.last_hidden.clone();base_kv=[t[:,0,context].clone()for t in a.model.stacked_kv_cache]
                base_check=a.check(base_logits,a.reference());base_meta=deepcopy(a.metadata)
                a.reset();a.config=replace(a.config,o_proj_grouped=True,name=a.config.name+'-grouped')
                a.globs.o_proj_grouped=True
                nodes,end=LatencyScheduleBuilder.make_dag(a.globs);a.schedule=Schedule(a.globs,nodes,end)
                contract=None
                if a.config.ready_queue:contract=tensorize_ready(a.schedule)
                else:tensorize_instructions(a.globs,assign_to_sms(a.config.scheduler,schedule=a.schedule))
                grouped=a.run().clone();torch.cuda.synchronize();group_check=a.check(grouped,a.reference())
                pairs={'logits':(base_logits,grouped),'hidden':(base_hidden,a.last_hidden),
                       'k_cache':(base_kv[0],a.model.stacked_kv_cache[0][:,0,context]),
                       'v_cache':(base_kv[1],a.model.stacked_kv_cache[1][:,0,context])}
                checks={name:{'bitwise_identical':torch.equal(x,y),'changed_elements':int((x!=y).sum()),
                              'difference':errors(y,x,nrmse_limit=.03)}for name,(x,y)in pairs.items()}
                passed=base_check['passed']and group_check['passed']and all(c['bitwise_identical']for c in checks.values())
                result['rows'].append({'context':context,'seed':seed,'passed':passed,'baseline_metadata':base_meta,
                    'grouped_config':a.config.to_dict(),'grouped_tasks':len(nodes),'grouped_ready_contract':contract,
                    'baseline_check':base_check,'grouped_check':group_check,'comparisons':checks})
                save()
                if not passed:
                    result['status']='FAIL';save();raise RuntimeError('grouping changed outputs or failed reference qualification')
                del a
        result['status']='PASS';save()
    print(json.dumps({'status':result['status'],'cases':len(result['rows']),'output':str(args.output)}))


if __name__=='__main__':main()
