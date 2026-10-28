"""Exercise native VM ready dispatch with empty, chain, fan-in and shuffled DAGs."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from tools.thor.model_bench import ModelAdapter
    from megakernels.instructions import NoOp
    from megakernels.scheduler import DAG_Node,Schedule
    from megakernels.demos.latency.ready_queue import tensorize_ready,completion_check
    rows=[]
    with torch.inference_mode():
        a=ModelAdapter('native',128,1)
        if not a.config.ready_queue:raise ValueError('ready build/config required')
        before=a.globs.hidden_states.clone()
        rng=random.Random(71)
        for count in (0,1,2,3,17,257):
            for graph in ('chain','fan','random'):
                nodes=[]
                for i in range(count):
                    deps=nodes[-1:] if graph=='chain' else nodes[:1] if graph=='fan' else [n for n in nodes if rng.random()<.03]
                    nodes.append(DAG_Node(NoOp(),list(deps)))
                rng.shuffle(nodes)
                schedule=Schedule(a.globs,nodes,nodes[-1] if nodes else None)
                contract=tensorize_ready(schedule,audit_model=False)
                a.interpret(a.globs,a.native);torch.cuda.synchronize()
                check=completion_check(a.globs)
                unchanged=torch.equal(a.globs.hidden_states,before)
                rows.append({'tasks':count,'graph':graph,'contract':contract,'coverage':check,
                             'activation_unchanged':unchanged,'passed':check['passed']and unchanged})
                if not rows[-1]['passed']:raise RuntimeError('ready task coverage/state failure')
    result={'status':'PASS','metadata':a.metadata,'rows':rows,
            'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','cases':len(rows),'output':str(args.output)}))


if __name__=='__main__':main()
