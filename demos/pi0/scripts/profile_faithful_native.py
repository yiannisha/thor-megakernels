"""Matched component/operation timings to locate native-port regressions."""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'demos/pi0'))
from faithful_pi0.shadow import BF16Shadow
from faithful_pi0.native import NativeModel
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
p=argparse.ArgumentParser();p.add_argument('--library');p.add_argument('--operators',action='store_true');args=p.parse_args()
torch.manual_seed(101);torch.set_num_threads(8)
cfg=SimpleNamespace(pi05=False,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',dtype='bfloat16',action_dim=32,action_horizon=50,pytorch_compile_mode=None)
with torch.device('cuda'):model=PI0Pytorch(cfg).eval()
s=BF16Shadow(model);n=NativeModel(s,library=args.library)
images=torch.randn_like(s.images);state=torch.randn_like(s.state);noise=torch.randn_like(s.noise);tokens=torch.empty_like(s.tokens)
results={'components':{},'operators':{}}
@torch.no_grad()
def graph(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):fn();fn()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    return g

def measure(a,b):
    for _ in range(10):a.replay();b.replay()
    torch.cuda.synchronize();samples={'shadow':[],'native':[]}
    for _ in range(10):
        for tag,g in [('shadow',a),('native',b)]:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();g.replay();end.record();end.synchronize();samples[tag].append(start.elapsed_time(end))
    return {tag:statistics.median(times) for tag,times in samples.items()}
with torch.no_grad():
    s.forward(images,state,noise,tokens);n.forward(images,state,noise,tokens);torch.cuda.synchronize()
    for name,(lo,hi) in s.components.items():
        def standalone():
            for op in s.ops[lo:hi]:op()
        def native():
            for (kind,h),(first,last) in zip(n.regions,n.region_ranges):
                if first>=lo and last<hi:
                    if kind=='group2':n.api.run(h)
                    else:
                        code=n.fmha.faithful_vm_run(h,n.api.stream)
                        if code:raise RuntimeError(code)
        a,b=graph(standalone),graph(native);result=measure(a,b)
        results['components'][name]=result;print(name,result,flush=True)
    if args.operators:
        seen=set()
        for i,op in enumerate(s.ops):
            if op.kind not in ('gemm','geglu','rms','softmax'):continue
            key=(op.kind,str(op.params),tuple((k,tuple(v.shape)) for k,v in op.tensors.items()))
            if key in seen:continue
            seen.add(key)
            handle=n.api.operation(op)
            try:
                a,b=graph(op),graph(lambda:n.api.run(handle));result=measure(a,b)
                results['operators'][str(i)]={'operation':op.manifest(),**result};print(i,op.kind,op.params,result,flush=True)
            finally:n.api.lib.faithful_vm_destroy(handle)
(ROOT/'local/pi0/faithful/native-profile.json').write_text(json.dumps(results,indent=2)+'\n')
n.close();s.close()
