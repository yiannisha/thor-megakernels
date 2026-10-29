"""Alternating phase timings for the exact three-megakernel workload."""
import argparse,json,statistics,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'demos/pi0'))
from faithful_pi0.shadow import BF16Shadow
from faithful_pi0.coherent import CoherentModel
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
p=argparse.ArgumentParser();p.add_argument('--library');p.add_argument('--geglu',default='two-phase');p.add_argument('--down',default='generic');p.add_argument('--report',type=Path,required=True);args=p.parse_args()
torch.manual_seed(101);torch.set_num_threads(8)
cfg=SimpleNamespace(pi05=False,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',dtype='bfloat16',action_dim=32,action_horizon=50,pytorch_compile_mode=None)
with torch.device('cuda'):m=PI0Pytorch(cfg).eval()
s=BF16Shadow(m);n=CoherentModel(s,partition='three',library=args.library,geglu=args.geglu,down=args.down)
result={}
with torch.no_grad():
 s.run();n.run();torch.cuda.synchronize()
 for index,(name,(lo,hi)) in enumerate(s.components.items()):
  def ref():
   for op in s.ops[lo:hi]:op()
  def native():n.api.run(n.regions[index][1])
  graphs={}
  for tag,fn in [('shadow',ref),('native',native)]:
   fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):fn()
   graphs[tag]=g
  for _ in range(5):
   for g in graphs.values():g.replay()
  torch.cuda.synchronize();samples={k:[] for k in graphs}
  for trial in range(20):
   for tag in (('shadow','native') if trial%2==0 else ('native','shadow')):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();graphs[tag].replay();b.record();b.synchronize();samples[tag].append(a.elapsed_time(b))
  result[name]={k:{'median_ms':statistics.median(v),'samples_ms':v} for k,v in samples.items()}
  print(name,{k:v['median_ms'] for k,v in result[name].items()},flush=True)
 args.report.write_text(json.dumps(result,indent=2)+'\n')
n.close();s.close()
