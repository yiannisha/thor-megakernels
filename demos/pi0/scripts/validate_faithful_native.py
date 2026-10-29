"""Full native source-policy qualification against the standalone BF16 shadow."""
import argparse
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from faithful_pi0.shadow import BF16Shadow
from faithful_pi0.native import NativeModel
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
p=argparse.ArgumentParser();p.add_argument('--views',type=int,default=3);p.add_argument('--chunk',type=int,default=50);p.add_argument('--steps',type=int,default=10);p.add_argument('--prompt',type=int,default=0);p.add_argument('--diagnose',action='store_true');args=p.parse_args()
torch.manual_seed(101);torch.set_num_threads(8)
cfg=SimpleNamespace(pi05=False,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',dtype='bfloat16',action_dim=32,action_horizon=args.chunk,pytorch_compile_mode=None)
print('Construct model',flush=True)
with torch.device('cuda'):model=PI0Pytorch(cfg).eval()
print('Build source-policy shadow',flush=True)
shadow=BF16Shadow(model,num_views=args.views,prompt_len=args.prompt,chunk_size=args.chunk,flow_steps=args.steps)
images=torch.randn((args.views,224,224,3),device='cuda');state=torch.randn((1,32),device='cuda');noise=torch.randn((1,args.chunk,32),device='cuda');tokens=torch.randint(0,257152,(1,args.prompt),device='cuda')
with torch.no_grad():
    gold=shadow.forward(images,state,noise,tokens).clone();torch.cuda.synchronize()
    print('Build native source-policy program',flush=True)
    native=NativeModel(shadow)
    if args.diagnose:
        # Compare each operation from identical live input buffers. Copies of
        # output/residual snapshots exist only during this diagnostic mode.
        for i,op in enumerate(shadow.ops):
            if op.kind=='fmha':continue
            tensor=op.tensors.get('out',op.tensors.get('x',op.tensors.get('q')))
            if op.kind=='rope':tensor=op.tensors['q']
            if i>=700:print('CHECK',i,op.kind,op.params,flush=True)
            before=tensor.clone();op();expected=tensor.clone();tensor.copy_(before)
            h=native.api.operation(op)
            try:native.api.run(h);torch.cuda.synchronize()
            finally:native.api.lib.faithful_vm_destroy(h)
            error=float((tensor.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-12))
            if not torch.isfinite(tensor).all() or error>.02:raise AssertionError((i,op.kind,op.params,error))
            if (i+1)%100==0:print('Checked operation',i+1,flush=True)
    output=native.forward(images,state,noise,tokens).clone();torch.cuda.synchronize()
    error=float((output.float()-gold.float()).square().mean().sqrt()/gold.float().square().mean().sqrt())
    assert torch.isfinite(output).all() and error<.03,error
    replay=native.forward(images,state,noise,tokens).clone();torch.cuda.synchronize();assert torch.equal(output,replay)
    print('Native output NRMSE',error,'capture',len(native.regions),'VM regions',flush=True)
    native.trace=False;native.capture();captured=native.forward(images,state,noise,tokens).clone();torch.cuda.synchronize();assert torch.equal(output,captured)
    shadow.capture();samples={}
    for tag,engine in [('shadow',shadow),('native',native)]:
        for _ in range(5):engine.forward(images,state,noise,tokens)
        torch.cuda.synchronize();timings=[]
        for _ in range(20):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();engine.forward(images,state,noise,tokens);b.record();b.synchronize();timings.append(a.elapsed_time(b))
        samples[tag]={'median_ms':statistics.median(timings),'samples_ms':timings}
    result={'shape':vars(args),'source_policy_nrmse':error,'replay_exact':True,'captured_replay_exact':True,'vm_regions':len(native.regions),'operations':len(shadow.ops),'timings':samples,'speedup':samples['shadow']['median_ms']/samples['native']['median_ms']}
    path=ROOT.parents[1]/'local/pi0/faithful/native-full.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    native.close();shadow.close()
