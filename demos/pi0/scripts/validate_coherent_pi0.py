"""Validate real one/three-megakernel programs against the BF16 source shadow."""
import hashlib
import argparse
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'demos/pi0'))
from faithful_pi0.shadow import BF16Shadow
from faithful_pi0.coherent import CoherentModel
from native_pi0.launch_trace import LaunchTrace
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
p=argparse.ArgumentParser();p.add_argument('--views',type=int,default=3);p.add_argument('--chunk',type=int,default=50);p.add_argument('--steps',type=int,default=10);p.add_argument('--prompt',type=int,default=0);p.add_argument('--library');p.add_argument('--action-library');p.add_argument('--geglu',choices=('fused','two-phase'),default='two-phase');p.add_argument('--down',choices=('generic','source'),default='generic');p.add_argument('--trace',action='store_true');p.add_argument('--partition',choices=('single','three','both'),default='both');p.add_argument('--report',type=Path,required=True);args=p.parse_args()
torch.manual_seed(101);torch.set_num_threads(8)
cfg=SimpleNamespace(pi05=False,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',dtype='bfloat16',action_dim=32,action_horizon=args.chunk,pytorch_compile_mode=None)
print('Construct model',flush=True)
with torch.device('cuda'):model=PI0Pytorch(cfg).eval()
s=BF16Shadow(model,num_views=args.views,prompt_len=args.prompt,chunk_size=args.chunk,flow_steps=args.steps)
images=torch.randn_like(s.images);state=torch.randn_like(s.state);noise=torch.randn_like(s.noise);tokens=torch.randint(0,257152,s.tokens.shape,device='cuda')
result={'shape':vars(args)|{'report':str(args.report)},'variants':{},'attention':'device FA2 BF16 mma.sync; projections tcgen05 group2; down cluster2x2'}
def save():
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
models={}
try:
 with torch.no_grad():
    gold=s.forward(images,state,noise,tokens).clone();torch.cuda.synchronize();s.capture()
    first_variant=None
    for partition in (('single','three') if args.partition=='both' else (args.partition,)):
        print('Build',partition,flush=True)
        n=CoherentModel(s,partition=partition,library=args.library,geglu=args.geglu,down=args.down,action_library=args.action_library);models[partition]=n;n.trace=args.trace
        out=n.forward(images,state,noise,tokens).clone();torch.cuda.synchronize()
        error=float((out.float()-gold.float()).square().mean().sqrt()/gold.float().square().mean().sqrt())
        assert torch.isfinite(out).all() and error<.03,(partition,error)
        if first_variant is None:first_variant=out.clone()
        else:
            difference=float((first_variant.float()-out.float()).square().mean().sqrt()/first_variant.float().square().mean().sqrt())
            result['single_vs_three']={'exact':torch.equal(first_variant,out),'nrmse':difference}
            if args.action_library:assert difference<.03,difference
            else:assert torch.equal(first_variant,out),difference
        again=n.forward(images,state,noise,tokens).clone();torch.cuda.synchronize();assert torch.equal(out,again)
        with LaunchTrace(ROOT/'local/pi0/model.so') as trace:n.run();torch.cuda.synchronize()
        expected=1 if partition=='single' else 3
        assert trace.count==expected,(partition,trace.count,trace.symbol)
        n.trace=False;n.capture();captured=n.forward(images,state,noise,tokens).clone();torch.cuda.synchronize();assert torch.equal(out,captured)
        result['variants'][partition]={'nrmse':error,'replay_exact':True,'captured_replay_exact':True,'kernel_launches':trace.count,'regions':n.region_ranges}
        print(partition,'qualified',error,'launches',trace.count,flush=True);save()
    engines={'shadow':s,**models};samples={tag:[] for tag in engines};tags=list(engines)
    for _ in range(5):
        for engine in engines.values():engine.forward(images,state,noise,tokens)
    torch.cuda.synchronize()
    for trial in range(20):
        order=tags[trial%len(tags):]+tags[:trial%len(tags)]
        for tag in order:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();engines[tag].forward(images,state,noise,tokens);b.record();b.synchronize();samples[tag].append(a.elapsed_time(b))
    result['protocol']='Rotating-order CUDA graph replays; 5 interleaved warmups and 20 timing rounds; includes resident input copies'
    if args.action_library:result['action_library_sha256']=hashlib.sha256(Path(args.action_library).read_bytes()).hexdigest()
    result['library_sha256']=hashlib.sha256(Path(args.library or ROOT/'local/pi0/faithful/coherent_program.so').read_bytes()).hexdigest()
    result['shadow']={'median_ms':statistics.median(samples['shadow']),'samples_ms':samples['shadow']}
    for tag in models:
        timing={'median_ms':statistics.median(samples[tag]),'samples_ms':samples[tag]}
        result['variants'][tag].update(timing=timing,speedup=result['shadow']['median_ms']/timing['median_ms'])
        print(tag,result['variants'][tag],flush=True)
    save()
finally:
 for n in reversed(list(models.values())):n.close()
 s.close()
