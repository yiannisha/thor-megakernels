"""Smoke and replay qualification for the complete BF16 source-policy shadow."""
import argparse
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from faithful_pi0.shadow import BF16Shadow
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
p=argparse.ArgumentParser();p.add_argument('--views',type=int,default=3);p.add_argument('--chunk',type=int,default=50);p.add_argument('--steps',type=int,default=10);p.add_argument('--prompt',type=int,default=0);p.add_argument('--capture',action='store_true');p.add_argument('--compare-openpi',action='store_true');args=p.parse_args()
torch.manual_seed(101);torch.set_num_threads(8)
cfg=SimpleNamespace(pi05=False,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',dtype='bfloat16',action_dim=32,action_horizon=args.chunk,pytorch_compile_mode=None)
print('Construct model',flush=True)
with torch.device('cuda'):model=PI0Pytorch(cfg).eval()
print('Build BF16 source-policy shadow',flush=True)
shadow=BF16Shadow(model,num_views=args.views,prompt_len=args.prompt,chunk_size=args.chunk,flow_steps=args.steps)
images=torch.randn((args.views,224,224,3),device='cuda');state=torch.randn((1,32),device='cuda');noise=torch.randn((1,args.chunk,32),device='cuda');tokens=torch.randint(0,257152,(1,args.prompt),device='cuda')
with torch.no_grad():
    print('Run shadow',len(shadow.ops),'operations',flush=True)
    first=shadow.forward(images,state,noise,tokens).clone();torch.cuda.synchronize()
    assert torch.isfinite(first).all()
    second=shadow.forward(images,state,noise,tokens).clone();torch.cuda.synchronize()
    assert torch.equal(first,second),float((first-second).abs().max())
    if args.capture:
        print('Capture full shadow',flush=True);shadow.capture()
        replay=shadow.forward(images,state,noise,tokens).clone();torch.cuda.synchronize()
        assert torch.equal(first,replay),float((first-replay).abs().max())
    comparison=None
    if args.compare_openpi:
        from types import MethodType
        from benchmarks.openpi_backend import capture_prefix,capture_suffix,fixed_sample
        model.embed_prefix=MethodType(capture_prefix,model);model.embed_suffix=MethodType(capture_suffix,model)
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation='eager'
        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation='eager'
        gold=fixed_sample(model,[images[i].permute(2,0,1)[None] for i in range(args.views)],
            [torch.ones(1,device='cuda',dtype=torch.bool) for _ in range(args.views)],tokens,torch.ones_like(tokens,dtype=torch.bool),state,noise,steps=args.steps)
        torch.cuda.synchronize()
        comparison={'nrmse':float((first.float()-gold.float()).square().mean().sqrt()/gold.float().square().mean().sqrt()),'max_abs':float((first.float()-gold.float()).abs().max())}
        print('Separate OpenPI comparison',comparison,flush=True)
    timings={}
    full_samples=[]
    for _ in range(10):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();shadow.forward(images,state,noise,tokens);b.record();b.synchronize();full_samples.append(a.elapsed_time(b))
    timings['full']={'median_ms':statistics.median(full_samples),'samples_ms':full_samples,'includes_input_copy':True}
    for name,(start,stop) in shadow.components.items():
        samples=[]
        for _ in range(5):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            for op in shadow.ops[start:stop]:op()
            b.record();b.synchronize();samples.append(a.elapsed_time(b))
        timings[name]={'median_ms':statistics.median(samples)}
    result={'policy':'BF16 FlashRT source adaptation','shape':vars(args),'operations':len(shadow.ops),'replay_exact':True,'graph_capture':args.capture,'scratch_bytes':sum(t.numel()*t.element_size() for t in shadow.buffers.values()),'timings':timings,'tactics':shadow.tactics,'operation_manifest':[op.manifest() for op in shadow.ops],'openpi_comparison':comparison}
    path=ROOT.parents[1]/'local/pi0/faithful/shadow.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('tactics','operation_manifest')},indent=2))
shadow.close()
