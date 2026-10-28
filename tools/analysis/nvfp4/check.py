"""Small independent format/kernel fixture suitable for Compute Sanitizer."""
import argparse
import json
from pathlib import Path
import torch
from .bench import Primitive, metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--library',type=Path);args=ap.parse_args()
    torch.manual_seed(19)
    p=Primitive(args.library)
    levels=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device='cuda')
    fixture=p.storage(levels.repeat(128*16).reshape(128,256).bfloat16())
    fixture['tensor_scale']=1.
    p.quantize(fixture,reference=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(fixture['decoded'],fixture['source'].float(),rtol=0,atol=0)
    report={'format_fixture':'passed','cases':[]}
    # Different row/column extents exercise partial MMA tiles and scale padding.
    for m,k in [(128,256),(264,512),(128,2048)]:
        w=p.storage(torch.randn((m,k),device='cuda',dtype=torch.bfloat16))
        x=p.storage(torch.randn((1,k),device='cuda',dtype=torch.bfloat16))
        p.quantize(w,reference=True);p.quantize(x,reference=True)
        # Decode through an independent PyTorch lookup, not the CUDA decoder.
        def decode(s):
            packed=s['packed'];lo=packed&15;hi=packed>>4
            codes=torch.stack((lo,hi),dim=-1).flatten(-2).long()
            sf=s['scales'].view(torch.float8_e4m3fn).float().repeat_interleave(16,dim=-1)
            return levels[codes]*sf*s['tensor_scale']
        dw,dx=decode(w),decode(x)
        torch.testing.assert_close(dw,w['decoded'],rtol=0,atol=0)
        torch.testing.assert_close(dx,x['decoded'],rtol=0,atol=0)
        output=torch.empty(m,device='cuda',dtype=torch.bfloat16)
        plan=p.prepare(w,x,output)
        refs=[w['source'].float()@x['source'].flatten().float(),dw@x['source'].flatten().float(),dw@dx.flatten()]
        try:
            errors=[]
            for variant,ref in zip(('bf16','w4a16','w4a4'),refs):
                if variant=='w4a4': p.check(p.lib.nv_run(plan,p.stream))
                else: p.simt(w,x['source'],output,variant=='w4a16')
                torch.cuda.synchronize()
                error=metrics(output,ref.bfloat16());errors.append(error)
                if not error['finite'] or error['relative_l2']>0.001:
                    raise AssertionError((variant,m,k,error))
            report['cases'].append(dict(m=m,k=k,errors=errors))
        finally:p.lib.nv_destroy(plan)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
