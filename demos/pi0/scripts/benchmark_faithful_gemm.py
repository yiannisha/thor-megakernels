"""Measure the actual BF16 source collectives before adapting VM scheduling."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from faithful_pi0.library import ShadowGemm


def measure(fn):
    for _ in range(5):fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(20):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();samples.append(a.elapsed_time(b))
    return {"median_ms":statistics.median(samples),"samples_ms":samples}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    torch.manual_seed(101);torch.set_num_threads(8)
    result={"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"cases":{},"protocol":"CUDA events around one direct standalone operator call; same protocol as isolated VM projection diagnostic"}
    for name,m,n,k,tactic,beta in [('prefix_qkv',768,2560,2048,0,0.),('prefix_o',768,2048,2048,1,1.),('prefix_down',768,2048,16384,2,1.)]:
        a=torch.randn((m,k),device='cuda',dtype=torch.bfloat16);b=torch.randn((n,k),device='cuda',dtype=torch.bfloat16)
        output=torch.zeros((m,n),device='cuda',dtype=torch.bfloat16)
        with ShadowGemm(a,b,tactic=tactic,output=output,beta=beta) as op:
            op();torch.cuda.synchronize();gold=a@b.T
            error=float((output.float()-gold.float()).square().mean().sqrt()/gold.float().square().mean().sqrt())
            if not torch.isfinite(output).all() or error>.02:raise AssertionError((name,error))
            timing=measure(op)
            result['cases'][name]={"traits":op.manifest,"nrmse_first_call":error,"timing":timing}
            print(name,timing['median_ms'],op.manifest,flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
