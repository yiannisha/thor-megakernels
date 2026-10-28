"""Nsight Systems capture of qualified real-model native decode after warmup."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=30)
    args=parser.parse_args()
    if args.steps<1:parser.error('positive steps required')
    import torch
    from tools.thor.model_bench import ModelAdapter
    with torch.inference_mode():
        a=ModelAdapter('native',1024,1)
        a.reset();output=a.run();torch.cuda.synchronize();check=a.check(output,a.reference())
        if not check['passed']:raise RuntimeError('pre-profile numerical qualification failed')
        for _ in range(5):a.reset();a.run()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for _ in range(args.steps):
            with torch.cuda.nvtx.range('native.reset'):a.reset()
            torch.cuda.synchronize()
            with torch.cuda.nvtx.range('native.decode'):output=a.run()
            torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        check=a.check(output,a.reference())
        result={'status':'PASS' if check['passed'] else 'FAIL','metadata':a.metadata,'check':check,
                'steps':args.steps,'warmups':5,'scope':'Nsight trace only; frozen-cache real-model decode, not performance benchmark',
                'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
        if not check['passed']:raise RuntimeError('post-profile numerical qualification failed')


if __name__=='__main__':main()
