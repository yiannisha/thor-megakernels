"""Generic dependent linear chain: cooperative FP4 runtime vs CUDA graph.

This is an experimental backend, not a change to the production BF16 VM.
Both variants use exactly the same CUTLASS arithmetic and fixed tensor scales.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import torch
from .bench import Primitive, ROOT, metrics, timings

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--width',type=int,default=2048)
    ap.add_argument('--length',type=int,default=8)
    ap.add_argument('--repeats',type=int,default=30)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--library',type=Path,default=ROOT/'local/thor-nvfp4/persistent-auto.so')
    args=ap.parse_args()
    if args.length<=0 or args.width<=0 or args.width%256: raise ValueError('Expected positive instruction count and width multiple of 256')
    torch.manual_seed(91)
    p=Primitive(args.library);lib=p.lib
    ptr=C.c_void_p
    lib.nv_runtime_prepare.argtypes=[C.POINTER(ptr)]*4+[C.POINTER(C.c_float),C.c_int]
    lib.nv_runtime_prepare.restype=ptr
    lib.nv_runtime_run.argtypes=[ptr,C.c_int,C.c_int,ptr]
    lib.nv_runtime_destroy.argtypes=[ptr]
    lib.nv_runtime_allocated_smem.argtypes=[ptr]
    width=args.width
    initial=torch.randn((1,width),device='cuda',dtype=torch.bfloat16)
    weights=[];inputs=[];outputs=[];plans=[]
    for i in range(args.length):
        w=p.storage(torch.randn((width,width),device='cuda',dtype=torch.bfloat16)/(width**0.5))
        p.quantize(w,reference=True)
        x=p.storage(initial if i==0 else outputs[-1].view(1,width))
        p.quantize(x,reference=True)
        y=(w['decoded'].float()@x['decoded'].flatten().float()).bfloat16()
        plans.append(p.prepare(w,x,y))
        weights.append(w);inputs.append(x);outputs.append(y)
    def ptrs(values): return (ptr*len(values))(*values)
    runtime=lib.nv_runtime_prepare(ptrs(plans),ptrs([x['source'].data_ptr() for x in inputs]),
        ptrs([x['packed'].data_ptr() for x in inputs]),ptrs([x['native'].data_ptr() for x in inputs]),
        (C.c_float*len(inputs))(*[x['tensor_scale'] for x in inputs]),len(inputs))
    if not runtime: raise RuntimeError(lib.nv_error().decode())
    def graph_chain():
        for x,handle in zip(inputs,plans):
            p.quantize(x);p.check(lib.nv_run(handle,p.stream))
    def persistent_chain():
        p.check(lib.nv_runtime_run(runtime,1,1,p.stream))
    report=dict(width=width,length=args.length,threads=lib.nv_runtime_threads(),
                shared_bytes=lib.nv_runtime_smem(),allocated_shared_bytes=lib.nv_runtime_allocated_smem(runtime),stages=lib.nv_stages(),provenance=p.provenance,variants={})
    try:
        for y in outputs: y.zero_()
        graph_chain();torch.cuda.synchronize()
        reference=[y.clone() for y in outputs]
        for y in outputs: y.zero_()
        persistent_chain();torch.cuda.synchronize()
        report['per_instruction_error_vs_identical_quantized_graph']=[metrics(y,r) for y,r in zip(outputs,reference)]
        if any(e['relative_l2']!=0 or not e['finite'] for e in report['per_instruction_error_vs_identical_quantized_graph']):
            raise RuntimeError(report)
        if args.check_only:
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('PASS persistent reentry, dependency and quantization checks',flush=True)
            return
        # Rotate AB/BA order to expose drift; all samples retained.
        for repeat in range(3):
            for name,fn in ([('graph',graph_chain),('persistent',persistent_chain)] if repeat%2==0
                            else [('persistent',persistent_chain),('graph',graph_chain)]):
                result=timings(fn,args.repeats,1,warmup_replays=1000)
                report['variants'].setdefault(name,[]).append(result)
                print(name,result['median_ms'],flush=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    finally:
        lib.nv_runtime_destroy(runtime)
        for handle in plans: lib.nv_destroy(handle)

if __name__=='__main__': main()
