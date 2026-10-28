"""Native FP4 instruction in the real VM: dynamic quantization and dependent GEMVs."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch
from .bench import Primitive,ROOT,metrics,timings

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--stress-replays',type=int,default=3)
 ap.add_argument('--fused-quant',action='store_true')
 ap.add_argument('--rows',type=int)
 ap.add_argument('--width',type=int,default=2048);ap.add_argument('--length',type=int,default=8)
 ap.add_argument('--quant-workers',type=int,default=0)
 ap.add_argument('--workers',type=int,default=20);ap.add_argument('--noops',action='store_true')
 ap.add_argument('--no-quant',action='store_true');ap.add_argument('--check-only',action='store_true')
 ap.add_argument('--repeats',type=int,default=30);ap.add_argument('--output',type=Path,required=True)
 args=ap.parse_args()
 if args.fused_quant and args.no_quant: raise ValueError('Fused mode needs quantized activations')
 if args.width%256 or args.width<=0 or args.length<=0:raise ValueError('Invalid linear program')
 if args.rows is not None and args.length!=1: raise ValueError('Rectangular fixture requires length 1')
 rows=args.rows or args.width
 torch.manual_seed(91)
 p=Primitive(ROOT/'local/thor-native-fp4-vm/vm.so');lib=p.lib
 from tools.thor.repro import source_hash
 if p.provenance.get('framework_source_hash')!=source_hash():raise RuntimeError('VM binary needs rebuilding after framework changes')
 for name,digest in p.provenance['vm_sources'].items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Rebuild required: {name}')
 ptr=C.c_void_p
 lib.vm_prepare.argtypes=[C.POINTER(ptr)]*5+[C.POINTER(C.c_float)]*2+[C.c_int]*7
 lib.vm_prepare.restype=ptr;lib.vm_run.argtypes=[ptr,ptr];lib.vm_destroy.argtypes=[ptr]
 weights=[];inputs=[];outputs=[];plans=[]
 initial=torch.randn((1,args.width),device='cuda',dtype=torch.bfloat16)
 for i in range(args.length):
  w=p.storage(torch.randn((rows,args.width),device='cuda',dtype=torch.bfloat16)/args.width**.5)
  p.quantize(w,reference=True)
  x=p.storage(initial if i==0 or args.no_quant else outputs[-1].view(1,args.width))
  p.quantize(x,reference=True)
  y=(w['decoded']@x['decoded'].flatten()).bfloat16()
  plans.append(p.prepare(w,x,y));weights.append(w);inputs.append(x);outputs.append(y)
 def array(values):return (ptr*len(values))(*values)
 def floats(values):return (C.c_float*len(values))(*values)
 vm=lib.vm_prepare(array(plans),array([x['source'].data_ptr() for x in inputs]),
  array([x['packed'].data_ptr() for x in inputs]),array([x['native'].data_ptr() for x in inputs]),
  array([y.data_ptr() for y in outputs]),floats([x['tensor_scale'] for x in inputs]),
  floats([w['tensor_scale']*x['tensor_scale'] for w,x in zip(weights,inputs)]),
  args.length,args.workers,int(not args.no_quant),int(not args.no_quant),int(args.noops),args.quant_workers,int(args.fused_quant))
 if not vm:raise RuntimeError(lib.nv_error().decode())
 def graph():
  for x,plan in zip(inputs,plans):
   if not args.no_quant:p.quantize(x)
   p.check(lib.nv_run(plan,p.stream))
 def native():p.check(lib.vm_run(vm,p.stream))
 report=dict(width=args.width,rows=rows,length=args.length,workers=args.workers,noops=args.noops,
  fused_quant=args.fused_quant,quantize=not args.no_quant,quant_workers=args.quant_workers,provenance=p.provenance,variants={})
 try:
  report['stress_errors']=[]
  for replay in range(args.stress_replays):
   if replay and not args.no_quant: initial.normal_()
   graph();torch.cuda.synchronize();reference=[y.clone() for y in outputs]
   for y in outputs:y.zero_()
   # Poison logical inputs so stale graph-produced packed values cannot mask
   # missing device-side quantization or publication in the VM.
   if not args.no_quant:
    for x in inputs:x['packed'].fill_(255);x['native'].zero_()
   native();torch.cuda.synchronize()
   errors=[metrics(y,r) for y,r in zip(outputs,reference)]
   report['stress_errors'].append(errors)
   if any(not e['finite'] or e['relative_l2']>0.001 for e in errors):
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    raise RuntimeError(errors)
  report['error_vs_identical_quantized_graph']=report['stress_errors'][-1]
  if not args.check_only:
   for block in range(3):
    for name,fn in ([('graph',graph),('vm',native)] if block%2==0 else [('vm',native),('graph',graph)]):
     result=timings(fn,args.repeats,1,warmup_replays=1000)
     report['variants'].setdefault(name,[]).append(result);print(name,result['median_ms'],flush=True)
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
  print('PASS',report['error_vs_identical_quantized_graph'],flush=True)
 finally:
  lib.vm_destroy(vm)
  for plan in plans:lib.nv_destroy(plan)
if __name__=='__main__':main()
