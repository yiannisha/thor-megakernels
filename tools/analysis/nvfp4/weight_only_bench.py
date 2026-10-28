"""Packed NVFP4 weights -> on-chip BF16 -> native BF16 tcgen05; BF16 inputs."""
import argparse,ctypes as C,json
from pathlib import Path
import torch
from .bench import Primitive,ROOT,metrics,timings
@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shapes',default='2048x2048,8192x2048,2048x8192,128256x2048');ap.add_argument('--repeats',type=int,default=30);ap.add_argument('--check-only',action='store_true');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 p=Primitive(ROOT/'local/thor-w4a16/primitive.so');lib=p.lib
 lib.wo_run.argtypes=[C.c_void_p]*4+[C.c_int,C.c_int,C.c_float,C.c_int,C.c_void_p]
 p.check(lib.wo_initialize());torch.manual_seed(71)
 eviction=torch.zeros(128*1024*1024,device='cuda',dtype=torch.uint8)
 result={'precision':'NVFP4 weights, BF16 activations, BF16 tcgen05 MMA, FP32 accumulators','cases':[],'provenance':p.provenance}
 for shape in args.shapes.split(','):
  m,k=map(int,shape.split('x'));w=p.storage(torch.randn((m,k),device='cuda',dtype=torch.bfloat16)*.02);p.quantize(w,reference=True)
  x=torch.randn(k,device='cuda',dtype=torch.bfloat16);y=torch.empty(m,device='cuda',dtype=torch.bfloat16)
  ref=(w['decoded']@x.float()).bfloat16()
  def native():p.check(lib.wo_run(w['packed'].data_ptr(),w['scales'].data_ptr(),x.data_ptr(),y.data_ptr(),m,k,w['tensor_scale'],20,p.stream))
  native();torch.cuda.synchronize();error=metrics(y,ref)
  if not error['finite'] or error['relative_l2']>.001:raise RuntimeError((shape,error))
  row={'m':m,'k':k,'error_vs_same_weights_fp32':error,'results':{}}
  if not args.check_only:
   for name,fn in [('w4a16_tcgen05',native),('bf16_torch',lambda:torch.mv(w['source'],x,out=y))]:
    row['results'][name]=timings(fn,args.repeats,1,eviction)
  result['cases'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(shape,row,flush=True)
if __name__=='__main__':main()
