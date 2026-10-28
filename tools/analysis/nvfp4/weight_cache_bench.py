"""Decoded BF16 cache experiment; matched MMA/staging, warm and evicted L2.

This is a primitive reuse experiment, not a full-model latency measurement.
"""
import argparse,ctypes as C,hashlib,json,statistics
from pathlib import Path
import torch
from .bench import Primitive,ROOT,metrics
def timings(fn,repeats,inner,eviction=None,graph_policy=None):
 for _ in range(3):fn()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph(keep_graph=True)
 with torch.cuda.graph(g):fn()
 if graph_policy:graph_policy(g.raw_cuda_graph())
 g.instantiate()
 for _ in range(10):g.replay()
 torch.cuda.synchronize();samples=[]
 for _ in range(repeats):
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  if eviction is not None:eviction.add_(1)
  start.record();g.replay();end.record();end.synchronize();samples.append(start.elapsed_time(end))
 return dict(median_ms=statistics.median(samples),samples_ms=samples,inner=1)

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shapes',default='512x2048,2048x2048,8192x2048,8192x8192');ap.add_argument('--repeats',type=int,default=20);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 p=Primitive(ROOT/'local/thor-w4a16/primitive.so');lib=p.lib;ptr=C.c_void_p
 for name in ('experimental/weight_only_fp4/primitive.cu','experimental/weight_only_fp4/cache_policy.cu'):
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=p.provenance['source_sha256'].get(name):raise RuntimeError('Rebuild weight-only cache probe: '+name)
 if hashlib.sha256((ROOT/'local/thor-w4a16/cache_policy.so').read_bytes()).hexdigest()!=p.provenance.get('cache_policy_binary_sha256'):raise RuntimeError('Rebuild cache policy helper')
 lib.wo_run.argtypes=[ptr]*4+[C.c_int,C.c_int,C.c_float,C.c_int,ptr]
 lib.wo_cached_run.argtypes=[ptr]*3+[C.c_int,C.c_int,C.c_float,C.c_int,ptr]
 lib.wo_unpack.argtypes=[ptr]*3+[C.c_int,C.c_int,ptr]
 p.check(lib.wo_initialize());p.check(lib.wo_cached_initialize())
 policy=C.CDLL(str(ROOT/'local/thor-w4a16/cache_policy.so'))
 policy.wo_cache_limits.argtypes=[C.POINTER(C.c_int)];limits=(C.c_int*3)();p.check(policy.wo_cache_limits(limits))
 policy.wo_cache_policy.argtypes=[ptr,ptr,C.c_size_t,C.POINTER(C.c_size_t)]
 policy.wo_cache_reset.argtypes=[ptr,C.c_size_t]
 policy.wo_graph_cache_policy.argtypes=[ptr,ptr,C.c_size_t]
 eviction=torch.zeros(128*1024*1024,device='cuda',dtype=torch.uint8);torch.manual_seed(71)
 result={'scope':'matched standalone W4A16 MMA, decode each use versus globally cached BF16; no full-model claim',
  'l2_bytes':limits[0],'max_persisting_bytes':limits[1],'max_policy_window_bytes':limits[2],
  'eviction_bytes':eviction.numel(),'persistence_policy':'explicit CUDA graph kernel node attribute','provenance':p.provenance,'cases':[]}
 def write():args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
 for shape in args.shapes.split(','):
  m,k=map(int,shape.split('x'));w=p.storage(torch.randn((m,k),device='cuda',dtype=torch.bfloat16)*.02);p.quantize(w,reference=True)
  x=torch.randn(k,device='cuda',dtype=torch.bfloat16);y=torch.empty(m,device='cuda',dtype=torch.bfloat16);cache=torch.empty((m,k),device='cuda',dtype=torch.bfloat16)
  def unpack():p.check(lib.wo_unpack(w['packed'].data_ptr(),w['scales'].data_ptr(),cache.data_ptr(),m,k,p.stream))
  def packed():p.check(lib.wo_run(w['packed'].data_ptr(),w['scales'].data_ptr(),x.data_ptr(),y.data_ptr(),m,k,w['tensor_scale'],20,p.stream))
  def cached():p.check(lib.wo_cached_run(cache.data_ptr(),x.data_ptr(),y.data_ptr(),m,k,w['tensor_scale'],20,p.stream))
  unpack();packed();original=y.clone();cached();torch.cuda.synchronize()
  error=metrics(y,original)
  if not torch.equal(y,original):raise RuntimeError((shape,'cached path differs',error))
  row={'m':m,'k':k,'packed_weight_bytes':w['packed'].numel()+w['scales'].numel(),
   'decoded_cache_bytes':cache.numel()*2,'cache_vs_decode_error':error,'timings':{}}
  for name,fn in [('decode_each_use',packed),('cached_bf16',cached),('unpack_once',unpack)]:
   for cold,evict in [('warm',None),('evicted',eviction)]:row['timings'][name+'_'+cold]=timings(fn,args.repeats,1,evict)
  previous=C.c_size_t();code=policy.wo_cache_policy(p.stream,cache.data_ptr(),cache.numel()*2,C.byref(previous));row['persistence_setup_status']=code
  if not code:
   try:
    def graph_policy(graph):p.check(policy.wo_graph_cache_policy(graph,cache.data_ptr(),cache.numel()*2))
    for cold,evict in [('warm',None),('evicted',eviction)]:row['timings']['cached_bf16_persisting_'+cold]=timings(cached,args.repeats,1,evict,graph_policy=graph_policy)
   finally:p.check(policy.wo_cache_reset(p.stream,previous.value))
  result['cases'].append(row);write();print(shape,{name:round(v['median_ms']*1000,2) for name,v in row['timings'].items()},flush=True)
if __name__=='__main__':main()
