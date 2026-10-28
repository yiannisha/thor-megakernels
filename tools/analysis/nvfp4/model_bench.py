"""Full Llama decode through native FP4 MMA in the existing VM.

Loading, weight packing, prefill and cache restore are outside decode timing.
Attention/RoPE/residuals/norms remain BF16/FP32. This is an experimental model
adapter, not the enabling-only BF16 production entry point.
"""
import argparse
import ctypes as C
import hashlib
import importlib
import json
import statistics
import sys
import time
from pathlib import Path
import torch
from .bench import Primitive,ROOT
from tools.thor.model_bench import ModelAdapter
from tools.thor.repro import source_hash

class FP4ModelAdapter(ModelAdapter):
 @torch.inference_mode()
 def __init__(self,context=1024,seed=1,decode_steps=100,weight_only=False,mixed_fp8=False):
  if weight_only and mixed_fp8:raise ValueError('select one precision backend')
  super().__init__('torch',context,seed,cache_capacity=context+decode_steps+1)
  from megakernels.demos.latency.scheduler import LatencyScheduleBuilder
  from megakernels.demos.latency.mk import interpret_with_mk
  self.globs=LatencyScheduleBuilder.make_globals(self.model)
  self.globs.instructions=torch.zeros((20,1,32),device='cuda',dtype=torch.int32)
  self.globs.timings=torch.zeros((20,1,128),device='cuda',dtype=torch.int32)
  self.globs.pos_id=context
  sys.path.insert(0,str(ROOT/'local/thor-native-fp4-vm'))
  self.module=importlib.import_module('mk_fp4_model')
  self.p=Primitive(Path(self.module.__file__))
  if self.p.provenance.get('framework_source_hash')!=source_hash():raise RuntimeError('stale FP4 model framework binary')
  for name,digest in self.p.provenance.get('model_sources',{}).items():
   if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'stale FP4 model source: {name}')
  self.weights=[];self.inputs=[];self.outputs=[];self.plans=[]
  self.qkv=torch.zeros(3072,device='cuda',dtype=torch.bfloat16)
  self.upgate=torch.zeros(16384,device='cuda',dtype=torch.bfloat16)
  matrices=[];sources=[];destinations=[]
  g=self.globs
  for layer in range(16):
   matrices += [g.qkv_proj_weights[layer],g.o_proj_weights[layer],
                torch.cat((g.up_proj_weights[layer],g.gate_proj_weights[layer])),g.down_proj_weights[layer]]
   sources += [g.hidden_states,g.attn_out,g.hidden_states,self.upgate[:8192]]
   destinations += [self.qkv,g.hidden_states,self.upgate,g.hidden_states]
  matrices.append(g.lm_head_weights.data);sources.append(g.hidden_states);destinations.append(g.logits)
  print('Packing 65 model matrices into resident NVFP4 weights',flush=True)
  for i,(matrix,source,destination) in enumerate(zip(matrices,sources,destinations)):
   w=self.p.storage(matrix);w['decoded']=None
   self.p.quantize(w)
   x=self.p.storage(source.reshape(1,-1));x['decoded']=None;x['tensor_scale']=1/64
   handle=self.p.prepare(w,x,destination)
   self.weights.append(w);self.inputs.append(x);self.outputs.append(destination);self.plans.append(handle)
  torch.cuda.synchronize()
  ptr=C.c_void_p;lib=self.p.lib
  lib.vm_prepare.argtypes=[C.POINTER(ptr)]*5+[C.POINTER(C.c_float)]*2+[C.c_int]*7
  lib.vm_prepare.restype=ptr
  def pointers(values):return (ptr*len(values))(*values)
  def floats(values):return (C.c_float*len(values))(*values)
  self.vm=lib.vm_prepare(pointers(self.plans),pointers([x['source'].data_ptr() for x in self.inputs]),
   pointers([x['packed'].data_ptr() for x in self.inputs]),pointers([x['native'].data_ptr() for x in self.inputs]),
   pointers([x.data_ptr() for x in self.outputs]),floats([x['tensor_scale'] for x in self.inputs]),
   floats([w['tensor_scale']*x['tensor_scale'] for w,x in zip(self.weights,self.inputs)]),65,20,1,1,0,20,0)
  if not self.vm:raise RuntimeError(lib.nv_error().decode())
  norms=[]
  for layer in range(16):norms += [g.attn_ln_weights[layer].data_ptr(),g.mlp_ln_weights[layer].data_ptr()]
  norms.append(g.lm_head_norm_weights.data_ptr())
  result=[]
  interpret_with_mk(g,lambda *args,**kwargs:result.append(self.module.configure(self.vm,norms,*args)))
  self.handle=result[0];self.engine='fp4';self.weight_only=weight_only;self.mixed_fp8=mixed_fp8
  if weight_only:
   self.bf16_activations=[torch.empty_like(x['source']) for x in self.inputs]
   self.module.weight_only(self.handle,[x.data_ptr() for x in self.bf16_activations],[w['scales'].data_ptr() for w in self.weights],[w['tensor_scale'] for w in self.weights])
  self.metadata.update(engine='fp4',precision='NVFP4 W4A4 linear, BF16/FP32 attention and nonlinear operations',
   native_build=self.p.provenance,variant='native-fp4-vm-model',activation_tensor_scale=1/64,
   quantized_weight_bytes=sum(w['packed'].numel()+w['native'].numel() for w in self.weights))
  if weight_only:self.metadata.update(precision='NVFP4 weights, BF16 activations and MMA, FP32 accumulation',variant='native-w4a16-vm-model')
  if mixed_fp8:
   from .mixed_format import weights
   self.weights=[weights(w['source']) for w in self.weights]
   self.fp8_activations=[torch.zeros(x['source'].numel(),device='cuda',dtype=torch.uint8) for x in self.inputs]
   self.fp8_scales=[torch.zeros((x['source'].numel()//128)*512,device='cuda',dtype=torch.uint8) for x in self.inputs]
   self.module.mixed8(self.handle,[w['packed'].data_ptr() for w in self.weights],[w['native'].data_ptr() for w in self.weights],
    [x.data_ptr() for x in self.fp8_activations],[x.data_ptr() for x in self.fp8_scales])
   self.metadata.update(precision='MXFP4 block32 weights, MXFP8 E4M3 block32 activations, FP32 accumulation',variant='native-w4a8-vm-model',
    activation_tensor_scale=None,quantized_weight_bytes=sum(w['packed'].numel()+w['native'].numel() for w in self.weights))
  self.reset()
  print('Full-model native FP4 VM is configured',flush=True)
 @torch.inference_mode()
 def run(self):
  if self.engine=='torch':return self._run_torch()
  from megakernels.model_types import BatchState
  self.globs.hidden_states.copy_(self.model.model.embed_tokens(BatchState(input_ids=self.case.token)).hidden_states.flatten())
  self.globs.barriers.zero_()
  self.module.run(self.handle,self.context,torch.cuda.current_stream().cuda_stream)
  self.last_hidden=self.globs.hidden_states
  self.last_token=self.globs.logits.argmax()
  return self.globs.logits

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--weight-only',action='store_true');ap.add_argument('--mixed-fp8',action='store_true');ap.add_argument('--quantized-reference',action='store_true');ap.add_argument('--shared-attention-reference',action='store_true');ap.add_argument('--context',type=int,default=1024);ap.add_argument('--seed',type=int,default=1)
 ap.add_argument('--decode-steps',type=int,default=100);ap.add_argument('--warmups',type=int,default=2)
 ap.add_argument('--repeats',type=int,default=5);ap.add_argument('--batches',type=int,default=1);ap.add_argument('--check-only',action='store_true')
 ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 result={'status':'RUNNING','protocol':vars(args)|{'output':str(args.output)},'requests':[]}
 def write():args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
 write()
 try:
  with torch.inference_mode():
   adapter=FP4ModelAdapter(args.context,seed=args.seed,decode_steps=args.decode_steps,weight_only=args.weight_only,mixed_fp8=args.mixed_fp8)
   result['metadata']=adapter.metadata
   adapter.reset();logits=adapter.run();torch.cuda.synchronize()
   result['first_decode']=adapter.check(logits,adapter.reference())
   if not bool(logits.isfinite().all()):raise RuntimeError('nonfinite FP4 model logits')
   print('First decode metrics:',result['first_decode'],flush=True)
   if args.quantized_reference:
    from .model_reference import check_quantized
    result['matched_quantized_reference']=check_quantized(adapter,native_attention=args.shared_attention_reference)
    print('Matched quantized reference:',result['matched_quantized_reference'],flush=True)
    write()
   first=adapter.case.token.clone()
   def run():
    adapter.reset();adapter.case.token.copy_(first)
    generated=torch.empty(args.decode_steps+1,device='cuda',dtype=torch.long);generated[0]=first.item()
    torch.cuda.synchronize();start=time.perf_counter()
    for step in range(args.decode_steps):
     adapter.context=args.context+step
     adapter.run();generated[step+1]=adapter.last_token
     adapter.case.token.copy_(adapter.last_token.reshape(1,1))
    ids=generated.cpu().tolist();elapsed=(time.perf_counter()-start)*1000
    return dict(tpot_ms=elapsed/args.decode_steps,total_ms=elapsed,token_ids=ids)
   if not args.check_only:
    for _ in range(args.warmups):run()
    for batch in range(args.batches):
     for i in range(args.repeats):
      r=run();r['batch']=batch;result['requests'].append(r);write();print('batch',batch,'request',i,'ms/token',r['tpot_ms'],flush=True)
    result['median_ms_per_token']=statistics.median(r['tpot_ms'] for r in result['requests'])
   result['status']='PASS'
 except Exception as e:result.update(status='FAIL',error=f'{type(e).__name__}: {e}');raise
 finally:write()
if __name__=='__main__':main()
