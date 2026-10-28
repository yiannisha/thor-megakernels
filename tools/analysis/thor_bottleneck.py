"""Profile the unchanged native port's autoregressive hot path; no runtime tuning."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from tools.thor.model_bench import ModelAdapter
from tools.thor.repro import source_hash

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',action='store_true');p.add_argument('--steps',type=int,default=100);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 with torch.inference_mode():
  a=ModelAdapter('native',1024,1,cache_capacity=1024+max(args.steps,100)+1);first=a.case.token.clone()
  for _ in range(5):
   a.reset();a.case.token.copy_(first)
   for step in range(100):
    a.context=1024+step;a.position.fill_(a.context);a.state.seq_len=a.context+1
    a.run();a.case.token.copy_(a.last_token.reshape(1,1))
  a.reset();a.case.token.copy_(first);ids=torch.empty(args.steps,device='cuda',dtype=torch.long)
  original=a.interpret
  def wrapped(*argv,**kwargs):
   torch.cuda.nvtx.range_push('native_binding')
   try:return original(*argv,**kwargs)
   finally:torch.cuda.nvtx.range_pop()
  a.interpret=wrapped
  torch.cuda.synchronize()
  if args.capture:torch.cuda.cudart().cudaProfilerStart()
  start=time.perf_counter();torch.cuda.nvtx.range_push('decode_only')
  for step in range(args.steps):
   torch.cuda.nvtx.range_push('step_'+str(step));a.context=1024+step;a.position.fill_(a.context);a.state.seq_len=a.context+1
   a.run();ids[step]=a.last_token;a.case.token.copy_(a.last_token.reshape(1,1));torch.cuda.nvtx.range_pop()
  tokens=ids.cpu().tolist();torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
  torch.cuda.nvtx.range_pop()
  if args.capture:torch.cuda.cudart().cudaProfilerStop()
  result={'status':'PASS','scope':'successive native decode steps; final token IDs copied to CPU','steps':args.steps,'warmup_requests':5,'warmup_steps':100,'capture':args.capture,'elapsed_ms':elapsed,'mean_ms_per_step':elapsed/args.steps,'generated_tokens':tokens,'metadata':a.metadata,'source_hash':source_hash(),'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','mean_ms_per_step':elapsed/args.steps}))
if __name__=='__main__':main()
