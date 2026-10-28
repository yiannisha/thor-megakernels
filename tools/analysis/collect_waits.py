"""Measure observer disturbance and collect per-instruction local-clock spans."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from tools.thor.model_bench import ModelAdapter
from tools.thor.repro import source_hash

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);args=p.parse_args()
 result={'status':'RUNNING','scope':'same benchmark loop; diagnostic buffer copied after timing','rows':[],'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 with torch.inference_mode():
  a=ModelAdapter('native',1024,1,cache_capacity=1125);first=a.case.token.clone();a.reset();output=a.run();result['initial_check']=a.check(output,a.reference());torch.cuda.synchronize()
  def run():
   a.reset();a.case.token.copy_(first);ids=torch.empty(100,device='cuda',dtype=torch.long);torch.cuda.synchronize();started=time.perf_counter()
   for step in range(100):
    a.context=1024+step;a.position.fill_(a.context);a.state.seq_len=a.context+1;a.run()
    ids[step]=a.last_token;a.case.token.copy_(a.last_token.reshape(1,1))
   tokens=ids.cpu().tolist();elapsed=(time.perf_counter()-started)*1000
   return elapsed/100,tokens
  for _ in range(5):run()
  for _ in range(args.repeats):
   ms,tokens=run();row={'mean_ms_per_token':ms,'tokens':tokens}
   if hasattr(a.globs,'diag'):row['diag']=a.globs.diag.cpu().tolist()
   result['rows'].append(row)
  result.update(status='PASS',metadata=a.metadata,source_hash=source_hash(),instrumented=hasattr(a.globs,'diag'),opcodes=a.globs.instructions[:,:,0].cpu().tolist(),instruction_sha256=hashlib.sha256(a.globs.instructions.cpu().numpy().tobytes()).hexdigest())
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','instrumented':result['instrumented'],'ms_per_token':[r['mean_ms_per_token']for r in result['rows']]}))
if __name__=='__main__':main()
