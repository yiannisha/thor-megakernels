"""Validate observer ownership; check short-run clock spans outside slow sanitizers."""
import argparse,json
import torch
from tools.thor.model_bench import ModelAdapter
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--allow-clock-range-overflow',action='store_true');args=p.parse_args()
with torch.inference_mode():
 a=ModelAdapter('native',1024,1,cache_capacity=1125);a.reset();a.run();torch.cuda.synchronize()
 diag=a.globs.diag.cpu().tolist();ops=a.globs.instructions[:,:,0].cpu().tolist();checked=0;range_rows=[]
 for worker,(rows,opcodes)in enumerate(zip(diag,ops)):
  for index,(row,op)in enumerate(zip(rows,opcodes)):
   if op in (1,4,5,6,7):
    assert row[10]==op and row[11]>0,('ownership',worker,index,op,row)
    valid=min(row)>=0 and sum(row[:3])<=row[3] and row[4]+row[5]<=row[6] and row[7]<=row[8]
    if not valid:range_rows.append({'worker':worker,'index':index,'opcode':op,'row':row})
    if not args.allow_clock_range_overflow:assert valid,range_rows[-1:]
    checked+=1
 assert checked==3028
 print(json.dumps({'status':'PASS','observer_rows_checked':checked,'clock_invariants_skipped':args.allow_clock_range_overflow,'clock_range_rows':len(range_rows),'clock_range_examples':range_rows[:3]}))
