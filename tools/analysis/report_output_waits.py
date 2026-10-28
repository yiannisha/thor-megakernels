"""Validate and summarize isolated output observer snapshots (not wall-time shares)."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory
x=json.loads((d/'instrumented.json').read_text());groups={}
assert x['status']=='PASS' and x['initial_check']['passed']
for f in ['control-before.json','control-after.json']:
 control=json.loads((d/f).read_text())
 assert control['status']=='PASS' and control['instruction_sha256']==x['instruction_sha256']
for run in x['rows']:
 count=0
 for rows,ops in zip(run['diag'],x['opcodes']):
  for r,op in zip(rows,ops):
   if op not in (1,4,5,6,7):continue
   assert len(r)==18 and min(r)>=0 and r[10]==op and r[11]>0
   assert sum(r[:3])<=r[3] and r[4]+r[5]<=r[6] and r[7]<=r[8]
   assert sum(r[12:15])<=r[15] and r[16]+r[17]<=r[13],r
   g=groups.setdefault(op,[0]*18)
   for i,v in enumerate(r):g[i]+=v
   count+=1
 assert count==3028,count
names={1:'QKV',4:'O projection',5:'Up/gate',6:'Down projection',7:'Head'}
result={'scope':'Three final-token snapshots, position 1123; local SM cycles summed by role; not end-to-end critical-path fractions','families':{}}
for op,r in groups.items():
 result['families'][names[op]]={'opcode':op,'consumer_output_wait_pct':100*r[1]/r[3], 'storer_arrival_wait_pct':100*r[12]/r[15], 'storer_store_call_pct':100*r[13]/r[15], 'storer_slot_publication_pct':100*r[14]/r[15], 'tma_completion_pct_of_store':100*r[16]/r[13], 'global_publication_pct_of_store':100*r[17]/r[13], 'remaining_store_pct':100*(r[13]-r[16]-r[17])/r[13]}
controls=[r['mean_ms_per_token'] for f in ['control-before.json','control-after.json'] for r in json.loads((d/f).read_text())['rows']]
times=[r['mean_ms_per_token']for r in x['rows']]
result.update(control_ms=controls,instrumented_ms=times,control_median_ms=statistics.median(controls),instrumented_median_ms=statistics.median(times),observer_change_pct=100*(statistics.median(times)/statistics.median(controls)-1))
(d/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
