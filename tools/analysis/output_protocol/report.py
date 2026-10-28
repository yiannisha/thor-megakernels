"""Compare actual full-decode latency; all samples are request TPOTs."""
from pathlib import Path
import json,statistics
root=Path(__file__).resolve().parents[3];directory=root/'docs/thor/output-protocol-results'
def pct(xs,p):
 xs=sorted(xs);i=(len(xs)-1)*p;lo=int(i);return xs[lo]+(xs[min(lo+1,len(xs)-1)]-xs[lo])*(i-lo)
def summarize(d,variants,baseline):
 result={};inputs=set()
 for name in variants:
  samples=[];blocks=[];qual=[]
  for block in range(3):
   x=json.loads((d/f'{name}-{block}.json').read_text());assert x['status']=='PASS',x['status']
   vals=[r['tpot_ms']for t in x['trials']for r in t['requests']];assert len(vals)==10
   samples+=vals;blocks.append(statistics.median(vals));qual.append(x['qualification']['passed']);inputs.add(x['metadata']['input_hash'])
  result[name]={'requests':len(samples),'excluded_initial_queue':name in ('async','combined'),'p50_ms':statistics.median(samples),'p95_ms':pct(samples,.95),'p99_ms':pct(samples,.99),'block_p50_ms':blocks,'first_step_reference_pass':qual}
 assert len(inputs)==1
 for name,x in result.items():
  x['median_latency_reduction_pct']=100*(1-x['p50_ms']/result[baseline]['p50_ms'])
  x['block_latency_reduction_pct']=[100*(1-v/b)for v,b in zip(x['block_p50_ms'],result[baseline]['block_p50_ms'])]
 (d/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(d.name,json.dumps(result,indent=2))
 return result
if __name__=='__main__':
 summarize(directory,['baseline','early','release','async','combined'],'baseline')
 if (directory/'confirmation/batched-2.json').exists():summarize(directory/'confirmation',['original','release','batched'],'original')
 if (directory/'queue-v2/queue_v2-2.json').exists():summarize(directory/'queue-v2',['original','queue_v2'],'original')
