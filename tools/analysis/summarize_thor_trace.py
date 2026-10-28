"""Summarize only the marked decode range; do not add overlapping API wait time."""
import argparse,json,sqlite3
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('sqlite',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 c=sqlite3.connect(args.sqlite);c.row_factory=sqlite3.Row
 span=c.execute("select start,end from NVTX_EVENTS where text='decode_only'").fetchone();start,end=span
 kernels=list(c.execute('select k.start,k.end,s.value name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.start>=? and k.end<=?',(start,end)))
 intervals=[(r['start'],r['end'])for r in kernels]
 tables={r[0]for r in c.execute("select name from sqlite_master where type='table'")}
 for table in ['CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
  if table in tables:intervals.extend((r[0],r[1])for r in c.execute(f'select start,end from {table} where start>=? and end<=?',(start,end)))
 covered=0;last_end=start
 for a,b in sorted(intervals):
  covered+=max(0,b-max(last_end,a));last_end=max(last_end,b)
 native=[r for r in kernels if 'megakernel::mk<'in r['name']];n=len(native);native_ns=sum(r['end']-r['start']for r in native);total=end-start
 apis=[dict(r)for r in c.execute('select s.value name,count(*) calls,sum(k.end-k.start)/1e6 total_ms from CUPTI_ACTIVITY_KIND_RUNTIME k join StringIds s on s.id=k.nameId where k.start>=? and k.end<=? group by k.nameId order by total_ms desc',(start,end))]
 binding=c.execute("select sum(end-start),count(*) from NVTX_EVENTS where text='native_binding' and start>=? and end<=?",(start,end)).fetchone()
 result={'steps':n,'decode_range_ms':total/1e6,'per_step_ms':total/1e6/n,'native_kernel_ms_per_step':native_ns/1e6/n,'native_kernel_fraction_of_range':native_ns/total,'gpu_activity_union_ms_per_step':covered/1e6/n,'uncovered_gpu_time_ms_per_step':(total-covered)/1e6/n,'binding_host_wall_ms_per_step':binding[0]/1e6/binding[1],'api_durations':apis,'note':'API durations may block while GPU executes; they overlap GPU activity and must not be added to it. Uncovered GPU time includes launch/dependency gaps and profiler overhead; it is not all removable host overhead.'}
 args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='api_durations'}))
if __name__=='__main__':main()
