"""Report the three frozen-port diagnoses without promoting runtime changes."""
import csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
from tools.thor.repro import source_hash

def main():
 root=Path.cwd();p=root/'docs/thor/bottleneck';traffic=json.loads((p/'traffic-context1024.json').read_text());w=p/'waits'
 assert traffic['status']=='PASS'
 cohorts=[json.loads(x)for seed in [71,72,73]for x in (p/f'tma-stream-seed{seed}.jsonl').read_text().splitlines()]
 assert len(cohorts)==30 and all(len(x['samples_ms'])==30 for x in cohorts)
 assert len({x['checksum']for x in cohorts if x['consumer_reads']})==1 and all(x['checksum_checked']for x in cohorts if x['consumer_reads'])
 for tool in ['memcheck','synccheck','racecheck']:
  s=(p/f'tma-{tool}.txt').read_text();assert ('RACECHECK SUMMARY: 0 hazards'if tool=='racecheck'else'ERROR SUMMARY: 0 errors')in s
 initial=json.loads((w/'sanitizers-initial.json').read_text());assert all(x['returncode']==0 for x in initial if x['name']!='racecheck')
 race=json.loads((w/'racecheck-process.json').read_text());assert race['returncode']==0 and 'RACECHECK SUMMARY: 0 hazards'in race['output']
 runs={n:json.loads((w/f'{n}.json').read_text())for n in ['control-before','instrumented','control-after']}
 assert all(x['status']=='PASS'and len(x['rows'])==3 for x in runs.values())
 assert runs['control-before']['source_hash']==runs['control-after']['source_hash']==source_hash()
 assert len({x['instruction_sha256']for x in runs.values()})==1
 controls=[r['mean_ms_per_token']for n,x in runs.items()if n!='instrumented'for r in x['rows']];observed=[r['mean_ms_per_token']for r in runs['instrumented']['rows']]
 base=statistics.median(controls);diag=statistics.median(observed);by=defaultdict(lambda:[0]*12);counts=defaultdict(int)
 for snap in runs['instrumented']['rows']:
  checked=0
  for worker,ops in zip(snap['diag'],runs['instrumented']['opcodes']):
   for row,op in zip(worker,ops):
    if op not in (1,4,5,6,7):continue
    assert row[10]==op and row[11]>0 and min(row)>=0
    assert sum(row[:3])<=row[3] and row[4]+row[5]<=row[6] and row[7]<=row[8]
    for j,v in enumerate(row):by[op][j]+=v
    counts[op]+=1;checked+=1
  assert checked==3028
 names={1:'QKV',4:'O projection',5:'Up/gate',6:'Down projection',7:'Output head'}
 table=[]
 for op,v in sorted(by.items()):table.append({'opcode':op,'family':names[op],'rows':counts[op],'weight_wait_pct':100*v[0]/v[3],'output_slot_wait_pct':100*v[1]/v[3],'matvec_work_pct':100*v[2]/v[3],'other_loop_pct':100*(v[3]-sum(v[:3]))/v[3],'global_dependency_pct_of_input_setup':100*v[7]/v[8],'input_setup_pct_of_setup_plus_loop':100*v[8]/(v[8]+v[3])})
 total=[sum(v[j]for v in by.values())for j in range(12)]
 result={'control_median_ms':base,'instrumented_median_ms':diag,'observed_difference_percent':100*(diag/base-1),'sampled_position':1123,'sampled_warp':0,'scope':'matvec consumer loops only; role times overlap and are not wall-time fractions','families':table,'pooled_consumer_loop_percent':{'weight_wait':100*total[0]/total[3],'output_slot_wait':100*total[1]/total[3],'matvec_work':100*total[2]/total[3],'other':100*(total[3]-sum(total[:3]))/total[3]},'global_dependency_pct_of_setup_plus_loop':100*total[7]/(total[8]+total[3])}
 (w/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 with (w/'families.csv').open('w',newline='')as f:writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
 b=traffic['bytes'];lines=['# Frozen-port diagnoses: traffic, TMA concurrency and VM waits','',
'No production runtime or arithmetic changes were made. The observer exists only in a separate worktree, with its exact patch/build fingerprint preserved. These findings select possible follow-up experiments; they are not optimization speedups.','',
'## 1. Source-level traffic accounting','',
f"The unchanged execution plan has {traffic['tasks']:,} tasks. Coverage checks confirm that every linear-weight tile appears exactly once, including all four down-projection column ranges and the complete output head.",'',
'| Counted payload | Bytes per decode |','|---|---:|']
 for k,v in b.items():lines.append(f'| {k} | {v:,} |')
 lines+=['',f"Total counted payload is {traffic['total_accounted_payload_bytes']:,} bytes. Linear weights are **{100*traffic['linear_weight_fraction']:.2f}%**. Repeated activation inputs are {traffic['activation_repeated_requested_bytes']:,} bytes and repeated normalization weights are {traffic['normalization_repeated_requested_bytes']:,} bytes; together these are {100*(traffic['activation_repeated_requested_bytes']+traffic['normalization_repeated_requested_bytes'])/traffic['total_accounted_payload_bytes']:.3f}% of counted payload.",'',
'These are requested payload bytes, not DRAM transactions. Repeated small inputs may hit cache. Poll counts, cache-line amplification, spills and atomic transaction effects are not counted; their latency can matter despite small payloads. Atomic completion RMW operations and outside-VM payload estimates are recorded separately in the raw JSON. Thus this accounting supports deprioritizing bulk activation-traffic elimination, not declaring all synchronization negligible.','',
'## 2. Matching TMA transfer probe','',
'The probe uses native 16×512 BF16 TMA tiles, four chunks per 64-KiB iteration, twenty CTAs, 640 threads/CTA, sixteen consumer warps and a fixed thirteen-page allocation. One-, two- and three-stage cases change prefetch depth without changing allocation/residency. Issuer cases use one/two/four lanes in the loader warp. Model arithmetic and inter-task dependencies are absent; registers are not forced to match the full VM. All cases admit one CTA/SM.','',
'Three independently initialized batches use shuffled case orders (seeds 71/72/73), five warmups and thirty timed launches per case. Consumer-read mode reads the complete shared payload and checks its final checksum against an independent global reader after each thirty-sample case. Discard mode waits for TMA completion without shared-data consumption. Tiny-output checksum fixtures exercise ring phases under all three sanitizers.','',
'| Stages | Issuing lanes | Consume: median of batch medians ms | Batch range ms | Discard ms |','|---:|---:|---:|---:|---:|']
 for stages,issuers in [(3,1),(1,1),(2,1),(3,2),(3,4)]:
  a=[r['median_ms']for r in cohorts if r['consumer_reads']and(r['stages'],r['issuers'])==(stages,issuers)];d=[r['median_ms']for r in cohorts if not r['consumer_reads']and(r['stages'],r['issuers'])==(stages,issuers)]
  lines.append(f'| {stages} | {issuers} | {statistics.median(a):.4f} | {min(a):.4f}–{max(a):.4f} | {statistics.median(d):.4f} |')
 lines+=['',
'The matching three-stage/single-issuer path reaches about 260 GB/s in this arithmetic-free probe. More issuing lanes do not materially improve it. One stage is slightly faster in the microbenchmark, but it has no matvec computation to overlap, so this does not establish a one-stage VM improvement. The probe demonstrates that one resident CTA/SM and one issuing lane can supply high bandwidth; a blanket shortage of loader concurrency is not established.','',
'## 3. Scoped VM observer','',
f"A control–instrumented–control sequence uses five full 100-step warmup requests and three measured 100-step requests per phase. Control median is **{base:.4f} ms/token**, instrumented median **{diag:.4f} ms/token** ({100*(diag/base-1):+.3f}%). This shows no measurable slowdown in this short check, not proof that clocks are free. Compiled register count, static SMEM, local storage and reported spill sizes remain unchanged.",'',
'Cycle snapshots are from the final position 1123 of each measured request. Only consumer warp 0 and loader lane 0 are sampled; 3,028 matvec instructions per snapshot are covered. Attention compute and other consumer warps are not sampled. All comparisons use elapsed clocks on the same SM; absolute clocks are not compared across SMs. Wait regions include primitive/timestamp costs and are not exact hardware stall counters.','',
'| Family | Weight wait % | Output-slot wait % | Matvec work % | Other loop % |','|---|---:|---:|---:|---:|']
 for r in table:lines.append(f"| {r['family']} | {r['weight_wait_pct']:.2f} | {r['output_slot_wait_pct']:.2f} | {r['matvec_work_pct']:.2f} | {r['other_loop_pct']:.2f} |")
 lines+=['',f"Pooled sampled consumer-loop cycles: weight readiness {result['pooled_consumer_loop_percent']['weight_wait']:.2f}%, output-slot availability {result['pooled_consumer_loop_percent']['output_slot_wait']:.2f}%, matvec work {result['pooled_consumer_loop_percent']['matvec_work']:.2f}%. Separately, global dependency waits account for {result['global_dependency_pct_of_setup_plus_loop']:.2f}% of sampled input-setup-plus-loop cycles.",'',
'**These percentages are not fractions of end-to-end latency and cannot be treated as recoverable time.** Loaders, storers and other workers progress while the sampled consumer waits. An output-slot wait may reflect storer reduction/publication or late consumer arrivals; the observer does not distinguish those causes. This is evidence of family-dependent pipeline backpressure, not a universal weight-prefetch shortage.','',
'Memcheck and synccheck pass for the observer. Initial racecheck reports zero hazards but the strict timing-value assertion fails because sanitizer-slowed output-head execution exceeds the 32-bit clock representation/range. That failed invocation is preserved. A second racecheck checks row ownership/bounds while skipping clock arithmetic, and passes with zero hazards; its twenty clock-range rows are output-head rows. Performance-cycle invariants are checked on every non-sanitized snapshot. Sanitizer timings are excluded from performance interpretation.','',
'## Decision for follow-up','',
'- Large reductions from eliminating repeated activations are not supported by the counted traffic volume.',
'- More weight-issuing lanes are not supported by the TMA probe.',
'- The output-slot/storer completion path is the strongest new runtime lead, particularly for paired-output tasks. First distinguish storer work/publication from late-arriving consumers before selecting a change.',
'- Weight waits remain prominent for long head/down-projection tasks; any output-path change must preserve their prefetch overlap.',
'- Hardware traffic/stall counters remain permission-blocked, so actual DRAM saturation versus internal stalls is not fully resolved.','',
'## Reproduction and evidence','',
'`traffic_accounting.py` runs against the unchanged native adapter and asserts exact instruction coverage. `tma_stream.cu` uses the pinned TK overlay; compile with C++20, `-DKITTENS_HOPPER`, native SM110a codegen, relaxed constexpr and extended lambda flags, the overlay include and `-lcuda`. Seeds are command-line arguments. Sanitizer builds use `TMA_DIAG_TILES=160`, `TMA_DIAG_WARMUPS=0`, `TMA_DIAG_REPEATS=1`; their timing is not a performance result.','',
'The observer is generated by `tools/analysis/instrument_waits.py` in an isolated linked worktree. `waits/observer-source.json` preserves the exact patch against `87fd234`; `waits/native-build.json` and `waits/build.txt` identify the actual binary/resources. The worktree reuses the original local dependencies/model and pinned TK headers. Production source and binary remain unchanged. `collect_waits.py` is identical in both checkouts, with `THOR_BINARY_DIR` selecting the diagnostic binary only in its worktree. `waits/processes.json` records the sequential collection commands.','',
'Raw traffic, all three shuffled TMA cohorts, sanitizer output, three VM snapshots, cycle schema and CSV summaries are under [bottleneck/](bottleneck/). Clock columns: 0 weight wait, 1 output-slot wait, 2 matvec work, 3 consumer-loop span, 4 loader reuse wait, 5 loader page wait, 6 loader-loop span, 7 global dependency wait, 8 consumer input-setup span, 9 unused, 10 opcode, 11 matvec iterations. The clock fields are per-instruction local-cycle durations; some are nested or overlap across roles.','']
 (root/'docs/thor/DIAGNOSES.md').write_text('\n'.join(lines))
 sources=['traffic_accounting.py','tma_stream.cu','instrument_waits.py','collect_waits.py','check_waits.py','report_diagnoses.py']
 (p/'diagnosis-provenance.json').write_text(json.dumps({'production_source_hash':source_hash(),'production_binary_sha256':runs['control-before']['metadata']['native_build']['binary_sha256'],'diagnostic_source_hash':runs['instrumented']['source_hash'],'analysis_sources':{x:hashlib.sha256((root/'tools/analysis'/x).read_bytes()).hexdigest()for x in sources}},indent=2)+'\n')
 print(json.dumps({'status':'PASS','control_ms':base,'instrumented_ms':diag,'loop_percent':result['pooled_consumer_loop_percent']}))
if __name__=='__main__':main()
