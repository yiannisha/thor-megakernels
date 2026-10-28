"""Compare native ready dispatch with static and matched instruction-join controls."""
import argparse
import csv
import json
from pathlib import Path
from tools.thor.bench import summarize
from tools.thor.report import measured


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rows=[];common=None;samples_by_mode={}
    for mode in ['static','joined','ready']:
        files=sorted(args.folder.glob(f'benchmark-{mode}-batch*.json'))
        if len(files)!=3:raise ValueError('three independent batches required')
        data=[json.loads(p.read_text())for p in files]
        for d in data:
            if not measured(d):raise ValueError('unqualified benchmark')
            if any(d['protocol'].get(k)!=v for k,v in {'warmups':5,'repeats':30,'batches':1}.items()):raise ValueError('unexpected timing protocol')
            m=d['metadata'];cfg=m['config']
            if cfg['ready_queue']!=(mode=='ready') or cfg['serial_instruction_handoff']!=(mode!='static'):raise ValueError('wrong queue policy')
            current={k:m[k]for k in ['model_provenance','input_hash','context','batch','precision','gate','prefill_attention','reference_attention','float32_matmul_precision','same_cache_replay']}
            current.update(config={k:v for k,v in cfg.items()if k not in ('name','ready_queue','serial_instruction_handoff')},source_hash=m['native_build']['source_hash'])
            if common is not None and common!=current:raise ValueError('incomparable workloads/configurations')
            common=current
            if mode=='ready':
                for t in d['trials']:
                    for c in t['correctness_checks']:
                        if c.get('ready_coverage',{}).get('passed')is not True:raise ValueError('missing task coverage')
        samples=[x for d in data for t in d['trials']for x in t['samples_ms']];samples_by_mode[mode]=samples
        rows.append({'mode':mode,'samples':len(samples),**summarize(samples),
                     'batch_min_p50_ms':min(d['summary']['p50_ms']for d in data),
                     'batch_max_p50_ms':max(d['summary']['p50_ms']for d in data)})
    q=json.loads((args.folder/'temporal-native-past.json').read_text())
    failed=[r['step']for r in q['rows']if not r['passed']]
    if len(q['rows'])!=100 or q['status']!=('FAIL' if failed else 'PASS'):raise ValueError('incomplete/inconsistent temporal result')
    lines=['# Model-level native ready dispatch','',
           '| Mode | Checks/samples | p50 ms | p95 ms | p99 ms | Batch p50 range ms |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['mode']} | {r['samples']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['p99_ms']:.3f} | {r['batch_min_p50_ms']:.3f}–{r['batch_max_p50_ms']:.3f} |")
    overhead=100*(rows[2]['p50_ms']/rows[0]['p50_ms']-1)
    lines+=['',f'Ready dispatch has {overhead:+.1f}% higher observed median than the ordinary static control in this cohort. The joined static control uses the same execution-role instruction join as ready dispatch while retaining its existing controller prefetch. Ready dispatch also adds CSR/atomic bookkeeping, workspace initialization and serialized controller dispatch. These observations do not isolate the cost of any one component.','',
            'Native batch-one BF16 Llama-3.2-1B-Instruct, context 1024, same-cache replay; three independently loaded batches, five warmups and thirty checked samples per batch, shuffled order with seed 71. Embedding/runtime/head/argmax are included; loading, prefill, cache restore and checks are excluded. Uncontrolled shared GPU load prevents an isolated causal speedup claim. Ready coverage is checked after every measured sample.','',
            f"Ready accumulated-native-cache qualification: **{q['status']}**, failed steps {failed}; worst logit NRMSE {100*q['summary']['worst_logit_nrmse']:.4f}%. All {q['summary']['greedy_matches']}/100 greedy tokens agree. The failure reproduces the SIMT baseline; task coverage and memory-order evidence do not waive numerical limits.",'',
            'Recommendation: retain ready dispatch as an opt-in prototype. The tested implementation is slower and does not justify replacing static queues. See `MODEL_READY_QUEUE.md` for memory/progress invariants, coverage and limitations.','',
            f'![Latency distributions]({args.output.with_suffix(".svg").name})','']
    args.output.write_text('\n'.join(lines))
    fields=['mode','samples','p50_ms','p95_ms','p99_ms','batch_min_p50_ms','batch_max_p50_ms']
    with args.output.with_suffix('.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for mode,values in samples_by_mode.items():
        values=sorted(values);ax.step(values,[(i+1)/len(values)for i in range(len(values))],where='post',label=mode)
    ax.set(xlabel='Decode replay CUDA-event interval (ms)',ylabel='Empirical cumulative probability',title='Model-level ready dispatch; shared Thor, context 1024');ax.grid(alpha=.25);ax.legend();fig.tight_layout();svg=args.output.with_suffix('.svg');fig.savefig(svg);plt.close(fig)
    svg.write_text('\n'.join(line.rstrip()for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
