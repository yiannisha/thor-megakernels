"""Report same-cache SIMT/BF16-MMA matvec A/B with temporal failures visible."""
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
    args=parser.parse_args();rows=[];signature=None;cohorts={}
    for mode in ['simt','mma']:
        files=sorted(args.folder.glob(f'benchmark-{mode}-batch*.json'))
        if len(files)!=3:raise ValueError('three independently loaded batches required')
        data=[json.loads(f.read_text()) for f in files]
        for d in data:
            if not measured(d):raise ValueError('unqualified timing batch')
            if any(d['protocol'][k]!=v for k,v in {'warmups':5,'repeats':30,'batches':1}.items()):raise ValueError('unexpected protocol')
            m=d['metadata'];cfg={k:v for k,v in m['config'].items() if k not in ('name','tensorcore_matvec')}
            current={k:m[k] for k in ['model_provenance','input_hash','context','precision','batch','gate','prefill_attention','reference_attention','float32_matmul_precision','same_cache_replay']}
            current.update(config=cfg,source_hash=m['native_build']['source_hash'])
            if signature is not None and current!=signature:raise ValueError('incomparable batches')
            signature=current
            if m['config']['tensorcore_matvec']!=(mode=='mma'):raise ValueError('wrong native mode')
        samples=[x for d in data for t in d['trials'] for x in t['samples_ms']]
        cohorts[mode]=samples
        rows.append({'mode':mode,'samples':len(samples),**summarize(samples),
                     'batch_min_p50_ms':min(d['summary']['p50_ms']for d in data),
                     'batch_max_p50_ms':max(d['summary']['p50_ms']for d in data)})
    q=json.loads((args.folder/'temporal-native-past.json').read_text())
    if len(q['rows'])!=100:raise ValueError('incomplete temporal test')
    failed=[r['step']for r in q['rows']if not r['passed']]
    if q['status']!=('FAIL' if failed else 'PASS'):raise ValueError('inconsistent temporal status')
    lines=['# Thor matvec: SIMT versus BF16 warp MMA','',
           '| Mode | Samples | p50 ms | p95 ms | p99 ms | Batch p50 range ms |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['mode']} | {r['samples']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['p99_ms']:.3f} | {r['batch_min_p50_ms']:.3f}–{r['batch_max_p50_ms']:.3f} |")
    diff=100*(rows[1]['p50_ms']/rows[0]['p50_ms']-1)
    worst=max(r['metrics']['logits']['normalized_rmse']for r in q['rows'])
    lines+=['',f'Observed pooled median difference: {diff:+.2f}%. These are context-1024 batch-one BF16 same-cache replays with matching weights and numerical/resource settings, five warmups and thirty checked samples in each of three batches. Embedding, runtime, head and argmax are included; loading, prefill, cache restoration and checking are excluded. Shared GPU load is uncontrolled. Seeded shuffling happened to put MMA before SIMT in each batch, leaving an order confound. The difference does not establish a controlled speedup.','',
            f"MMA accumulated-cache numerical qualification: **{q['status']}**, {len(failed)}/100 failed steps, worst logit NRMSE {100*worst:.4f}%. Failed steps: {failed}. Matching greedy tokens does not waive failures. The earlier SIMT baseline fails 3/100 steps; MMA fails more steps. No quality-matched generation speedup or default promotion is claimed.",'',
            'The MMA candidate separately passes all three native sanitizer tools and 45 frozen-input checks across contexts 128/1024/8192 and seeds 1/2/3. See `TENSORCORE_MATVEC.md` for implementation, pilot, resources and reproduction.','',
            f'![Observed latency distributions]({args.output.with_suffix(".svg").name})','']
    args.output.write_text('\n'.join(lines))
    with args.output.with_suffix('.csv').open('w',newline='') as stream:
        fields=['mode','samples','p50_ms','p95_ms','p99_ms','batch_min_p50_ms','batch_max_p50_ms'];w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for mode,values in cohorts.items():
        values=sorted(values);ax.step(values,[(i+1)/len(values)for i in range(len(values))],where='post',label=mode)
    ax.set(xlabel='Decode replay CUDA-event interval (ms)',ylabel='Empirical cumulative probability',title='Thor BF16 matvec; shared GPU, context 1024');ax.grid(alpha=.25);ax.legend();fig.tight_layout();svg=args.output.with_suffix('.svg');fig.savefig(svg);plt.close(fig)
    svg.write_text('\n'.join(line.rstrip()for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
