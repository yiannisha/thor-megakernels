"""Report balanced O-projection activation-reuse comparisons and identity checks."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from tools.thor.bench import summarize
from tools.thor.report import measured


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();modes=['baseline','static','static-reuse','ready','ready-reuse']
    orders=json.loads((args.folder/'benchmark-orders.json').read_text())['orders']
    if len(orders)!=5 or any(sorted(order)!=sorted(modes)for order in orders):raise ValueError('incomplete benchmark design')
    if any(Counter(order[position]for order in orders)!=Counter(modes)for position in range(5)):raise ValueError('order positions are not balanced')
    common=None;rows=[];cohorts={};images={};structures={}
    for mode in modes:
        files=sorted(args.folder.glob(f'benchmark-{mode}-batch*.json'))
        if len(files)!=5:raise ValueError('five independent batches required')
        data=[json.loads(f.read_text())for f in files]
        for d in data:
            if not measured(d):raise ValueError('unqualified benchmark')
            if any(d['protocol'].get(k)!=v for k,v in {'warmups':5,'repeats':30,'batches':1}.items()):raise ValueError('unexpected protocol')
            m=d['metadata'];c=m['config'];family='ready'if mode.startswith('ready')else'baseline'if mode=='baseline'else'static'
            if m['engine']!='native' or c['ready_queue']!=(family=='ready') or c['o_proj_grouped']!=mode.endswith('reuse'):raise ValueError('wrong backend/host schedule')
            if c['handoff_after_opcode']!=(2 if family=='static'else -1):raise ValueError('wrong retirement control')
            current={k:m[k]for k in ['model_provenance','input_hash','context','batch','precision','gate','prefill_attention','reference_attention','float32_matmul_precision','same_cache_replay']}
            current.update(config={k:v for k,v in c.items()if k not in ('name','ready_queue','serial_instruction_handoff','o_proj_grouped','handoff_after_opcode')},source_hash=m['native_build']['source_hash'])
            if common is not None and current!=common:raise ValueError('incomparable workload/configuration')
            common=current
            images.setdefault(family,set()).add(m['native_build']['binary_sha256'])
            structures[mode]=m['schedule_structure']
        samples=[x for d in data for t in d['trials']for x in t['samples_ms']];cohorts[mode]=samples
        rows.append({'mode':mode,'checks':len(samples),**summarize(samples),
                     'batch_min_p50_ms':min(d['summary']['p50_ms']for d in data),
                     'batch_max_p50_ms':max(d['summary']['p50_ms']for d in data)})
    if any(len(v)!=1 for v in images.values()):raise ValueError('grouping must use the same binary within each backend')
    for mode in ['static','ready']:
        identity=json.loads((args.folder/f'identity-{mode}.json').read_text())
        if identity['status']!='PASS' or len(identity['rows'])!=9:raise ValueError('incomplete identity matrix')
        if any(not r['passed']or not all(c['bitwise_identical']for c in r['comparisons'].values())for r in identity['rows']):raise ValueError('output identity failed')
    lines=['# Native O-projection activation reuse','',
           '| Backend / grouping | Checks | p50 ms | p95 ms | p99 ms | Batch p50 range ms |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['mode']} | {r['checks']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['p99_ms']:.3f} | {r['batch_min_p50_ms']:.3f}–{r['batch_max_p50_ms']:.3f} |")
    lines+=['','Five independently loaded batches per variant, five warmups and thirty checked samples each. A seeded Latin-square design balances order positions; ordered adjacent pairs are not fully balanced. Native batch-one BF16 Llama-3.2-1B-Instruct, context 1024, same-cache replay. Embedding, runtime, head and argmax are included; loading/prefill/cache restore/checking are excluded. The GPU has uncontrolled existing workloads, so these are cohort observations rather than isolated causal speedups.','']
    for family,i in [('static',1),('ready',3)]:
        delta=100*(rows[i+1]['p50_ms']/rows[i]['p50_ms']-1)
        lines.append(f'- {family}: observed pooled median difference {delta:+.2f}%; grouping uses the exact same native binary within this backend.')
    plain,group=structures['static'],structures['static-reuse']
    lines+=['',f"O-projection instructions fall from {plain['o_proj_tasks']} to {group['o_proj_tasks']} per decode; total tasks {plain['tasks']} to {group['tasks']}. Logical activation input loads fall from {plain['logical_o_proj_input_bytes']:,} to {group['logical_o_proj_input_bytes']:,} bytes. This is not measured LPDDR traffic. All native output tiles and arithmetic are preserved. The static grouping pair both retire attention through the selected instruction join; the baseline row retains the previous no-retirement setting. This separates grouping from the handoff change.",'',
            'Logits, final hidden state and newly appended K/V are bitwise identical in all nine context/seed pairs per backend (contexts 128/1024/8192, seeds 1/2/3). Both grouped backends also pass all three sanitizer tools. The earlier grouped static run without retirement failed racecheck and is excluded from this qualified cohort. These are frozen-input identity checks, not a claim of improved model quality.','']
    for family in ['static','ready']:
        q=json.loads((args.folder/f'{family}-reuse-temporal.json').read_text());failed=[r['step']for r in q['rows']if not r['passed']]
        if len(q['rows'])!=100 or q['status']!=('FAIL'if failed else'PASS'):raise ValueError('incomplete/inconsistent temporal qualification')
        lines.append(f"- {family} grouped accumulated-native-cache check: **{q['status']}**, failed steps {failed}, worst logit NRMSE {100*q['summary']['worst_logit_nrmse']:.4f}%; {q['summary']['greedy_matches']}/100 greedy matches. Numerical failures are not waived.")
    lines+=['','This is a host schedule/data-reuse specialization, not a precision change. It reduces instruction setup and changes task placement as well as activation loads; those effects are not isolated here. Final autoregressive/vLLM comparison remains separate. See `ACTIVATION_REUSE.md` for implementation, memory/progress contracts and reproduction.','',f'![Latency distributions]({args.output.with_suffix(".svg").name})','']
    args.output.write_text('\n'.join(lines))
    fields=['mode','checks','p50_ms','p95_ms','p99_ms','batch_min_p50_ms','batch_max_p50_ms']
    with args.output.with_suffix('.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,family in zip(axes,['static','ready']):
        for mode,label in [(family,'ungrouped'),(family+'-reuse','activation reuse')]:
            values=sorted(cohorts[mode]);ax.step(values,[(i+1)/len(values)for i in range(len(values))],where='post',label=label)
        ax.set(xlabel='CUDA-event decode replay (ms)',title=family);ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_ylabel('Empirical cumulative probability');fig.suptitle('O-projection activation reuse; shared Thor, context 1024');fig.tight_layout()
    svg=args.output.with_suffix('.svg');fig.savefig(svg);plt.close(fig);svg.write_text('\n'.join(line.rstrip()for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
