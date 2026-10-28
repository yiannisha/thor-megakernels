"""Aggregate qualified native scheduling batches without promoting a default."""
import argparse
import csv
import json
from pathlib import Path

from tools.thor.bench import summarize
from tools.thor.report import measured
from tools.experiments.calibrated_schedule import signature


def aggregate(folder):
    groups={};sig=None
    for path in sorted(folder.glob('*-batch*.json')):
        data=json.loads(path.read_text())
        if not measured(data):raise ValueError(f'unqualified batch: {path}')
        protocol=data['protocol']
        if any(protocol.get(k)!=v for k,v in {'warmups':5,'repeats':30,'batches':1}.items()):
            raise ValueError('requires five warmups and thirty samples in each independent batch')
        if data['metadata']['context']!=1024 or data['metadata']['config']['timing']:
            raise ValueError('report requires untimed context-1024 runtime builds')
        current=signature(data['metadata'])
        if sig is not None and current!=sig:raise ValueError('incomparable workload/source/configuration')
        sig=current
        name=path.name.split('-batch')[0]
        groups.setdefault(name,[]).append((path,data))
    if not groups:raise ValueError('no scheduling batches')
    result=[]
    for name,batches in groups.items():
        if len(batches)!=3:raise ValueError(f'{name} requires three independent batches')
        samples=[x for _,d in batches for t in d['trials'] for x in t['samples_ms']]
        hashes={d['metadata'].get('cost_model',{}).get('queue_sha256') for _,d in batches}
        if len(hashes)!=1:raise ValueError('queues changed between repeated batches')
        result.append({'variant':name,'samples':len(samples),**summarize(samples),
                       'batch_p50_min_ms':min(d['summary']['p50_ms'] for _,d in batches),
                       'batch_p50_max_ms':max(d['summary']['p50_ms'] for _,d in batches),
                       'queue_sha256':next(iter(hashes)), 'files':[str(p) for p,_ in batches],
                       'raw_samples_ms':samples})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); rows=aggregate(args.folder)
    lines=['# Measured static scheduling on Thor','',
           '| Variant | Checks / samples | p50 ms | p95 ms | p99 ms | Batch p50 range ms |',
           '|---|---:|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda r:r['p50_ms']):
        lines.append(f"| {row['variant']} | {row['samples']} | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} | {row['p99_ms']:.3f} | {row['batch_p50_min_ms']:.3f}–{row['batch_p50_max_ms']:.3f} |")
    lines+=['','All rows combine three independently loaded batches with five warmups and thirty checked samples per batch. Variant order is shuffled with seed 20261027. Native batch-one BF16 Llama-3.2-1B-Instruct, context 1024, same-cache replay; prefill/loading/cache restore and correctness checks are outside the CUDA-event interval. Embedding, runtime, head and argmax are included. The shared GPU is uncontrolled; no isolated or quality-matched speedup, energy or DRAM-traffic claim is made. Temporal numerical qualification remains open. These measurements do not select a default.','',
            'Measured-cost buckets come from inclusive maximum-consumer same-warp cycle spans. They contain waits and synchronization and are not pure compute costs. The critical policy prioritizes remaining modeled critical path. Locality only breaks equal estimated completion-time worker choices using producer affinity; activation buffers remain in global memory.','',
            '![Latency distributions]('+args.output.with_suffix('.svg').name+')','']
    identical={}
    for row in rows:
        if row['queue_sha256']:
            identical.setdefault(row['queue_sha256'],[]).append(row['variant'])
    for names in identical.values():
        if len(names)>1:
            lines+=['Serialized instruction queues are identical for '+', '.join(names)+'. Their latency differences reflect repeated-run variation, not different queue placement. The producer-affinity tie-break makes no placement change on this measured Llama DAG.','']
    args.output.write_text('\n'.join(lines))
    fields=['variant','samples','p50_ms','p95_ms','p99_ms','batch_p50_min_ms','batch_p50_max_ms','queue_sha256']
    with args.output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(figsize=(8,4.5))
    for row in rows:
        values=sorted(row['raw_samples_ms'])
        axes.step(values,[(i+1)/len(values) for i in range(len(values))],where='post',label=row['variant'])
    axes.set(xlabel='Decode replay CUDA-event interval (ms)',ylabel='Empirical cumulative probability',title='Static scheduling; shared Thor, context 1024')
    axes.grid(alpha=.25);axes.legend(fontsize=8);figure.tight_layout()
    svg=args.output.with_suffix('.svg');figure.savefig(svg);plt.close(figure)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':main()
