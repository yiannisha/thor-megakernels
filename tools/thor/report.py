"""Render raw-verified benchmark tables/plots; failures never become measurements."""
import argparse
import csv
import json
import math
from pathlib import Path

try:
    from .bench import summarize
except ImportError:
    from bench import summarize


def measured(data):
    if data.get('status') != 'PASS' or data.get('correctness', {}).get('passed') is not True:
        return False
    trials = data.get('trials', [])
    protocol = data.get('protocol', {})
    if not trials or len(trials) != protocol.get('batches'):
        return False
    samples = []
    for trial in trials:
        values = trial.get('samples_ms', [])
        checks = trial.get('correctness_checks', [])
        if len(values) != protocol.get('repeats') or len(checks) != len(values):
            return False
        if any(check.get('passed') is not True for check in checks):
            return False
        samples.extend(values)
    try:
        raw = summarize(samples)
        return all(math.isclose(data['summary'][key], value, rel_tol=1e-9, abs_tol=1e-9)
                   for key, value in raw.items())
    except (ValueError, KeyError, TypeError):
        return False


def records(paths):
    for path in paths:
        yield path, json.loads(path.read_text())


def clean(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def render(paths):
    lines = ['| Run | Engine / variant | Context | Status | Scope | Samples | p50 ms | p95 ms | p99 ms |',
             '|---|---|---|---|---|---|---|---|---|']
    for path, data in records(paths):
        valid = measured(data)
        summary = data.get('summary') if valid else None
        meta = data.get('metadata', {})
        metrics = [f'{summary[key]:.6f}' if summary else 'UNAVAILABLE'
                   for key in ('p50_ms', 'p95_ms', 'p99_ms')]
        scope = meta.get('measurement_scope', 'UNAVAILABLE')
        status = data['status'] if valid or data['status'] != 'PASS' else 'INVALID RAW DATA'
        samples = sum(len(t.get('samples_ms', [])) for t in data.get('trials', []))
        lines.append('| ' + ' | '.join(clean(x) for x in [data['run_id'], meta.get('variant', 'UNAVAILABLE'),
                     meta.get('context', 'UNAVAILABLE'), status, scope, samples, *metrics]) + ' |')
    lines.append('')
    lines.append('Same-cache decode replays are not autoregressive generation. Read each run\'s scope, accuracy gates, power profile and contention metadata before comparing. Tail estimates from small sample sets have limited precision.')
    return '\n'.join(lines) + '\n'


def write_csv(paths, output):
    fields = ['run_id', 'status', 'engine', 'variant', 'scope', 'precision', 'context', 'batch',
              'samples', 'warmups', 'batches', 'power_mode', 'concurrent_workload',
              'p50_ms', 'p95_ms', 'p99_ms', 'mean_ms', 'tokens_per_s']
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path, data in records(paths):
            meta = data.get('metadata', {})
            row = {'run_id': data['run_id'], 'status': data['status']}
            row.update({key: meta.get(key) for key in fields if key in meta})
            row['scope'] = meta.get('measurement_scope')
            row['samples'] = sum(len(t.get('samples_ms', [])) for t in data.get('trials', []))
            row['warmups'] = data.get('warmups_completed')
            row['batches'] = data.get('protocol', {}).get('batches')
            if measured(data):
                row.update({k: data['summary'].get(k) for k in ('p50_ms', 'p95_ms', 'p99_ms', 'mean_ms', 'tokens_per_s')})
            writer.writerow(row)


def plot(paths, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(figsize=(8, 4.5))
    count = 0
    for path, data in records(paths):
        if not measured(data):
            continue
        samples = sorted(x for trial in data['trials'] for x in trial['samples_ms'])
        meta = data['metadata']
        axes.step(samples, [(i+1)/len(samples) for i in range(len(samples))], where='post',
                  label=f"{meta.get('variant', meta.get('engine', 'unknown'))}; context {meta['context']}")
        count += 1
    if not count:
        plt.close(figure)
        raise ValueError('no verified measurements to plot')
    axes.set(xlabel='CUDA-event interval (ms)', ylabel='Empirical cumulative probability',
             title='Measured latency distributions (see run conditions)')
    axes.grid(alpha=.25)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output)
    svg = Path(output)
    if svg.suffix.lower() == '.svg':
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    report = render(args.inputs)
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end='')
    if args.csv:
        write_csv(args.inputs, args.csv)
    if args.plot:
        plot(args.inputs, args.plot)


if __name__ == '__main__':
    main()
