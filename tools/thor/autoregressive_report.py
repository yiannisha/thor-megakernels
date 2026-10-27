"""Report fixed-length autoregressive runs separately from one-token replays."""
import argparse
import json
from pathlib import Path
from .bench import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    lines = ['# Autoregressive decode measurements', '',
             'Batch/concurrency one, BF16, context 1024, 100 decode tokens after the first prefill token. '
             'Existing external load and automatic clocks are uncontrolled. Request-level p95/p99 describe mean TPOT across requests, not individual-token tails.', '',
             '| Engine | Timing protocol | Requests | Median TPOT (ms) | p95 (ms) | p99 (ms) |',
             '|---|---|---:|---:|---:|---:|']
    runs = []
    for path in args.inputs:
        data = json.loads(path.read_text())
        if data['schema_version'] != 'thor.autoregressive.v1':
            raise ValueError(f'incompatible timing schema: {path}')
        meta = data.get('metadata', {})
        engine = meta.get('engine', path.stem)
        if data['status'] != 'PASS':
            lines.append(f'| {engine} | FAIL | — | — | — | — |')
            continue
        requests = [r for trial in data['trials'] for r in trial['requests']]
        protocol = data['protocol']
        if len(requests) != protocol['repeats'] * protocol['batches']:
            raise ValueError('incomplete measured requests')
        steps = meta['decode_steps_after_prefill']
        if any(len(r['generated_tokens']) != steps + 1 for r in requests):
            raise ValueError('wrong generated length')
        stats = summarize([r['tpot_ms'] for r in requests])
        if stats != data['summary']['request_mean_tpot_ms']:
            raise ValueError('stored summary differs from raw data')
        lines.append(f"| {engine} | PASS | {len(requests)} | {stats['p50_ms']:.3f} | {stats['p95_ms']:.3f} | {stats['p99_ms']:.3f} |")
        runs.append((engine, data, requests))
    lines += ['', 'Native wall timing includes host dispatch, embedding, decoder, head, greedy selection and final token transfer; prefill and KV restoration are excluded. '
              'vLLM timing uses engine first/last-token timestamps and includes engine scheduling/sampling. TTFT/end-to-end latency is saved separately. '
              'These are application-level decode scopes with different runtime boundaries; they are not isolated kernel speedups.', '',
              'Native sequence numerical qualification remains open. A ten-step teacher-forced diagnostic exceeded the 3% raw logit NRMSE gate at one step (3.076%); see [raw checks](autoregressive-teacher-forced.json). Native first-step acceptance does not qualify the full trajectory. No quality-matched speedup is claimed.', '']
    if runs:
        first_meta = runs[0][1]['metadata']
        for engine, data, requests in runs:
            meta = data['metadata']
            if any(meta[k] != first_meta[k] for k in ['context', 'input_hash', 'decode_steps_after_prefill']):
                raise ValueError('workload mismatch')
            lines.append(f"- {engine}: raw data [{Path(data['protocol']['output']).name}]({Path(data['protocol']['output']).name}); "
                         f"{sum(r['generated_tokens'] == requests[0]['generated_tokens'] for r in requests)}/{len(requests)} requests reproduce its first trajectory.")
        if len(runs) == 2:
            a, b = runs[0][2][0]['generated_tokens'], runs[1][2][0]['generated_tokens']
            lines += ['', f'First measured trajectories agree at {sum(x == y for x, y in zip(a,b))}/{len(a)} token positions. '
                      'Timing PASS means protocol completion; full numerical equivalence is not implied. Native first-step numerical checks and its PyTorch trajectory agreement are retained in the raw report.']
    args.output.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
