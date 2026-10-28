"""Report fixed-length autoregressive runs separately from one-token replays."""
import argparse
import json
import math
from pathlib import Path
from .bench import summarize


def qualification_note(qualification, native, path):
    """Bind per-step errors to the measured binary, inputs and trajectory."""
    qmeta, meta = qualification['metadata'], native['metadata']
    for key in ('context', 'input_hash', 'config', 'prefill_attention'):
        if key not in meta or qmeta.get(key) != meta[key]:
            raise ValueError(f'qualification {key} differs from measured run')
    weights = lambda m: sorted((w['file'], w['sha256']) for w in m['model_provenance']['weights'])
    if weights(qmeta) != weights(meta):
        raise ValueError('qualification weights differ from measured run')
    digest = meta.get('native_build', {}).get('binary_sha256')
    if not digest or digest != qmeta.get('native_build', {}).get('binary_sha256'):
        raise ValueError('qualification binary differs from measured run')
    rows = qualification['rows']
    tokens = native['trials'][0]['requests'][0]['generated_tokens']
    if len(rows) != meta['decode_steps_after_prefill'] or not qualification['protocol']['native_past']:
        raise ValueError('qualification must cover the full native KV trajectory')
    for i, row in enumerate(rows):
        if row['input_token'] != tokens[i] or row['native_token'] != tokens[i+1]:
            raise ValueError('qualification does not follow the measured native trajectory')
        for metric in row['metrics'].values():
            value = metric.get('normalized_rmse')
            if metric['passed'] and (value is None or not math.isfinite(value) or value > metric['nrmse_limit']):
                raise ValueError('passing metric violates its recorded numerical gate')
        if row['passed'] != all(m['passed'] for m in row['metrics'].values()):
            raise ValueError('inconsistent per-step qualification status')
    failed = sum(not r['passed'] for r in rows)
    if qualification['status'] != ('FAIL' if failed else 'PASS'):
        raise ValueError('qualification status differs from its step records')
    logits = max(r['metrics']['logits']['normalized_rmse'] for r in rows)
    hidden = max(r['metrics']['hidden']['normalized_rmse'] for r in rows)
    return (f"Native per-step numerical qualification against {qmeta['reference_attention']} attention: "
            f"**{qualification['status']}**, {failed}/{len(rows)} failed steps; "
            f"worst logit NRMSE {100*logits:.3f}%, hidden NRMSE {100*hidden:.3f}%. "
            f"The tested binary, inputs and native KV trajectory match this timing run. "
            f"See [raw qualification]({path.name}). No quality-matched speedup is claimed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--qualification', type=Path)
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
              'Timing protocol completion alone does not establish numerical equivalence or a quality-matched speedup.', '']
    if args.qualification:
        native = [data for engine, data, _ in runs if engine == 'native']
        if len(native) != 1:
            raise ValueError('qualification needs exactly one native timing run')
        lines += [qualification_note(json.loads(args.qualification.read_text()), native[0], args.qualification), '']
    if runs:
        first_meta = runs[0][1]['metadata']
        for engine, data, requests in runs:
            meta = data['metadata']
            if any(meta[k] != first_meta[k] for k in ['context', 'input_hash', 'decode_steps_after_prefill']):
                raise ValueError('workload mismatch')
            lines.append(f"- {engine}: raw data [{Path(data['protocol']['output']).name}]({Path(data['protocol']['output']).name}); "
                         f"{sum(r['generated_tokens'] == requests[0]['generated_tokens'] for r in requests)}/{len(requests)} requests reproduce its first trajectory.")
            if engine == 'native' and 'trajectory_agreement' in data:
                agreement = data['trajectory_agreement']
                exact = sum(n == agreement['tokens_per_request'] for n in agreement['matching_tokens_per_request'])
                lines.append(f"- Native/PyTorch greedy sequences match at all positions in {exact}/{len(requests)} requests.")
        if len(runs) == 2:
            a, b = runs[0][2][0]['generated_tokens'], runs[1][2][0]['generated_tokens']
            lines += ['', f'First measured trajectories agree at {sum(x == y for x, y in zip(a,b))}/{len(a)} token positions. '
                      'Timing PASS means protocol completion; full numerical equivalence is not implied. Native first-step numerical checks and its PyTorch trajectory agreement are retained in the raw report.']
    args.output.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
