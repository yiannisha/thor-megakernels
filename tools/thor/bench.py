"""CUDA-event benchmark runner. Adapter factory returns run/reference/check callbacks.

Use --dry-run to inspect prerequisites without importing an adapter or launching.
An adapter's run/reset must cover exactly the metadata['measurement_scope']; check
returns numerical errors and passed. reset prepares identical KV/activation state
outside timing. reference must use matched state and precision.
"""
import argparse
import csv
import importlib
import json
import math
import statistics
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from .inspect_device import inventory
except ImportError:
    from inspect_device import inventory


def percentile(values, quantile):
    ordered = sorted(values)
    offset = (len(ordered) - 1) * quantile
    lo = math.floor(offset)
    hi = math.ceil(offset)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (offset - lo)


def summarize(samples):
    if not samples or any(not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError('timings must be nonempty, finite and nonnegative')
    return {'p50_ms': percentile(samples, .5), 'p95_ms': percentile(samples, .95),
            'p99_ms': percentile(samples, .99), 'mean_ms': statistics.mean(samples),
            'min_ms': min(samples), 'max_ms': max(samples)}


def benchmark(factory, warmups=5, repeats=30, batches=3, checkpoint=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch is required')
    adapter = factory()
    metadata = adapter.metadata
    state = {'metadata': metadata, 'correctness': None, 'trials': [], 'summary': None,
             'status': 'RUNNING', 'errors': [], 'warmups_completed': 0}
    try:
        for key in ('model', 'precision', 'context', 'batch', 'measurement_scope', 'variant',
                    'power_mode', 'concurrent_workload'):
            if key not in metadata:
                raise ValueError(f'adapter metadata missing {key}')
        if metadata['measurement_scope'] not in ('prefill', 'decode', 'generate', 'kernel', 'robotics_block'):
            raise ValueError('unknown measurement scope')
        if checkpoint: checkpoint(state)
        adapter.reset()
        reference = adapter.reference()
        torch.cuda.synchronize()
        adapter.reset()
        output = adapter.run()
        torch.cuda.synchronize()
        state['correctness'] = adapter.check(output, reference)
        if state['correctness'].get('passed') is not True:
            raise RuntimeError(f'numerical correctness failed: {state["correctness"]}')
        for _ in range(warmups):
            adapter.reset()
            adapter.run()
            state['warmups_completed'] += 1
        torch.cuda.synchronize()
        for batch in range(batches):
            trial = {'batch_index': batch, 'samples_ms': [], 'correctness_checks': []}
            state['trials'].append(trial)
            for _ in range(repeats):
                adapter.reset()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = adapter.run()
                end.record()
                end.synchronize()
                trial['samples_ms'].append(start.elapsed_time(end))
                check = adapter.check(output, reference)
                trial['correctness_checks'].append(check)
                if check.get('passed') is not True:
                    raise RuntimeError(f'correctness failed during trial: {check}')
            trial.update(summarize(trial['samples_ms']))
            if checkpoint: checkpoint(state)
        state['summary'] = summarize([x for trial in state['trials'] for x in trial['samples_ms']])
        if metadata['measurement_scope'] == 'decode' and metadata.get('tokens_per_run'):
            state['summary']['tokens_per_s'] = (1000 * metadata['tokens_per_run'] / state['summary']['mean_ms']
                                                if state['summary']['mean_ms'] > 0 else None)
        state['status'] = 'PASS'
    except Exception as error:
        state['status'] = 'FAIL'
        state['errors'].append(f'{type(error).__name__}: {error}')
        state['summary'] = None
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-id', help=argparse.SUPPRESS)
    parser.add_argument('--adapter', help='importable module:factory; no model download by this runner')
    parser.add_argument('--output', type=Path, default=Path('thor-benchmark.json'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--batches', type=int, default=3)
    args = parser.parse_args()
    if args.timeout < 1 or args.warmups < 0 or args.repeats < 1 or args.batches < 1:
        parser.error('warmups must be nonnegative; repeats and batches must be positive')
    args.output = args.output.resolve()
    if not args.worker and not args.dry_run:
        try:
            from .sweep import timed_command
        except ImportError:
            from sweep import timed_command
        run_id = str(uuid.uuid4())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        child = [sys.executable, '-m', 'tools.thor.bench', '--worker', '--run-id', run_id,
                 '--output', str(args.output), '--warmups', str(args.warmups), '--repeats', str(args.repeats),
                 '--batches', str(args.batches), '--timeout', str(args.timeout)]
        if args.adapter: child += ['--adapter', args.adapter]
        process = timed_command(child, args.timeout)
        result = json.loads(args.output.read_text()) if args.output.exists() else {}
        if result.get('run_id') != run_id:
            result = {'schema_version': 1, 'run_id': run_id, 'summary': None, 'errors': []}
        result['process'] = process
        if process['returncode'] != 0:
            result['status'] = 'FAIL'
            result['summary'] = None
            result.setdefault('errors', []).append('watchdog timeout' if process.get('timeout') else 'benchmark worker failed')
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'status': result.get('status'), 'output': str(args.output), 'errors': result.get('errors', [])}))
        return 0 if result.get('status') == 'PASS' else 1
    result = {'schema_version': 1, 'run_id': args.run_id or str(uuid.uuid4()),
              'timestamp': datetime.now(timezone.utc).isoformat(), 'inventory': inventory(),
              'protocol': vars(args) | {'output': str(args.output)}, 'status': 'DRY_RUN',
              'errors': [], 'summary': None}
    if not args.dry_run:
        result['status'] = 'RUNNING'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        def checkpoint(state):
            args.output.write_text(json.dumps(result | state, indent=2, allow_nan=False) + '\n')
        try:
            if not args.adapter or ':' not in args.adapter:
                raise ValueError('--adapter module:factory required')
            module, factory = args.adapter.split(':', 1)
            result.update(benchmark(getattr(importlib.import_module(module), factory),
                                    args.warmups, args.repeats, args.batches, checkpoint))
        except Exception as error:
            result['status'] = 'FAIL'
            result['errors'].append(f'{type(error).__name__}: {error}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['run_id', 'status', 'batch_index', 'sample_index', 'elapsed_ms'])
        writer.writeheader()
        for trial in result.get('trials', []):
            for index, elapsed in enumerate(trial['samples_ms']):
                writer.writerow({'run_id': result['run_id'], 'status': result['status'],
                                 'batch_index': trial['batch_index'], 'sample_index': index,
                                 'elapsed_ms': elapsed})
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'errors': result['errors']}))
    return 1 if result['status'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
