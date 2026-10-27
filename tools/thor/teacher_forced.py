"""Qualify successive native steps against PyTorch with reference past KV.

Both paths receive the same token, position, BF16 weights and reference past KV.
Native current-step K/V is compared before PyTorch overwrites it. This isolates
per-step errors; it does not establish free-running trajectory equivalence.
"""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-attention', choices=('flash', 'math'), default='flash')
    parser.add_argument('--native-past', action='store_true', help='propagate native KV separately from reference KV')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.context, args.steps, args.timeout) < 1:
        parser.error('context, steps and timeout must be positive')
    result = {'status': 'RUNNING', 'schema_version': 'thor.teacher_forced.v1',
              'timestamp': datetime.now(timezone.utc).isoformat(),
              'protocol': vars(args) | {'output': str(args.output)}, 'rows': [], 'errors': []}
    write(args.output, result)
    if not args.worker:
        from .sweep import timed_command
        process = timed_command([sys.executable, '-m', 'tools.thor.teacher_forced', '--worker',
                                 '--context', str(args.context), '--steps', str(args.steps),
                                 '--seed', str(args.seed), '--reference-attention', args.reference_attention,
                                 *(['--native-past'] if args.native_past else []), '--output', str(args.output.resolve())], args.timeout)
        result = json.loads(args.output.read_text())
        result['process'] = process
        if process['returncode'] != 0:
            result['status'] = 'FAIL'
            result['errors'].append('watchdog timeout' if process.get('timeout') else 'teacher-forced qualification failed')
        write(args.output, result)
        print(json.dumps({'status': result['status'], 'steps_completed': len(result['rows']), 'output': str(args.output)}))
        return 0 if result['status'] == 'PASS' else 1
    import torch
    from .model_bench import ModelAdapter
    from .model_check import errors, attention_context
    try:
        with torch.inference_mode():
            adapter = ModelAdapter('native', args.context, args.seed,
                                   cache_capacity=args.context + args.steps + 1, reference_attention=args.reference_attention)
            result['metadata'] = adapter.metadata | {'measurement_scope': 'teacher_forced_native_KV' if args.native_past else 'teacher_forced_reference_KV',
                                                     'same_cache_replay': False,
                                                     'timing_measured': False}
            adapter.reset()
            native_cache = [t.clone() for t in adapter.model.stacked_kv_cache] if args.native_past else None
            reference_cache = [t.clone() for t in adapter.model.stacked_kv_cache] if args.native_past else None
            for step in range(args.steps):
                adapter.context = args.context + step
                adapter.position.fill_(adapter.context)
                adapter.state.seq_len = adapter.context + 1
                if args.native_past:
                    for tensor, cache in zip(adapter.model.stacked_kv_cache, native_cache):
                        tensor.copy_(cache)
                logits = adapter.run().clone()
                hidden = adapter.last_hidden.clone()
                kv = [tensor[:, 0, adapter.context].clone() for tensor in adapter.model.stacked_kv_cache]
                if args.native_past:
                    for cache, current in zip(native_cache, kv):
                        cache[:, 0, adapter.context].copy_(current)
                    for tensor, cache in zip(adapter.model.stacked_kv_cache, reference_cache):
                        tensor.copy_(cache)
                adapter.engine = 'torch'
                with attention_context(args.reference_attention):
                    reference = adapter.run().clone()
                if args.native_past:
                    for cache, tensor in zip(reference_cache, adapter.model.stacked_kv_cache):
                        cache[:, 0, adapter.context].copy_(tensor[:, 0, adapter.context])
                adapter.engine = 'native'
                metrics = {'logits': errors(logits, reference, nrmse_limit=adapter.logit_limit),
                           'hidden': errors(hidden, adapter.last_hidden),
                           'k_cache': errors(kv[0], adapter.model.stacked_kv_cache[0][:, 0, adapter.context]),
                           'v_cache': errors(kv[1], adapter.model.stacked_kv_cache[1][:, 0, adapter.context])}
                row = {'step': step, 'position': adapter.context, 'input_token': adapter.case.token.item(),
                       'native_token': logits.argmax().item(), 'reference_token': reference.argmax().item(),
                       'passed': all(m['passed'] for m in metrics.values()), 'metrics': metrics}
                result['rows'].append(row)
                write(args.output, result)
                adapter.case.token.copy_(reference.argmax().reshape(1, 1))
            result['status'] = 'PASS' if all(row['passed'] for row in result['rows']) else 'FAIL'
            result['summary'] = {'failed_steps': [r['step'] for r in result['rows'] if not r['passed']],
                                 'greedy_matches': sum(r['native_token'] == r['reference_token'] for r in result['rows']),
                                 'worst_logit_nrmse': max(r['metrics']['logits']['normalized_rmse'] for r in result['rows'])}
    except Exception as error:
        result['status'] = 'FAIL'
        result['errors'].append(f'{type(error).__name__}: {error}')
        raise
    finally:
        write(args.output, result)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
