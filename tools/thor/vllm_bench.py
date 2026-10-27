"""Offline vLLM autoregressive baseline using a verified, shared token prompt.

TPOT uses vLLM's engine timestamps, excluding the first token/prefill. It includes
engine scheduling/sampling and must not be mixed with CUDA-event replay timings.
"""
import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]


def summary(values):
    from tools.thor.bench import summarize
    return summarize(values)


def worker(args, result):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['VLLM_NO_USAGE_STATS'] = '1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from tools.thor.model_check import inspect_model
    prompt = json.loads(args.prompt.read_text())
    tokens = prompt['token_ids']
    # Serialize int64 explicitly, matching the port's saved prompt hash.
    import struct
    digest = hashlib.sha256(struct.pack('<' + 'q' * len(tokens), *tokens)).hexdigest()
    if len(tokens) != prompt['context'] or digest != prompt['input_hash']:
        raise ValueError('prompt content/hash mismatch')
    provenance = inspect_model(args.model)
    for item in provenance['weights']:
        with (args.model / item['file']).open('rb') as stream:
            item['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    options = {'model': str(args.model.resolve()), 'tokenizer': str(args.model.resolve()),
               'dtype': 'bfloat16', 'quantization': None, 'kv_cache_dtype': 'auto',
               'tensor_parallel_size': 1, 'max_num_seqs': 1,
               'max_model_len': len(tokens) + args.decode_steps + 16,
               'max_num_batched_tokens': max(2048, len(tokens)),
               'enable_prefix_caching': False, 'disable_log_stats': False,
               'kv_cache_memory_bytes': 512 * 1024**2, 'cpu_offload_gb': 0,
               'enforce_eager': args.enforce_eager, 'seed': args.seed,
               'kernel_config': {'enable_flashinfer_autotune': not args.disable_flashinfer_autotune},
               'compilation_config': {'cudagraph_capture_sizes': [1], 'max_cudagraph_capture_size': 1}}
    result['metadata'] = {'engine': 'vllm', 'version': vllm.__version__, 'torch': torch.__version__,
                          'cuda': torch.version.cuda, 'model': provenance, 'precision': 'bf16',
                          'kv_precision': 'auto from BF16 model', 'batch': 1, 'max_concurrency': 1,
                          'context': len(tokens), 'input_hash': digest, 'seed': args.seed,
                          'decode_steps_after_prefill': args.decode_steps, 'options': options,
                          'measurement_scope': 'autoregressive_engine_decode',
                          'timing_source': 'vLLM RequestStateStats engine monotonic timestamps',
                          'prefix_caching': False, 'speculative_decode': False,
                          'concurrent_workload': 'uncontrolled-existing',
                          'not_comparable_to': 'same-cache CUDA-event replay without engine scheduling'}
    write(args.output, result)
    startup = time.perf_counter()
    llm = LLM(**options)
    result['startup_seconds'] = time.perf_counter() - startup
    config = llm.llm_engine.vllm_config
    result['effective_config'] = str(config)
    if config.model_config.dtype != torch.bfloat16 or config.model_config.quantization is not None:
        raise ValueError('vLLM did not select the requested unquantized BF16 model')
    if config.cache_config.enable_prefix_caching or config.scheduler_config.max_num_seqs != 1:
        raise ValueError('vLLM cache/concurrency configuration differs from protocol')
    params = SamplingParams(temperature=0, max_tokens=args.decode_steps+1,
                            min_tokens=args.decode_steps+1, ignore_eos=True, detokenize=False)
    def run():
        started = time.perf_counter()
        outputs = llm.generate([{'prompt_token_ids': tokens}], params, use_tqdm=False)
        wall_ms = (time.perf_counter() - started) * 1000
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError('expected exactly one request and one completion')
        output = outputs[0]
        generated = list(output.outputs[0].token_ids)
        if len(generated) != args.decode_steps+1:
            raise RuntimeError('unexpected output length; EOS must be ignored')
        metrics = output.metrics
        if metrics is None:
            raise RuntimeError('vLLM request metrics unavailable; cannot infer decode timing from end-to-end latency')
        first, last = metrics.first_token_ts, metrics.last_token_ts
        if not (math.isfinite(first) and math.isfinite(last) and 0 < first < last):
            raise RuntimeError(f'invalid first/last-token timing: {first}, {last}')
        if metrics.is_corrupted:
            raise RuntimeError('vLLM marked the request as corrupted')
        if output.num_cached_tokens:
            raise RuntimeError('prefix-cache reuse occurred despite disabled caching')
        decode_ms = (last-first) * 1000
        return {'e2e_wall_ms': wall_ms, 'ttft_ms': metrics.first_token_latency * 1000,
                'decode_total_ms': decode_ms, 'tpot_ms': decode_ms / args.decode_steps,
                'generated_tokens': generated,
                'request_metrics': dataclasses.asdict(metrics) if dataclasses.is_dataclass(metrics) else vars(metrics)}
    result['warmups_completed'] = 0
    for _ in range(args.warmups):
        run()
        result['warmups_completed'] += 1
    for batch in range(args.batches):
        trial = {'batch_index': batch, 'requests': []}
        result['trials'].append(trial)
        for _ in range(args.repeats):
            trial['requests'].append(run())
            write(args.output, result)
        trial['tpot_summary_ms'] = summary([r['tpot_ms'] for r in trial['requests']])
    requests = [r for t in result['trials'] for r in t['requests']]
    result['summary'] = {'request_mean_tpot_ms': summary([r['tpot_ms'] for r in requests]),
                         'request_ttft_ms': summary([r['ttft_ms'] for r in requests]),
                         'request_e2e_ms': summary([r['e2e_wall_ms'] for r in requests])}
    result['status'] = 'PASS'
    write(args.output, result)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'local/models/Llama-3.2-1B-Instruct')
    parser.add_argument('--prompt', type=Path, default=ROOT / 'docs/thor/prompt-context1024-seed1.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'local/vllm-benchmark.json')
    parser.add_argument('--decode-steps', type=int, default=100)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--enforce-eager', action='store_true', help='diagnostic only; default uses optimized graph path')
    parser.add_argument('--disable-flashinfer-autotune', action='store_true', help='disable autotuning while retaining graph execution')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.decode_steps, args.repeats, args.batches, args.timeout) < 1 or args.warmups < 0:
        parser.error('invalid benchmark counts or timeout')
    args.output = args.output.resolve()
    if not args.worker:
        from tools.thor.sweep import timed_command
        command = [sys.executable, '-m', 'tools.thor.vllm_bench', '--worker', '--model', str(args.model.resolve()),
                   '--prompt', str(args.prompt.resolve()), '--output', str(args.output),
                   '--decode-steps', str(args.decode_steps), '--warmups', str(args.warmups),
                   '--repeats', str(args.repeats), '--batches', str(args.batches), '--seed', str(args.seed)]
        if args.enforce_eager: command.append('--enforce-eager')
        if args.disable_flashinfer_autotune: command.append('--disable-flashinfer-autotune')
        result = {'schema_version': 'thor.autoregressive.v1', 'status': 'RUNNING', 'summary': None,
                  'run_id': str(uuid.uuid4()), 'errors': [], 'trials': []}
        write(args.output, result)
        process = timed_command(command, args.timeout)
        current = json.loads(args.output.read_text())
        current['process'] = process
        if process['returncode'] != 0:
            current['status'] = 'FAIL'
            current['summary'] = None
            current.setdefault('errors', []).append('watchdog timeout' if process.get('timeout') else 'vLLM worker failed')
        write(args.output, current)
        print(json.dumps({'status': current['status'], 'output': str(args.output), 'errors': current.get('errors')}))
        return 0 if current['status'] == 'PASS' else 1
    from tools.thor.inspect_device import inventory
    result = {'schema_version': 'thor.autoregressive.v1', 'status': 'RUNNING', 'summary': None,
              'timestamp': datetime.now(timezone.utc).isoformat(), 'run_id': str(uuid.uuid4()),
              'inventory': inventory(), 'errors': [], 'trials': [],
              'protocol': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    write(args.output, result)
    try:
        worker(args, result)
    except Exception as error:
        result['status'] = 'FAIL'
        result['summary'] = None
        result['errors'].append(f'{type(error).__name__}: {error}')
        write(args.output, result)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
