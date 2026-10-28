"""Native/PyTorch autoregressive decode, sharing the vLLM prompt and fixed length.

Wall timing starts with the prefill token ready and ends with output IDs on CPU.
Loading, prefill, cache restoration and numerical qualification are excluded.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    import torch
    from .model_bench import ModelAdapter
    from .vllm_bench import ROOT, summary, write
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['native', 'torch'], default='native')
    parser.add_argument('--context', type=int, default=1024)
    parser.add_argument('--decode-steps', type=int, default=100)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-numerical-match', action='store_true', help='optionally make reported reference tolerances a timing gate')
    args = parser.parse_args()
    if min(args.context, args.decode_steps, args.repeats, args.batches) < 1 or args.warmups < 0:
        parser.error('invalid counts')
    result = {'schema_version': 'thor.autoregressive.v1', 'status': 'RUNNING',
              'protocol': vars(args) | {'output': str(args.output)}, 'summary': None, 'trials': [], 'errors': []}
    write(args.output, result)
    try:
        with torch.inference_mode():
            adapter = ModelAdapter(args.engine, args.context, args.seed,
                                   cache_capacity=args.context + args.decode_steps + 1)
            prompt = json.loads((ROOT / f'docs/thor/prompt-context{args.context}-seed{args.seed}.json').read_text())
            if adapter.case.inputs.flatten().tolist() != prompt['token_ids']:
                raise ValueError('input tokens differ from shared vLLM prompt')
            first_token = adapter.case.token.clone()
            result['metadata'] = adapter.metadata | {
                'measurement_scope': 'autoregressive_host_decode', 'same_cache_replay': False,
                'decode_steps_after_prefill': args.decode_steps, 'input_hash': prompt['input_hash'],
                'timing_source': 'host perf_counter with final CUDA synchronization and IDs on CPU',
                'timing_includes': ['embedding', 'runtime_dispatch', 'decoder', 'lm_head', 'argmax', 'final_ID_copy'],
                'timing_excludes': ['loading', 'prefill', 'KV_restore', 'correctness_checks'],
                'prefill_engine': 'repository PyTorch', 'first_token': first_token.item()}
            adapter.reset()
            output = adapter.run()
            qualification = adapter.check(output, adapter.reference())
            result['qualification'] = qualification
            write(args.output, result)
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError('nonfinite native output')
            result['numerical_policy'] = 'reference metrics reported; timing is not gated on tolerances unless explicitly requested'
            if args.require_numerical_match and not qualification['passed']:
                raise RuntimeError('first decode numerical qualification failed')

            def run():
                adapter.reset()
                adapter.case.token.copy_(first_token)
                generated = torch.empty((args.decode_steps + 1,), device='cuda', dtype=torch.long)
                generated[0] = first_token.item()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for step in range(args.decode_steps):
                    adapter.context = args.context + step
                    adapter.position.fill_(adapter.context)
                    adapter.state.seq_len = adapter.context + 1
                    logits = adapter.run()
                    next_token = adapter.last_token if adapter.engine == 'native' else logits.argmax()
                    generated[step + 1] = next_token
                    adapter.case.token.copy_(next_token.reshape(1, 1))
                ids = generated.cpu().tolist()
                elapsed = (time.perf_counter() - start) * 1000
                return {'decode_total_ms': elapsed, 'tpot_ms': elapsed / args.decode_steps,
                        'generated_tokens': ids}

            original_engine = adapter.engine
            adapter.engine = 'torch'
            reference_ids = run()['generated_tokens']
            adapter.engine = original_engine
            result['reference_generated_tokens'] = reference_ids
            result['qualification_scope'] = 'first decode logits/hidden/KV against PyTorch; trajectory token agreement reported separately'
            for _ in range(args.warmups):
                run()
            for batch in range(args.batches):
                trial = {'batch_index': batch, 'requests': []}
                result['trials'].append(trial)
                for _ in range(args.repeats):
                    trial['requests'].append(run())
                    write(args.output, result)
            requests = [r for t in result['trials'] for r in t['requests']]
            result['trajectory_agreement'] = {'all_requests_match_torch': all(r['generated_tokens'] == reference_ids for r in requests),
                                              'matching_tokens_per_request': [sum(a == b for a, b in zip(r['generated_tokens'], reference_ids)) for r in requests],
                                              'tokens_per_request': len(reference_ids)}
            result['summary'] = {'request_mean_tpot_ms': summary([r['tpot_ms'] for r in requests])}
            result['status'] = 'PASS'
    except Exception as error:
        result['status'] = 'FAIL'
        result['summary'] = None
        result['errors'].append(f'{type(error).__name__}: {error}')
        raise
    finally:
        write(args.output, result)
    print(json.dumps({'status': result['status'], 'summary': result['summary']}))


if __name__ == '__main__':
    main()
