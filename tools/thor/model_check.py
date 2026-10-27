"""Offline Llama-1B decode correctness against the repository's PyTorch/Python VM.

Uses existing local weights only. The parent process watchdog covers loading,
prefill and native execution. No performance claims are made by this command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {'hidden_size': 2048, 'intermediate_size': 8192, 'head_dim': 64,
            'num_hidden_layers': 16, 'num_attention_heads': 32,
            'num_key_value_heads': 8, 'vocab_size': 128256}


def inspect_model(path):
    path = Path(path).resolve()
    if not (path / 'config.json').is_file():
        raise FileNotFoundError(f'local model config missing: {path / "config.json"}')
    config = json.loads((path / 'config.json').read_text())
    wrong = {key: config.get(key) for key, value in EXPECTED.items() if config.get(key) != value}
    if wrong:
        raise ValueError(f'local model does not match compiled Llama-1B shapes: {wrong}')
    weight_files = sorted(path.glob('*.safetensors'))
    if not weight_files:
        raise FileNotFoundError(f'no local safetensors weights in {path}; no download will be attempted')
    from safetensors import safe_open
    names = {}
    for file in weight_files:
        with safe_open(file, framework='pt', device='cpu') as tensors:
            for key in tensors.keys():
                names[key] = list(tensors.get_slice(key).get_shape())
    for key, shape in {'model.embed_tokens.weight': [128256, 2048],
                       'model.layers.0.self_attn.q_proj.weight': [2048, 2048],
                       'model.layers.15.mlp.down_proj.weight': [2048, 8192]}.items():
        if names.get(key) != shape:
            raise ValueError(f'weight shape mismatch/missing {key}: {names.get(key)}')
    commit_file = path / '.cache/huggingface/download/config.json.metadata'
    commit = commit_file.read_text().splitlines()[0] if commit_file.exists() else None
    return {'path': str(path), 'config': config, 'config_sha256': hashlib.sha256((path / 'config.json').read_bytes()).hexdigest(),
            'download_revision': commit, 'tensor_count': len(names),
            'weights': [{'file': p.name, 'bytes': p.stat().st_size} for p in weight_files]}


def errors(output, reference, atol=.25, rtol=.05, nrmse_limit=.02):
    import torch
    output, reference = output.float(), reference.float()
    finite = bool(torch.isfinite(output).all() and torch.isfinite(reference).all())
    if not finite:
        return {'passed': False, 'reason': 'nonfinite output or reference', 'max_absolute_error': None,
                'normalized_rmse': None, 'nrmse_limit': nrmse_limit}
    difference = (output - reference).abs()
    relative = difference / reference.abs().clamp_min(1e-6)
    nrmse = ((output-reference).square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-6)).item()
    return {'passed': bool(torch.allclose(output, reference, atol=atol, rtol=rtol)) and nrmse <= nrmse_limit,
            'max_absolute_error': difference.max().item(), 'mean_absolute_error': difference.mean().item(),
            'max_relative_error': relative.max().item(), 'mean_relative_error': relative.mean().item(),
            'normalized_rmse': nrmse, 'atol': atol, 'rtol': rtol, 'nrmse_limit': nrmse_limit}


def load_reference(args):
    """Load local weights and construct one frozen, matched-cache decode case."""
    import torch
    from types import SimpleNamespace
    from megakernels.llama import LlamaForCausalLM
    from megakernels.model_types import ExtraModelConfig, BatchState
    provenance = inspect_model(args.model)
    for item in provenance['weights']:
        with (args.model / item['file']).open('rb') as stream:
            item['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        capacity = ((max(args.context + 1, getattr(args, "cache_capacity", 0)) + 15) // 16) * 16
        model = LlamaForCausalLM.from_pretrained(str(args.model.resolve()), device='cuda:0', dtype=torch.bfloat16,
                    extra_config=ExtraModelConfig(interleave_rope=True, max_len_override=capacity, ordered_downproj=getattr(args, 'ordered_downproj', False))).eval()
        inputs = torch.randint(32, model.config.vocab_size, (1, args.context), device=model.device)
        positions = torch.arange(args.context, device=model.device).unsqueeze(0)
        prefill = model(BatchState(input_ids=inputs, position_ids=positions, seq_len=args.context))
        token = prefill.output_ids[:, -1:].clone()
        cached = [tensor.clone() for tensor in model.stacked_kv_cache]
        output = model(BatchState(input_ids=token, position_ids=torch.tensor([[args.context]], device=model.device),
                                  seq_len=args.context + 1))
        ref_hidden = output.hidden_states.flatten().clone()
        ref_logits = model.lm_head.lm_head(model.lm_head.input_norm(output.hidden_states)).flatten().clone()
        ref_cache = [tensor[:, 0, args.context].clone() for tensor in model.stacked_kv_cache]
    return SimpleNamespace(model=model, provenance=provenance, token=token, inputs=inputs,
                           cached=cached, ref_hidden=ref_hidden, ref_logits=ref_logits, ref_cache=ref_cache)


def check(args):
    import torch
    from megakernels.llama import LlamaForCausalLM
    from megakernels.model_types import ExtraModelConfig, BatchState
    from megakernels.demos.latency.scheduler import LatencyScheduleBuilder
    from megakernels.demos.latency.mk import interpret_with_mk
    from megakernels.scheduler import assign_to_sms, tensorize_instructions
    from tools.thor.sweep import source_hash
    with torch.inference_mode():
        case = load_reference(args)
        model, provenance, token, cached = case.model, case.provenance, case.token, case.cached
        ref_hidden, ref_logits, ref_cache = case.ref_hidden, case.ref_logits, case.ref_cache
        schedule = LatencyScheduleBuilder.build(model)
        globs = schedule.globs
        queues = assign_to_sms(args.scheduler, schedule=schedule)
        tensorize_instructions(globs, queues)
        sys.path.insert(0, str((ROOT / 'demos/low-latency-llama').resolve()))
        import mk_llama
        resources = {}
        def preflight(*call_args, **kwargs):
            resources.update(mk_llama.mk_llama(*call_args, **kwargs, dry_run=True))
        interpret_with_mk(globs, preflight)
        if resources.get('compensated_attention') != args.compensated_attention:
            raise RuntimeError('compensated attention build and requested variant disagree')
        if resources.get('match_torch_bf16_epilogues') != args.match_torch_bf16_epilogues:
            raise RuntimeError('BF16 epilogue build and requested variant disagree')
        if resources.get('fp32_downproj') != args.fp32_downproj:
            raise RuntimeError('FP32 downprojection build and requested variant disagree')
        if resources.get('ordered_downproj') != args.ordered_downproj:
            raise RuntimeError('ordered-downproj build flag and requested host schedule disagree')
        if resources.get('build_source_hash') != source_hash():
            raise RuntimeError('native binary is stale; rebuild before model testing')
        def reset():
            for tensor, before in zip(model.stacked_kv_cache, cached):
                tensor.copy_(before)
            embedded = model.model.embed_tokens(BatchState(input_ids=token)).hidden_states
            globs.hidden_states.copy_(embedded.flatten())
            globs.barriers.zero_()
            globs.pos_id = args.context
        results = {}
        if args.pyvm:
            from megakernels.demos.latency.python_vm import INSTRUCTION_TO_SOLVER
            from megakernels.python_vm import interpret_with_pyvm
            reset()
            interpret_with_pyvm(globs, schedule.get_linear_instructions(), INSTRUCTION_TO_SOLVER)
            torch.cuda.synchronize()
            results['python_vm'] = {'logits': errors(globs.logits, ref_logits, nrmse_limit=args.logit_nrmse_limit),
                                    'hidden': errors(globs.hidden_states, ref_hidden),
                                    'greedy_token': int(globs.logits.argmax())}
        reference_token = int(ref_logits.argmax())
        trials = []
        diagnostics = []
        for repeat in range(args.correctness_repeats):
            reset()
            interpret_with_mk(globs, mk_llama.mk_llama)
            torch.cuda.synchronize()
            trial = {'logits': errors(globs.logits, ref_logits, nrmse_limit=args.logit_nrmse_limit),
                     'hidden': errors(globs.hidden_states, ref_hidden),
                     'greedy_token': int(globs.logits.argmax()),
                     'k_cache': errors(model.stacked_kv_cache[0][:, 0, args.context], ref_cache[0]),
                     'v_cache': errors(model.stacked_kv_cache[1][:, 0, args.context], ref_cache[1])}
            # Isolate the output head from accumulated decoder error. These
            # diagnostics do not replace the declared end-to-end gates.
            own_hidden_logits = model.lm_head.lm_head(model.lm_head.input_norm(
                globs.hidden_states.view(1, 1, -1))).flatten()
            delta = globs.logits.float() - ref_logits.float()
            centered = delta - delta.mean()
            ref_logp = ref_logits.float().log_softmax(-1)
            native_logp = globs.logits.float().log_softmax(-1)
            diagnostics.append({'lm_head_on_identical_hidden': errors(globs.logits, own_hidden_logits,
                                                                      nrmse_limit=args.logit_nrmse_limit),
                                'logit_mean_bias': delta.mean().item(),
                                'centered_logit_nrmse': (centered.square().mean().sqrt() /
                                    ref_logits.float().square().mean().sqrt().clamp_min(1e-6)).item(),
                                'kl_reference_to_native': (ref_logp.exp() * (ref_logp-native_logp)).sum().item()})
            trial['greedy_token_matches'] = trial['greedy_token'] == reference_token
            trials.append(trial)
        native_passed = all(metric['passed'] for trial in trials for metric in trial.values() if isinstance(metric, dict))
        results['native'] = dict(trials[0], trials=trials, all_trials_passed=native_passed,
                                 worst_logits_nrmse=(max(t['logits']['normalized_rmse'] for t in trials)
                                     if all(t['logits']['normalized_rmse'] is not None for t in trials) else None))
        for result in results.values():
            result['greedy_token_matches'] = result['greedy_token'] == reference_token
        passed = native_passed and all(metric['passed'] for result in results.values() for metric in result.values() if isinstance(metric, dict))
        result = {'status': 'PASS' if passed else 'FAIL', 'model': provenance, 'context': args.context,
                  'seed': args.seed, 'scheduler': args.scheduler, 'precision': 'bf16',
                  'reference': 'repository PyTorch model, same interleaved weights and prefilled KV cache',
                  'input_token': int(token.item()), 'reference_greedy_token': reference_token,
                  'diagnostics': diagnostics, 'native_resources': resources, 'queue_shape': list(globs.instructions.shape), 'results': results,
                  'correctness_repeats': args.correctness_repeats,
                  'reference_top_logit_margin': float(ref_logits.topk(2).values.diff().abs().item()),
                  'gate': {'logit_nrmse_limit': args.logit_nrmse_limit, 'state_nrmse_limit': .02,
                           'revision': 'user-directed provisional BF16 logit gate; historical 2% failures retained'},
                  'tolerance_note': 'Logit NRMSE defaults to 3% following user direction; state checks stay at 2%. Max-error checks are unchanged. All repeated trials must pass.'}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'local/models/Llama-3.2-1B-Instruct')
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--scheduler', choices=('rr', 'zz', 'wave', 'dag'), default='rr')
    parser.add_argument('--correctness-repeats', type=int, default=3)
    parser.add_argument('--logit-nrmse-limit', type=float, default=.03)
    parser.add_argument('--match-torch-bf16-epilogues', action='store_true')
    parser.add_argument('--compensated-attention', action='store_true')
    parser.add_argument('--fp32-downproj', action='store_true')
    parser.add_argument('--ordered-downproj', action='store_true')
    parser.add_argument('--pyvm', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--output', type=Path, default=ROOT / 'local/model-check.json')
    args = parser.parse_args()
    if args.context < 1 or args.timeout < 1 or args.correctness_repeats < 1 or not 0 < args.logit_nrmse_limit < 1:
        parser.error('context/timeout must be positive')
    if args.dry_run:
        print(json.dumps(inspect_model(args.model), indent=2))
        return 0
    if not args.worker:
        from tools.thor.sweep import timed_command
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            child_output = Path(tmp) / 'result.json'
            command = [sys.executable, '-m', 'tools.thor.model_check', '--worker', '--model', str(args.model),
                       '--context', str(args.context), '--seed', str(args.seed), '--scheduler', args.scheduler,
                       '--output', str(child_output), '--correctness-repeats', str(args.correctness_repeats),
                       '--logit-nrmse-limit', str(args.logit_nrmse_limit)]
            if args.match_torch_bf16_epilogues: command.append('--match-torch-bf16-epilogues')
            if args.compensated_attention: command.append('--compensated-attention')
            if args.fp32_downproj: command.append('--fp32-downproj')
            if args.pyvm: command.append('--pyvm')
            if args.ordered_downproj: command.append('--ordered-downproj')
            process = timed_command(command, args.timeout)
            result = json.loads(child_output.read_text()) if child_output.exists() else {'status': 'FAIL'}
            result['process'] = process
            if process['returncode'] != 0: result['status'] = 'FAIL'
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            print(json.dumps({'status': result['status'], 'output': str(args.output), 'diagnostics': process['output'][-2000:]}))
            return 0 if result['status'] == 'PASS' else 1
    result = check(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'results': result['results']}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
