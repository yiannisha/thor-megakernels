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


def attention_context(name):
    """Choose an explicit SDPA backend; math uses FP32 intermediate arithmetic."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if name not in ('flash', 'math'):
        raise ValueError('reference attention must be flash or math')
    if name == 'math' and torch.get_float32_matmul_precision() != 'highest':
        raise RuntimeError('math oracle requires highest FP32 matmul precision')
    if name == 'math' and torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed():
        raise RuntimeError('math oracle requires FP32 intermediates; reduced-precision math SDPA is enabled')
    return sdpa_kernel(SDPBackend.MATH if name == 'math' else SDPBackend.FLASH_ATTENTION)


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
                    extra_config=ExtraModelConfig(interleave_rope=True, max_len_override=capacity)).eval()
        inputs = torch.randint(32, model.config.vocab_size, (1, args.context), device=model.device)
        positions = torch.arange(args.context, device=model.device).unsqueeze(0)
        with attention_context('flash'):
            prefill = model(BatchState(input_ids=inputs, position_ids=positions, seq_len=args.context))
        token = prefill.output_ids[:, -1:].clone()
        cached = [tensor.clone() for tensor in model.stacked_kv_cache]
        reference_attention = getattr(args, 'reference_attention', 'flash')
        with attention_context(reference_attention):
            output = model(BatchState(input_ids=token, position_ids=torch.tensor([[args.context]], device=model.device),
                                      seq_len=args.context + 1))
        ref_hidden = output.hidden_states.flatten().clone()
        ref_logits = model.lm_head.lm_head(model.lm_head.input_norm(output.hidden_states)).flatten().clone()
        ref_cache = [tensor[:, 0, args.context].clone() for tensor in model.stacked_kv_cache]
    return SimpleNamespace(model=model, provenance=provenance, token=token, inputs=inputs,
                           cached=cached, ref_hidden=ref_hidden, ref_logits=ref_logits, ref_cache=ref_cache,
                           reference_attention=reference_attention, prefill_attention='flash')
