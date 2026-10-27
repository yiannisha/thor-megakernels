"""Model-free numerical checks of native Llama matvec opcodes using seeded BF16 tensors.

No pretrained weights or network access. This verifies selected opcode outputs,
not trained-model decode parity. Run under an external timeout (see --timeout).
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def reset_isolated_barriers(barriers, op, fp32_downproj):
    """Precomplete external dependencies; clear only counters produced here."""
    barriers.fill_(512)
    if op == 'down' and fp32_downproj:
        barriers[:, 7:10].zero_()
        barriers[:, 5].zero_()


def fixture(op='qkv', iterations=4, seed=1, context=128, partitions=3, partial_index=1,
            branch='q', rotary=False, empty_first=False, negative_scores=False):
    import torch
    from megakernels.demos.latency.instructions import Globals
    from megakernels.instructions import NoOp
    from megakernels.scheduler import tensorize_instructions
    from tools.thor.config import Config
    from megakernels.demos.latency.instructions import (
        LayerNorm_QKV_MatVecRopeAppend, O_ProjResidual, DownProjResidual,
        LayerNormDoubleMatVecSiLU, RMS_LM_Head, PartialAttention, AttentionReduction)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA torch required')
    config = Config(**json.loads(os.environ.get('THOR_CONFIG', '{}')))
    if config != Config(name=config.name, match_torch_bf16_rms=config.match_torch_bf16_rms, ordered_downproj=config.ordered_downproj, register_mode=config.register_mode, fp32_downproj=config.fp32_downproj, match_torch_bf16_epilogues=config.match_torch_bf16_epilogues, weight_stages=config.weight_stages, page_count=config.page_count, compensated_attention=config.compensated_attention):
        raise ValueError('synthetic fixture currently requires the verified default build (20 workers, 16 warps, static cap 96)')
    if op not in ('qkv', 'o_proj', 'down', 'upgate', 'lm_head', 'noop', 'attention', 'partial', 'reduce') or not 0 <= iterations <= 8:
        raise ValueError('unsupported opcode or iterations outside 0..8')
    if negative_scores and op not in ('attention', 'partial'):
        raise ValueError('negative-score stress case requires attention/partial')
    if iterations == 0 and op not in ('qkv', 'o_proj', 'down', 'upgate', 'lm_head'):
        raise ValueError('zero-iteration fixture is defined for matvec opcodes only')
    if context < 1 or not 1 <= partitions <= 20 or not 0 <= partial_index < partitions:
        raise ValueError('invalid context/attention partition parameters')
    torch.manual_seed(seed)
    device, dtype = torch.device('cuda:0'), torch.bfloat16
    def random(shape):
        return torch.randn(shape, device=device, dtype=dtype) / 32
    def zeros(shape, dtype=dtype):
        return torch.zeros(shape, device=device, dtype=dtype)
    cache_length = ((context + 15) // 16) * 16
    angle = torch.randn((cache_length, 32), device=device).repeat_interleave(2, dim=-1) if rotary else torch.zeros((cache_length, 64), device=device)
    start_block = {'q': 0, 'k': 128, 'v': 160}[branch]
    globs = Globals(
        qkv_proj_weights=random((1, 3072, 2048)), attn_ln_weights=torch.ones((1, 2048), device=device, dtype=dtype),
        o_proj_weights=random((1, 2048, 2048)), mlp_ln_weights=torch.ones((1, 2048), device=device, dtype=dtype),
        up_proj_weights=random((1, 8192, 2048)), gate_proj_weights=random((1, 8192, 2048)),
        down_proj_weights=random((1, 2048, 8192)),
        lm_head_norm_weights=torch.ones((1, 2048), device=device, dtype=dtype),
        lm_head_weights=random((128, 2048)),
        k_cache=torch.randn((1, 1, cache_length, 8, 64), device=device, dtype=dtype),
        v_cache=torch.randn((1, 1, cache_length, 8, 64), device=device, dtype=dtype),
        rope_cos=angle.cos(), rope_sin=angle.sin(),
        num_hidden_layers=16, num_attention_heads=32, num_kv_heads=8, head_dim=64, hidden_size=2048,
        intermediate_size=8192, vocab_size=128, attn_scale=1/8, rms_norm_eps=1e-5, device=device,
        hidden_states=torch.randn(2048, device=device, dtype=dtype), barriers=zeros((16, 10, 48), torch.int32),
        pos_id=context-1, post_ln_rope_q=torch.randn(2048, device=device, dtype=dtype), attn_out=random(2048),
        attn_lse_intermediates=zeros((32, 32), torch.float32),
        attn_out_intermediates=zeros((32, 32, 64), torch.float32), silu_out=random(8192), logits=zeros(128),
        down_partials=zeros((4, 2048), torch.float32),
        skip_attn_reduction=op != 'partial', up_gate_proj_block_size=16, down_proj_block_size=16,
        o_proj_block_size=16, lm_head_block_size=16, matvec_reduction_size=2048,
        qkv_block_size=16, attn_kv_block_size=16, attn_reduction_size=4)
    if negative_scores:
        globs.post_ln_rope_q.fill_(8)
        globs.k_cache.fill_(-8)
    instructions = {'qkv': LayerNorm_QKV_MatVecRopeAppend(0, start_block, start_block + iterations),
                    'o_proj': O_ProjResidual(0, 0, iterations, 0),
                    'down': DownProjResidual(0, 0, iterations, 0),
                    'upgate': LayerNormDoubleMatVecSiLU(0, list(range(iterations))),
                    'lm_head': RMS_LM_Head(0, iterations), 'noop': NoOp(),
                    'attention': PartialAttention(0, 0, 1, 0),
                    'partial': PartialAttention(0, 0, partitions, partial_index),
                    'reduce': AttentionReduction(0, 0, partitions, True, list(range(partitions)))}
    if op == 'reduce':
        globs.attn_out_intermediates.normal_()
        globs.attn_lse_intermediates.normal_()
        if empty_first:
            globs.attn_lse_intermediates[:, 0] = -float('inf')
            globs.attn_out_intermediates[:, 0].zero_()
    queues = [[instructions[op]]] + [[NoOp()] for _ in range(config.workers - 1)]
    if op == 'down' and config.fp32_downproj:
        queues = [[DownProjResidual(0, 0, iterations, col)] for col in range(4)] + [[NoOp()] for _ in range(config.workers - 4)]
    if iterations == 0:
        queues[0] = [instructions[op], NoOp(), instructions[op], NoOp(), instructions[op]]
    tensorize_instructions(globs, queues)
    return globs, config


class SyntheticAdapter:
    def __init__(self, op='qkv', iterations=4, seed=1, **options):
        import torch
        from megakernels.demos.latency.mk import interpret_with_mk
        self.globs, self.config = fixture(op, iterations, seed, **options)
        binary = os.environ.get('THOR_BINARY_DIR', str(ROOT / 'demos/low-latency-llama'))
        sys.path.insert(0, binary)
        import mk_llama
        self.native = mk_llama.mk_llama
        self.interpret = interpret_with_mk
        self.op, self.iterations, self.options = op, iterations, options
        self.atol = .004 if op in ('attention', 'partial', 'reduce') else .02
        self.original = {name: getattr(self.globs, name).clone() for name in
                         ('hidden_states', 'attn_out', 'silu_out', 'post_ln_rope_q', 'k_cache', 'v_cache')}
        self.metadata = {'model': f'synthetic-llama1b-{op}', 'precision': 'bf16', 'context': self.globs.pos_id + 1,
                         'batch': 1, 'measurement_scope': 'kernel', 'variant': self.config.name,
                         'power_mode': 'observed separately in inventory', 'concurrent_workload': 'uncontrolled-existing',
                         'seed': seed, 'iterations': iterations, 'pretrained_weights': False,
                         'reference': 'FP32 accumulation on the same BF16 inputs/weights; BF16 output',
                         'tolerance': {'atol': self.atol, 'rtol': .03}, 'case_options': options,
                         'timing_scope_note': 'CUDA event interval includes Python binding enqueue gaps and TMA descriptor setup'}
        self.reset()
        def preflight(*args, **kwargs):
            self.metadata['native_resources'] = self.native(*args, **kwargs, dry_run=True)
        self.interpret(self.globs, preflight)
        observed = self.metadata['native_resources']
        from tools.thor.sweep import source_hash
        if observed.get('build_source_hash') != source_hash():
            raise RuntimeError('native binary is stale or lacks a source fingerprint; rebuild before testing')
        expected = {'consumer_warps': self.config.consumer_warps,
                    'active_workers': self.config.workers, 'backoff_ns': self.config.backoff_ns,
                    'instruction_barrier_parties_per_warp': self.config.instruction_parties,
                    'weight_pipeline_stages': self.config.weight_stages, 'page_count': self.config.page_count,
                    'page_size': self.config.page_size,
                    'timing_enabled': self.config.timing,
                    'match_torch_bf16_rms': self.config.match_torch_bf16_rms,
                    'ordered_downproj': self.config.ordered_downproj, 'fp32_downproj': self.config.fp32_downproj, 'match_torch_bf16_epilogues': self.config.match_torch_bf16_epilogues, 'register_mode': self.config.register_mode, 'compensated_attention': self.config.compensated_attention}
        if any(observed.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f'loaded native binary does not match fixture configuration: {observed}')

    def reset(self):
        g = self.globs
        for name, tensor in self.original.items():
            getattr(g, name).copy_(tensor)
        g.logits.zero_()
        # All dependencies of this isolated operation are precompleted, not produced
        # by another CTA. Cross-CTA visibility is outside this test's claim.
        reset_isolated_barriers(g.barriers, self.op, self.config.fp32_downproj)
        if self.op == "down" and self.config.fp32_downproj:
            g.down_partials.fill_(float("nan"))

    def reference(self):
        import torch
        import torch.nn.functional as F
        g, count = self.globs, self.iterations * 16
        hidden = self.original['hidden_states'].float()
        normalized = hidden * torch.rsqrt(hidden.square().mean() + g.rms_norm_eps)
        if self.config.match_torch_bf16_rms:
            normalized = normalized.to(torch.bfloat16).float()
        if self.op == 'qkv':
            start = {'q': 0, 'k': 2048, 'v': 2560}[self.options.get('branch', 'q')]
            result = g.qkv_proj_weights[0, start:start+count].float() @ normalized
            if self.config.match_torch_bf16_epilogues:
                result = result.to(torch.bfloat16).float()
            if self.options.get('branch', 'q') != 'v':
                cos = g.rope_cos[g.pos_id].repeat((count + 63) // 64)[:count]
                sin = g.rope_sin[g.pos_id].repeat((count + 63) // 64)[:count]
                pairs = result.reshape(-1, 2)
                rotated = torch.stack((-pairs[:, 1], pairs[:, 0]), dim=-1).flatten()
                result = result * cos + rotated * sin
        elif self.op == 'lm_head':
            result = g.lm_head_weights[:count].float() @ normalized
        elif self.op == 'o_proj':
            result = hidden[:count] + g.o_proj_weights[0, :count].float() @ self.original['attn_out'].float()
        elif self.op == 'down':
            if self.config.fp32_downproj:
                projected = (g.down_proj_weights[0, :count].float() @ self.original['silu_out'].float()).to(torch.bfloat16).float()
                result = hidden[:count] + projected
            else:
                result = hidden[:count] + g.down_proj_weights[0, :count, :2048].float() @ self.original['silu_out'][:2048].float()
        elif self.op == 'upgate':
            up = g.up_proj_weights[0, :count].float() @ normalized
            gate = g.gate_proj_weights[0, :count].float() @ normalized
            if self.config.match_torch_bf16_epilogues:
                up, gate = up.to(torch.bfloat16).float(), gate.to(torch.bfloat16).float()
                result = up * F.silu(gate).to(torch.bfloat16).float()
            else:
                result = up * F.silu(gate)
        elif self.op in ('attention', 'partial'):
            import math
            length = g.pos_id + 1
            if self.op == 'attention':
                start, end = 0, length
            else:
                blocks = (length + 15) // 16
                chunk = (blocks + self.options.get('partitions', 3) - 1) // self.options.get('partitions', 3)
                start = self.options.get('partial_index', 1) * chunk * 16
                end = min(start + chunk * 16, length)
            q = self.original['post_ln_rope_q'].view(32, 64)[:4].float()
            k = g.k_cache[0, 0, start:end, 0].float()
            v = g.v_cache[0, 0, start:end, 0].float()
            if start >= end:
                out = torch.zeros((4, 64), device=g.device)
                lse = torch.full((4,), -float('inf'), device=g.device)
            else:
                scores = (q @ k.T) * g.attn_scale
                out = scores.softmax(-1) @ v
                lse = scores.logsumexp(-1) / math.log(2)
            if self.op == 'partial':
                return torch.cat((out.flatten(), lse))
            result = out.flatten()
        elif self.op == 'reduce':
            parts = self.options.get('partitions', 3)
            lse = g.attn_lse_intermediates[:4, :parts]
            scale = torch.softmax(lse * 0.6931471805599453, dim=-1)
            result = (g.attn_out_intermediates[:4, :parts] * scale[..., None]).sum(1).flatten()
        else:
            result = hidden
        return result.to(torch.bfloat16).clone()

    def run(self):
        self.interpret(self.globs, self.native)
        count = self.iterations * 16
        if self.op == 'qkv' and self.options.get('branch', 'q') in ('k', 'v'):
            cache = self.globs.k_cache if self.options['branch'] == 'k' else self.globs.v_cache
            return cache[0, 0, self.globs.pos_id].flatten()[:count]
        if self.op in ('attention', 'reduce'):
            return self.globs.attn_out[:256]
        if self.op == 'partial':
            import torch
            index = self.options.get('partial_index', 1)
            return torch.cat((self.globs.attn_out_intermediates[:4, index].flatten(),
                              self.globs.attn_lse_intermediates[:4, index]))
        output = {'qkv': self.globs.post_ln_rope_q, 'lm_head': self.globs.logits,
                  'upgate': self.globs.silu_out}.get(self.op, self.globs.hidden_states)
        return output if self.op == 'noop' else output[:count]

    def check(self, output, reference):
        import torch
        finite = torch.isfinite(output) & torch.isfinite(reference)
        invalid = (~finite) & (output != reference)
        difference = (output[finite].float() - reference[finite].float()).abs()
        relative = difference / reference[finite].float().abs().clamp_min(1e-6)
        bad = bool(invalid.any())
        result = {'passed': not bad and torch.allclose(output, reference, atol=self.atol, rtol=.03),
                'nonfinite_mismatches': int(invalid.count_nonzero()),
                'max_absolute_error': None if bad else (difference.max().item() if difference.numel() else 0),
                'mean_absolute_error': None if bad else (difference.mean().item() if difference.numel() else 0),
                'max_relative_error': None if bad else (relative.max().item() if relative.numel() else 0),
                'mean_relative_error': None if bad else (relative.mean().item() if relative.numel() else 0)}
        if self.iterations == 0:
            result['zero_iteration_state_unchanged'] = all(torch.equal(getattr(self.globs, name), value)
                                                          for name, value in self.original.items())
            result['passed'] = bool(result['passed'] and result['zero_iteration_state_unchanged']
                                    and bool((self.globs.logits == 0).all()))
        if self.op == 'down' and self.config.fp32_downproj:
            counts = self.globs.barriers[0, 7:10].flatten()
            complete = self.globs.barriers[0, 5, 0].item()
            result['finalization_counts'] = counts[:self.iterations].tolist()
            result['published_outputs'] = complete
            result['passed'] = bool(result['passed'] and complete == 4 * self.iterations and
                                    bool((counts[:self.iterations] == 4).all()) and
                                    bool((counts[self.iterations:] == 0).all()))
        return result


def make_adapter():
    return SyntheticAdapter(os.environ.get('THOR_SYNTHETIC_OP', 'qkv'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--op', choices=('noop', 'qkv', 'o_proj', 'down', 'upgate', 'lm_head', 'attention', 'partial', 'reduce'), default='qkv')
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--partitions', type=int, default=3)
    parser.add_argument('--partial-index', type=int, default=1)
    parser.add_argument('--branch', choices=('q', 'k', 'v'), default='q')
    parser.add_argument('--rotary', action='store_true')
    parser.add_argument('--negative-scores', action='store_true')
    parser.add_argument('--empty-first', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path, default=ROOT / 'local/synthetic-result.json')
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        try:
            from .sweep import timed_command
        except ImportError:
            from sweep import timed_command
        options = ['--context', str(args.context), '--partitions', str(args.partitions),
                   '--partial-index', str(args.partial_index), '--branch', args.branch]
        if args.rotary: options.append('--rotary')
        if args.empty_first: options.append('--empty-first')
        if args.negative_scores: options.append('--negative-scores')
        result = timed_command([sys.executable, '-m', 'tools.thor.synthetic', '--worker', '--op', args.op,
                                '--iterations', str(args.iterations), '--seed', str(args.seed),
                                '--output', str(args.output), *options], args.timeout)
        if result['returncode'] != 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'status': 'FAIL', **result}, indent=2) + '\n')
        print(result['output'])
        return 0 if result['returncode'] == 0 else 1
    import torch
    adapter = SyntheticAdapter(args.op, args.iterations, args.seed, context=args.context,
                               partitions=args.partitions, partial_index=args.partial_index,
                               branch=args.branch, rotary=args.rotary, empty_first=args.empty_first, negative_scores=args.negative_scores)
    adapter.reset()
    reference = adapter.reference()
    output = adapter.run()
    torch.cuda.synchronize()
    check = adapter.check(output, reference)
    result = {'status': 'PASS' if check['passed'] else 'FAIL', 'metadata': adapter.metadata, 'correctness': check}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result))
    return 0 if check['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
