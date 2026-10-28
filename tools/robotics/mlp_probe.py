"""Synthetic OpenPI-shaped GEMM/GELU probe; no checkpoint or native VLA claim.

Use tools.thor.bench with this factory. THOR_ROBOTICS_MODE=eager|graph and
THOR_ROBOTICS_STEPS=1..10 select execution; all weights remain synthetic.
"""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class MLPProbe:
    def __init__(self):
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA PyTorch required for robotics shape probe')
        self.mode = os.environ.get('THOR_ROBOTICS_MODE', 'eager')
        self.steps = int(os.environ.get('THOR_ROBOTICS_STEPS', '10'))
        if self.mode not in ('eager', 'graph') or not 1 <= self.steps <= 10:
            raise ValueError('mode must be eager|graph and steps must be in 1..10')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.manual_seed(1)
        self.x = torch.randn(1, 50, 1024, device='cuda', dtype=torch.bfloat16)
        self.scale = torch.randn(1024, device='cuda', dtype=torch.float32) / 32
        self.weights = [torch.randn(out, inp, device='cuda', dtype=torch.bfloat16) / math.sqrt(inp)
                        for out, inp in ((4096, 1024), (4096, 1024), (1024, 4096))]
        self.float_weights = [weight.float() for weight in self.weights]
        audit = json.loads((ROOT / 'docs/thor/openpi-source-inventory.json').read_text())
        from tools.thor.inspect_device import command
        digest = hashlib.sha256()
        for value in [self.x, self.scale, *self.weights]:
            digest.update(value.detach().cpu().view(torch.uint8).numpy().tobytes())
        self.metadata = {
            'model': 'OpenPI Gemma-300M MLP shape probe; synthetic weights',
            'precision': 'bf16', 'context': 50, 'batch': 1,
            'measurement_scope': 'robotics_block', 'engine': 'torch_' + self.mode,
            'variant': 'synthetic-gemma300m-mlp-' + self.mode,
            'power_mode': command(['nvpmodel', '-q']).get('stdout'),
            'concurrent_workload': 'uncontrolled-existing',
            'shape': {'batch': 1, 'tokens': 50, 'hidden': 1024, 'mlp': 4096},
            'repetitions_per_run': self.steps, 'weights_shared_across_repetitions': True,
            'openpi_commit': audit['commit'], 'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_and_weights_sha256': digest.hexdigest(), 'seed': 1,
            'reference': 'FP32 linear arithmetic with BF16 rounding at each matched intermediate boundary',
            'normalization': 'FP32 RMS, epsilon 1e-6, learned 1+scale; no adaptive modulation',
            'activation': 'GELU tanh approximation', 'float32_matmul_precision': 'highest',
            'gate': {'nrmse': .02, 'atol': .1, 'rtol': .05},
            'excludes': ['vision', 'text', 'attention', 'RoPE', 'adaptive RMS', 'action projection', 'flow integration'],
            'qualification_scope': 'Synthetic MLP arithmetic only; not OpenPI checkpoint parity, action quality, or vision-to-action latency',
            'timing_includes': ['normalization', 'three dense projections', 'GELU', 'gating', 'residual', 'all repetitions'],
            'timing_excludes': ['initialization', 'reference', 'correctness_checks', 'graph_capture']}
        if self.mode == 'graph':
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._forward(False)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self._forward(False)

    def _forward(self, reference):
        torch = self.torch
        x = self.x
        weights = self.float_weights if reference else self.weights
        def linear(value, weight):
            return torch.nn.functional.linear(value.float() if reference else value, weight).to(torch.bfloat16)
        for _ in range(self.steps):
            normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
            normalized = (normalized * (1 + self.scale)).to(torch.bfloat16)
            gate = torch.nn.functional.gelu(linear(normalized, weights[0]), approximate='tanh')
            mixed = gate * linear(normalized, weights[1])
            x = x + linear(mixed, weights[2])
        return x

    def reset(self):
        # Every invocation starts from immutable self.x, including graph replays.
        pass

    def run(self):
        if self.mode == 'graph':
            self.graph.replay()
            return self.output
        return self._forward(False)

    def reference(self):
        return self._forward(True)

    def check(self, output, reference):
        torch = self.torch
        a, b = output.float(), reference.float()
        difference = a - b
        finite = bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())
        nrmse = (difference.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)).item()
        pointwise = bool(torch.allclose(a, b, atol=.1, rtol=.05))
        return {'passed': finite and pointwise and nrmse <= .02,
                'finite': finite, 'nrmse': nrmse, 'pointwise_passed': pointwise,
                'max_abs': difference.abs().max().item(), 'mean_abs': difference.abs().mean().item(),
                'nrmse_limit': .02, 'atol': .1, 'rtol': .05}


def make_adapter():
    return MLPProbe()
