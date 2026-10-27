"""Matched-cache one-token decode adapters for native, PyTorch eager/graphs and PyVM.

All weights are local. Prefill/loading and restoring the frozen KV cache are
outside timing; timed decode includes embedding, runtime work, lm-head and argmax.
This measures same-cache replay, not 100 autoregressive tokens or prefill.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from .config import Config
from .inspect_device import ROOT, command, driver_devices
from .model_check import errors, load_reference
from .sweep import source_hash


class ModelAdapter:
    @torch.inference_mode()
    def __init__(self, engine='native', context=128, seed=1, cache_capacity=0, reference_attention=None):
        if engine not in ('native', 'torch', 'torch_graph', 'pyvm'):
            raise ValueError('unsupported engine')
        self.engine = engine
        self.config = Config(**json.loads(os.environ.get('THOR_CONFIG', '{}')))
        self.logit_limit = float(os.environ.get('THOR_LOGIT_NRMSE_LIMIT', '.03'))
        if not 0 < self.logit_limit < 1:
            raise ValueError('logit NRMSE limit must lie between zero and one')
        model_path = Path(os.environ.get('THOR_MODEL_PATH', str(ROOT / 'local/models/Llama-3.2-1B-Instruct')))
        self.case = load_reference(SimpleNamespace(model=model_path, context=context, seed=seed,
                                                   ordered_downproj=self.config.ordered_downproj, cache_capacity=cache_capacity,
                                                   reference_attention=reference_attention or os.environ.get('THOR_REFERENCE_ATTENTION', 'flash')))
        self.model = self.case.model
        self.position = torch.tensor([[context]], device=self.model.device)
        self.context = context
        from megakernels.model_types import BatchState
        self.state = BatchState(input_ids=self.case.token, position_ids=self.position, seq_len=context+1)
        self.captured_logits = None
        self.hook = self.model.lm_head.lm_head.register_forward_hook(self._capture_logits)
        self.globs = None
        resources = None
        build = None
        if engine in ('native', 'pyvm'):
            from megakernels.demos.latency.scheduler import LatencyScheduleBuilder
            from megakernels.scheduler import Schedule, assign_to_sms, tensorize_instructions
            self.globs = LatencyScheduleBuilder.make_globals(self.model)
            self.globs.set_active_workers(self.config.workers)
            nodes, end = LatencyScheduleBuilder.make_dag(self.globs)
            self.schedule = Schedule(self.globs, nodes, end)
            tensorize_instructions(self.globs, assign_to_sms(self.config.scheduler, schedule=self.schedule))
        if engine == 'native':
            from megakernels.demos.latency.mk import interpret_with_mk
            sys.path.insert(0, os.environ.get('THOR_BINARY_DIR', str(ROOT / 'demos/low-latency-llama')))
            import mk_llama
            self.native = mk_llama.mk_llama
            self.interpret = interpret_with_mk
            resources = {}
            self.interpret(self.globs, lambda *a, **k: resources.update(self.native(*a, **k, dry_run=True)))
            if resources.get('build_source_hash') != source_hash():
                raise RuntimeError('stale native binary; rebuild before benchmarking')
            expected = {'active_workers': self.config.workers,
                        'weight_pipeline_stages': self.config.weight_stages, 'page_count': self.config.page_count,
                        'page_size': self.config.page_size, 'consumer_warps': self.config.consumer_warps,
                        'backoff_ns': self.config.backoff_ns, 'timing_enabled': self.config.timing,
                        'ordered_downproj': self.config.ordered_downproj,
                        'fp32_downproj': self.config.fp32_downproj,
                        'match_torch_bf16_epilogues': self.config.match_torch_bf16_epilogues,
                        'compensated_attention': self.config.compensated_attention,
                        'register_mode': self.config.register_mode,
                        'consumer_register_target': self.config.consumer_registers,
                        'service_register_target': self.config.service_registers,
                        'match_torch_bf16_rms': self.config.match_torch_bf16_rms,
                        'instruction_barrier_parties_per_warp': self.config.instruction_parties}
            if any(resources.get(k) != value for k, value in expected.items()):
                raise RuntimeError(f'compiled configuration does not match requested config: {resources}')
            device = driver_devices()['devices'][0]
            invalid = self.config.validate(device)
            if invalid:
                raise ValueError(invalid)
            binary = Path(mk_llama.__file__)
            manifest = Path(str(binary) + '.build.json')
            if not manifest.exists():
                raise RuntimeError('native build manifest missing; rebuild before benchmarking')
            build = json.loads(manifest.read_text())
            with binary.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if build['binary_sha256'] != digest or build['source_hash'] != source_hash():
                raise RuntimeError('native build manifest does not match the loaded binary/source')
        self.metadata = {'model': model_path.name, 'model_provenance': self.case.provenance,
                         'precision': 'bf16', 'reference_attention': self.case.reference_attention,
                         'prefill_attention': self.case.prefill_attention,
                         'float32_matmul_precision': torch.get_float32_matmul_precision(), 'context': context, 'batch': 1, 'seed': seed,
                         'measurement_scope': 'decode', 'tokens_per_run': 1, 'engine': engine,
                         'variant': f'{engine}:{self.config.name}', 'config': self.config.to_dict(),
                         'power_mode': command(['nvpmodel', '-q']).get('stdout'),
                         'concurrent_workload': 'uncontrolled-existing', 'same_cache_replay': True,
                         'input_hash': hashlib.sha256(self.case.inputs.cpu().numpy().tobytes()).hexdigest(),
                         'native_resources': resources, 'native_build': build,
                         'gate': {'logit_nrmse_limit': self.logit_limit, 'state_nrmse_limit': .02},
                         'timing_includes': ['embedding', 'runtime_dispatch', 'decoder', 'lm_head', 'argmax'],
                         'timing_excludes': ['model_loading', 'prefill', 'frozen_KV_restore', 'correctness_checks'],
                         'cuda_event_note': 'interval includes CPU enqueue gaps; all work uses current stream',
                         'dram_gb_s': None, 'gpu_power_w': None, 'energy_per_token_j': None}
        if engine == 'torch_graph':
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.reset()
                    self._run_torch()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.reset()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._run_torch()
            self.graph_logits = self.captured_logits
            self.graph_hidden = self.last_hidden
        self.reset()

    def _capture_logits(self, module, inputs, output):
        self.captured_logits = output.flatten()

    @torch.inference_mode()
    def reset(self):
        for tensor, before in zip(self.model.stacked_kv_cache, self.case.cached):
            tensor.copy_(before)

    @torch.inference_mode()
    def reference(self):
        return self.case.ref_logits

    def _run_torch(self):
        output = self.model(self.state)
        self.last_hidden = output.hidden_states.flatten()
        return self.captured_logits

    @torch.inference_mode()
    def run(self):
        if self.engine == 'torch':
            return self._run_torch()
        if self.engine == 'torch_graph':
            self.graph.replay()
            self.last_hidden = self.graph_hidden
            return self.graph_logits
        from megakernels.model_types import BatchState
        self.globs.hidden_states.copy_(self.model.model.embed_tokens(BatchState(input_ids=self.case.token)).hidden_states.flatten())
        self.globs.barriers.zero_()
        self.globs.pos_id = self.context
        if self.engine == 'native':
            self.interpret(self.globs, self.native)
        else:
            from megakernels.demos.latency.python_vm import INSTRUCTION_TO_SOLVER
            from megakernels.python_vm import interpret_with_pyvm
            interpret_with_pyvm(self.globs, self.schedule.get_linear_instructions(), INSTRUCTION_TO_SOLVER)
        self.last_hidden = self.globs.hidden_states
        # Include the same greedy selection done by the PyTorch model.
        self.last_token = self.globs.logits.argmax()
        return self.globs.logits

    @torch.inference_mode()
    def check(self, output, reference):
        metrics = {'logits': errors(output, reference, nrmse_limit=self.logit_limit),
                   'hidden': errors(self.last_hidden, self.case.ref_hidden),
                   'k_cache': errors(self.model.stacked_kv_cache[0][:, 0, self.context], self.case.ref_cache[0]),
                   'v_cache': errors(self.model.stacked_kv_cache[1][:, 0, self.context], self.case.ref_cache[1])}
        return {'passed': all(m['passed'] for m in metrics.values()), 'metrics': metrics,
                'greedy_token_matches': bool(output.argmax() == reference.argmax())}


def make_adapter():
    return ModelAdapter(os.environ.get('THOR_ENGINE', 'native'),
                        int(os.environ.get('THOR_CONTEXT', '128')), int(os.environ.get('THOR_SEED', '1')))
