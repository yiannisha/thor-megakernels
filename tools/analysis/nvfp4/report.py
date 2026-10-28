"""Summarize measured primitives and optional persistent runtime comparisons."""
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs/thor/nvfp4'
def main():
    data=json.loads((OUT/'primitive-streaming.json').read_text())
    assert len(data['cases'])==4
    for case in data['cases']:
        for name,result in case['results'].items():
            assert len(result['samples_ms'])==30
            if result['error'] is not None:
                assert result['error']['finite'] and result['error']['relative_l2']<=0.001
    qualified=json.loads((OUT/'persistent-sanitizers.json').read_text())
    assert len(qualified)==6 and all(x['returncode']==0 for x in qualified)
    for name in ('primitive','persistent-auto','persistent-2'):
        for tool in ('memcheck','synccheck','racecheck'):
            log=(OUT/f'{name}-{tool}.txt').read_text()
            assert ('ERROR SUMMARY: 0 errors' in log or '0 hazards displayed (0 errors, 0 warnings)' in log)
    rows=[]
    for case in data['cases']:
        r=case['results'];bf=r['torch_bf16_mv']['median_ms'];tc=r['w4a4_tcgen05_including_activation_quant']['median_ms']
        rows.append(f"| {case['m']} × {case['k']} | {bf*1000:.2f} | {r['w4a16_simt']['median_ms']*1000:.2f} | {tc*1000:.2f} | {bf/tc:.2f}× |")
    runtime_rows=[]
    for path in sorted(OUT.glob('runtime-*.json')):
        d=json.loads(path.read_text())
        assert len(d['per_instruction_error_vs_identical_quantized_graph'])==d['length']
        assert all(x['finite'] and x['relative_l2']==0 for x in d['per_instruction_error_vs_identical_quantized_graph'])
        assert all(len(values)==3 and all(len(x['samples_ms'])==30 for x in values) for values in d['variants'].values())
        med={k:statistics.median([v['median_ms'] for v in values]) for k,values in d['variants'].items()}
        runtime_rows.append(f"| {d['width']} / {d['length']} | {d['stages']} | {d['shared_bytes']} / {d['allocated_shared_bytes']} | {med['graph']*1000:.2f} | {med['persistent']*1000:.2f} | {med['graph']/med['persistent']:.3f}× |")
    text='''# Thor NVFP4 investigation

Scope: generic batch-one linear primitives and an experimental cooperative FP4
instruction runtime. The enabling-only production BF16 VM and its previously
measured model/vLLM comparison remain unchanged. These results do not establish
a new Llama or vLLM end-to-end speedup.

## Primitive measurements

Resident prequantized weights. Native `sm_110a` CUTLASS block-scaled tcgen05 MMA,
FP32 accumulation, BF16 outputs. A separate W4A16 SIMT control expands weights
only in registers, with CUDA's native FP4-to-half conversion. Its simple kernel
is a feasibility control, not a tuned weight-only implementation.

Cold-cache protocol: 128 MiB GPU eviction buffer before each single-operation
CUDA graph; eviction excluded from timing. 30 event samples per case. The FP4
column includes dynamic per-16-value activation scales and packing, and the
extra quantization launch. No reference BF16 expansion is in the timed path.
Weights' global FP32 scale and E4M3 block scales are computed once at setup.
Activation tensor scales are fixed from the setup input amax; activation block
scales are dynamic. A deployment needs calibration or a dynamic global-scale
policy; dynamic global amax reduction is not included here.

| Output × reduction | Torch BF16 µs | W4A16 SIMT µs | Native W4A4 incl. quant µs | BF16 / W4A4 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

All input/output tensors are device resident. Synthetic random BF16 weights;
no trained-model quality assertion. Errors against the same decoded quantized
values and differences from original BF16 are recorded separately in raw JSON.
An independent 16-code FP4 lookup validates the CUDA decoder. Golden fixtures
cover exact FP4 values, partial 128-row tiles, and multiple K extents. Memcheck,
synccheck and racecheck reports are retained beside the measurements.

NVFP4 uses 4-bit values plus an E4M3 scale per 16 values: 0.5625 bytes/value,
3.56× smaller than BF16 before layout padding. Data stays compressed in LPDDR.
A native scale tensor has a padded/interleaved physical layout; actual allocated
bytes, not just logical scale count, are reported per shape.

## Runtime and storage work

`experimental/quantized_runtime/format_storage.cuh` separates packed data,
scales, alignment, activations and stages. A 16×2048 weight stage needs 65,536
BF16 bytes or 18,432 canonical NVFP4 bytes. The compile-time page planner guards
the existing VM's one-warp/32-page map limit. This planner is an experimental
building block; it has not replaced the production VM's page allocator.

The cooperative prototype executes dependent generic linear instructions inside
one kernel. Quantization reads each previous BF16 result on GPU. A static
scheduler admits at most one CTA per SM and no more CTAs than initial output
tiles. Twenty SMs and 128-row MMA tiles mean a 2048-row operation initially
uses 16 CTAs; that granularity matters on Thor.

CUTLASS normally relinquishes its TMEM allocation permit before returning from
a kernel. Repeated allocation from the same CTA after relinquishing is invalid.
A generated, narrowly scoped overlay adds a descriptor flag to retain the permit
across instructions, then releases it once at loop exit. Ordinary graph kernel
descriptors retain the original behavior. The pinned upstream checkout is not
modified. The instruction loop also waits for complete TMA global stores before
publishing results at the grid barrier.

Automatic staging and two-stage storage are compared using the same arithmetic.
The two-stage prototype reserves 116,864 bytes to guarantee one resident CTA
per SM despite its smaller 83,968-byte payload. Smaller shared allocations do
not by themselves prove higher occupancy: registers,
TMEM, work granularity and the cooperative admission limit still apply.

'''
    if runtime_rows:
        text+='''| Width / instructions | Stages | Shared payload / reserved bytes | Graph chain µs | Persistent chain µs | Graph / persistent |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(runtime_rows)+'\n\n'
    text+='''The prototype is slower than the equivalent FP4 CUDA graph at both tested
widths. Reducing the pipeline to two stages also failed to establish a latency
win. It remains an experimental negative result, not a production optimization.
The native FP4 arithmetic/weight-traffic result is separate from this runtime
comparison. Each variant gets 1,000 untimed graph replays before measurement;
three AB/BA blocks of 30 samples are retained. The chains contain eight distinct
square matrices and real output-to-input dependencies. The 4096-wide fixture
is synthetic, not a benchmark of a larger trained model.

The runtime experiment is not wired into the existing mixed attention/linear
VM. That integration requires a quantized operand descriptor, native MMA/TMEM
roles, format-aware page ownership, and operator-by-operator validation. Attention,
residual arithmetic, normalization and output buffers retain their own precision
and storage needs. No global page shrink or model-specific specialization has
been applied to the production branch.

## Reproduce

Use CUDA 13.2 via `source tools/thor/env.sh`. Clone NVIDIA CUTLASS to
`local/cutlass-nvfp4` and checkout revision
`91843b71d90737e0a9b86f7d9df213cff5a3526f`.

```sh
bash tools/analysis/nvfp4/setup_cutlass.sh
bash tools/analysis/nvfp4/build.sh
python -m tools.analysis.nvfp4.check
python -m tools.analysis.nvfp4.bench --streaming --inner 1 --repeats 30 --output docs/thor/nvfp4/primitive-streaming.json
bash experimental/quantized_runtime/build.sh auto
bash experimental/quantized_runtime/build.sh 2
python -m tools.analysis.nvfp4.runtime_bench --library local/thor-nvfp4/persistent-auto.so --width 2048 --length 8 --output docs/thor/nvfp4/runtime-auto-2048.json
python -m tools.analysis.nvfp4.runtime_bench --library local/thor-nvfp4/persistent-2.so --width 2048 --length 8 --output docs/thor/nvfp4/runtime-2-2048.json
python -m tools.analysis.nvfp4.report
```

Run GPU measurements serially, including work from other sessions. Shared system
load and clocks are uncontrolled; report observed latency, not a guaranteed gain.
Do not use sanitizer timings as performance results.

Next integration should reuse native FP4 MMA inside the existing VM, with TMEM
owned across instructions and a measured format-aware staging policy. Wrapping a
complete standalone GEMM body inside a global instruction loop is not sufficient.
The current prototype establishes correctness mechanisms and a negative latency
control; it does not demonstrate that the existing VM can simply embed that
body efficiently. A trained-model quantization quality check and fresh model/vLLM
comparison remain necessary after actual VM integration.

Implementation basis: [NVIDIA CUTLASS NVFP4 BF16 example](https://github.com/NVIDIA/cutlass/blob/91843b71d90737e0a9b86f7d9df213cff5a3526f/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu).

'''
    (ROOT/'docs/thor/NVFP4.md').write_text(text.rstrip()+'\n')
if __name__=='__main__':main()
