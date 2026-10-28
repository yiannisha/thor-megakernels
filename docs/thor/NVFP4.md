# Thor NVFP4 investigation

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
| 2048 × 2048 | 46.50 | 47.09 | 28.96 | 1.61× |
| 8192 × 2048 | 191.30 | 161.63 | 55.30 | 3.46× |
| 2048 × 8192 | 199.54 | 167.78 | 56.27 | 3.55× |
| 128256 × 2048 | 2030.35 | 2364.96 | 612.16 | 3.32× |

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

| Width / instructions | Stages | Shared payload / reserved bytes | Graph chain µs | Persistent chain µs | Graph / persistent |
|---|---:|---:|---:|---:|---:|
| 2048 / 8 | 2 | 83968 / 116864 | 90.02 | 246.27 | 0.366× |
| 4096 / 8 | 2 | 83968 / 116864 | 438.10 | 728.24 | 0.602× |
| 2048 / 8 | 7 | 227328 / 227328 | 80.22 | 216.99 | 0.370× |
| 4096 / 8 | 7 | 227328 / 227328 | 448.00 | 671.68 | 0.667× |

The prototype is slower than the equivalent FP4 CUDA graph at both tested
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
