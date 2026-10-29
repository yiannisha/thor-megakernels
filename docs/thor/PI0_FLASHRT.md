# FlashRT-style BF16 PiZero implementation

2026-10-28. The second implementation lives in
[`demos/pi0/flashrt_pi0`](../../demos/pi0/flashrt_pi0/README.md). Its Python
entry point is `FlashPi0`, its binary is `local/pi0/flashrt/model.so`, and its
device globals are `FlashModelGlobals`. The reference sources and reference
binary remain unchanged.

This reimplements FlashRT's execution techniques as original static VM range
operators. It retains one cooperative launch for vision, prefix/cache and all
ten action-expert/Euler steps. There are no FlashRT host calls, per-operation
kernel launches, CUDA graphs or generic device task queues in native forward.

## Compute changes

- **Pipelined BF16 GEMM:** 128×128 output tiles, a 64-row expert specialization,
  K=128 reduction tiles and three TMA stages. Twelve contiguous 16-KiB VM pages
  hold the stages. The fixed identity page map makes each 32-KiB operand
  contiguous. Stage retirement protects TMA reuse; the launcher drains the
  final MMA before handing the accumulator to the consumer instead of waiting
  after every K tile.
- **Fused Gemma attention:** a BF16 FlashAttention device collective performs
  single-pass attention inside the VM consumer group, using 64-query/64-key
  tiles and the resident GQA cache. The state query has a prefix+state-only
  tile; action queries have full suffix visibility. This replaces the reference
  Gemma operator's two QK passes. Small action chunks use a 16-query, one-warp
  eager-compatible algorithm across Gemma layers, with a separate barrier ID.
- **Immutable weight packing:** QKV and interleaved gate/up weights are packed
  once per layer and shared across the ten steps. An isolated setup comparison
  reduced packed weight buffers from 8,773,042,176 to 5,205,786,624 bytes: a
  40.7% reduction, or 3.32 GiB. This is packed storage, not total model memory.

The model keeps the reference's fused residual/GELU/GeGLU output consumers,
QKV/RoPE/cache writes where their ownership permits fusion, action/time
composition, precomputed timestep biases, final-prefix pruning and FP32
patch/action heads. Numerically qualified normalization and vision attention
are shared device code. Explicit RMS normalization remains the default;
algebraic RMS folding is selectable.

The FlashRT audit uses revision
`86ef457d16ce5ea490d94493991e76078fcaa97d`. Its current PiZero paths use
FP16/FP8. This implementation uses BF16 transformer computation and FP32
accumulation/heads. It retains the VM's single-CTA clusters rather than the
standalone FlashRT CUTLASS kernels' 2×2 cluster scheduler. It is not a claim of
complete FlashRT tactic coverage or matched FlashRT performance.

## Numerical qualification

Same randomly initialized OpenPI weights and GPU-resident inputs:

| Workload | Action NRMSE | Normalized max error | Cosine | Result |
|---|---:|---:|---:|---|
| Three views / 50 actions / ten steps | 0.8238% | 0.030611 | 0.999966 | PASS |
| Same workload, 16-token prompt | 1.0345% | 0.045492 | 0.999947 | PASS |
| One view / eight actions / one step | 1.4453% | 0.037321 | 0.999896 | PASS |

The existing numerical limits are unchanged. These cases pass changed-root-input
checks and bit-exact identical replay. The default case includes 20 identical
replays. CUPTI confirms one VM launch per forward. This is numerical smoke
qualification with random weights, not trained policy or robot-task evaluation.

The selected binary passes full-workload memcheck on its first matching launch,
and full-workload plus small-workload synccheck across all matching replays.
Each tool reports zero errors; the corresponding numerical/replay gates also
pass. The kernel filter is `kns=FlashModelGlobals`; CUPTI tracing is disabled
under Compute Sanitizer. Memcheck uses `--launch-count 1`.

Reports and exact commands are in [pi0-flashrt](pi0-flashrt/runs.json):
[full](pi0-flashrt/full.json), [prompt](pi0-flashrt/prompt16.json),
[small](pi0-flashrt/small.json), [memcheck](pi0-flashrt/memcheck.txt),
[synccheck](pi0-flashrt/synccheck.txt),
[small synccheck](pi0-flashrt/small-synccheck.txt), and
[build manifest](pi0-flashrt/build.json).

## Matched performance

Full batch-one workload: three 224×224 views, zero prompt tokens, 50 actions,
ten steps. Each backend receives the same initialized model and inputs within
its benchmark subprocess. Three warmups and 15 samples per backend; CUDA-event
timing; no other compute process was observed. The reference and second backend
are measured in separate sequential subprocesses using the same seed/workload.

| Implementation | Median latency |
|---|---:|
| Unchanged reference VM | 441.392 ms |
| FlashRT-style BF16 VM | 261.580 ms |
| OpenPI explicit CUDA graph | 151.4 ms |

The second VM is **1.687× faster**, reducing latency by **40.74%**. OpenPI's
CUDA graph remains faster. Its medians are 151.448 and 151.414 ms in the two
sessions. Eager OpenPI varies from 204.705 to 229.501 ms; both complete sample
sets are retained rather than presenting that variation as a native speedup.

The second VM's matched-run action NRMSE is 0.8154%. These timings cover all
model computation from GPU-resident inputs through complete actions and exclude
setup, packing, preprocessing and transfers. They are not a matched FlashRT
FP16 comparison or a trained-policy latency/accuracy result.

[Reference samples](pi0-flashrt/reference-matched.json),
[second-backend samples](pi0-flashrt/flashrt-matched.json),
[packing comparison](pi0-flashrt/packing.json),
[summary](pi0-flashrt/summary.json).

## Follow-up decomposition and scope correction

The second backend is a **partial FlashRT-inspired implementation**, not a
faithful port of FlashRT's full compute path with only its scheduler replaced.
The 261.580 ms comparison above is against our reference and OpenPI; FlashRT
was not benchmarked on the matched workload. There is no established matched
FlashRT BF16 PiZero baseline in these results.

Diagnostic component launches measure vision at 70.421 ms, prefix at 98.894 ms,
and the ten-step expert at 91.808 ms. Their sum is approximately 261.123 ms;
the three-launch diagnostic forward measured 260.302 ms. These isolated
component medians are not an additive instrumentation trace of the one-launch
forward. [Raw component measurements](pi0-flashrt/decomposition.json).

Isolated plain BF16 projections, with identical random operands per case:

| Projection | VM ms | cuBLAS ms | VM/cuBLAS |
|---|---:|---:|---:|
| Prefix QKV, M768/N2560/K2048 | 0.216 | 0.060 | 3.61× |
| Prefix gate/up, M768/N32768/K2048 | 2.299 | 0.751 | 3.06× |
| Prefix down, M768/N2048/K16384 | 1.960 | 0.588 | 3.33× |
| Expert QKV, M51/N2560/K1024 | 0.033 | 0.021 | 1.55× |
| Expert gate/up, M51/N8192/K1024 | 0.062 | 0.032 | 1.91× |
| Expert down, M51/N1024/K4096 | 0.046 | 0.031 | 1.47× |

These are single-operation CUDA-event measurements, including possible host
submission gaps. The native path uses the isolated linear VM kernel, not the
full-model kernel. They demonstrate an operator-level gap before full-model
dependencies but do not separate VM controller cost from MMA/data-path cost.
They are not FlashRT kernel measurements. [Raw GEMM diagnostics](pi0-flashrt/gemm-decomposition.json).

Important compute paths still differ: FlashRT's 16-bit shared Thor operators
use cuBLAS for SigLIP and shape-specific CUTLASS tactics for the prefix,
including clustered GeGLU and two-SM down projections. This backend uses its
general 128×128 single-CTA projection family. Its default patch embedding and
action/time heads also retain scalar FP32 FMA loops rather than FlashRT's
tensor-core GEMMs. Each VM opcode inherits the same resident CTA, shared-memory
reservation and register policy. No measurement here establishes that these
differences are an unavoidable cost of megakernel execution.

## Reproduction

From the Megakernels root:

```bash
source demos/pi0/env.sh
bash demos/pi0/build_flashrt.sh > local/pi0/flashrt-build.log 2>&1
python demos/pi0/scripts/validate_native_pi0.py --implementation flashrt \
  --replays 20 --report local/pi0/flashrt/full.json
python demos/pi0/benchmarks/matched_native.py --implementation flashrt \
  --library local/pi0/flashrt/model.so --warmup 3 --iterations 15 \
  --backends native eager graph --report local/pi0/flashrt/matched.json
```

The validator and matched runner default to the reference unless
`--implementation flashrt` is selected. Build manifests fingerprint both the new
backend and shared reference device sources. Calls remain serialized per model
context, with fixed resident buffers and host-assigned replay epochs.
