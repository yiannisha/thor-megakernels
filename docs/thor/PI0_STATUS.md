# Native PiZero correctness and final validation

2026-10-28. Numerical fixes and final validation are complete. Work stops here
at the user's request; further optimization, compiled-baseline comparisons and
trained-policy evaluation are deferred. These results use the same randomly
initialized OpenPI weights and inputs in both implementations.

The complete model executes in one cooperative launch of the original static
Megakernels VM: 27 vision layers, 18 prefix layers/cache construction, ten
18-layer expert passes, FP32 heads and all Euler updates. No generic ready queue
or per-stage host model dispatch is used. The Llama defaults are unchanged.

## Final numerical gates

| Workload | Action NRMSE | Normalized max error | Cosine | Result |
|---|---:|---:|---:|---|
| Default full model | 0.6331% | 0.024961 | 0.99998003 | PASS |
| 16-token prompt | 0.7961% | 0.026326 | 0.99996829 | PASS |
| One view / eight actions / one step | 1.4453% | 0.037321 | 0.99989557 | PASS |

The limits remain NRMSE ≤ 0.015625, normalized maximum error ≤ 0.0625, and
cosine ≥ 0.999. They were not relaxed. All cases pass identical replay and
changed-noise/image/state comparisons; the prompted case also checks changed
token IDs. The full workload passes 50 bit-exact identical replays. CUPTI
records one VM launch for each measured forward.

Reports: [full](pi0-validation/full.json), [prompted](pi0-validation/prompt16.json),
[small](pi0-validation/small.json).

## Corrections

- Vision LayerNorm uses the oracle's vectorized Welford evaluation order.
- Vision attention uses PyTorch's pinned FlashAttention device arithmetic inside
  a native range operator. A generated overlay uses the 128-thread consumer
  barrier and VM-owned shared pages. It does not launch a separate attention
  kernel. All 27 vision layers match the oracle bit-for-bit in the numerical
  diagnostic runs.
- Gemma RMS normalization uses ATen's vector-accumulator/reduction order and
  logical row counts. Padded storage remains available to GEMMs; padding no
  longer selects a different RMS reduction geometry.

The earlier approach spent too long approximating BF16 arithmetic before
comparing the exact reference evaluation order. Source pins and adaptations are
listed in [PROVENANCE.md](../../demos/pi0/PROVENANCE.md).

## Final build and checks

Selected configuration: SM110a, 20 logical workers, four consumer and four
service warps, 255-register cap, two projection stages, thirteen 16-KiB pages,
64-query expert tiles and pinned HD96/M128/N64 vision arithmetic. The binding
checks cooperative support and compiled one-CTA-per-SM occupancy.

Binary SHA-256: `aad9697ec6a0c8bd68ffa380684d73cc69c07d025eaf7c5ff163ded5ae458441`.
[Build manifest](pi0-validation/build.json) records internal sources, compiler,
Torch revision, TK overlay, FlashAttention overlay and CUTLASS revision. Source
and binary digests were checked against the final reports.

- Native VM memcheck: zero errors on the complete workload's first launch.
- Native VM synccheck: zero errors across the complete workload and replays.
- Both sanitizer runs also pass numerical and changed-input checks.
- The 15 existing/new Thor host tests pass; Python/shell syntax and diff checks
  pass.

Sanitizers use `--kernel-name kns=NativeModelGlobals`, excluding the Torch
oracle's kernels. Memcheck uses `--launch-count 1`; synccheck checks every
matching launch. CUPTI tracing is disabled under the sanitizers.
[Commands and exit codes](pi0-validation/runs.json),
[memcheck](pi0-validation/memcheck.txt),
[synccheck](pi0-validation/synccheck.txt),
[host checks](pi0-validation/host-tests.txt).

## Performance follow-up

At the user's subsequent request, the unchanged final binary was timed with
three warmups and 15 samples per backend. Same model/weights and full default
workload: three views, chunk 50, ten steps, zero prompt. No other compute
processes were observed during the samples.

| Backend | Median CUDA-event latency |
|---|---:|
| Final native VM | 439.183 ms |
| OpenPI eager | 213.107 ms |
| OpenPI explicit CUDA graph | 151.264 ms |

The earlier approximately 426 ms native result was from a separate run of the
previous arithmetic version. The final measurement is approximately 3.1% higher;
this is not a controlled attribution of the difference to a particular change.
The final native result still passes the numerical smoke limits (0.6494% NRMSE).
[Raw samples and environment](pi0-validation/performance.json).

This follow-up only measured the final build. No tuning was performed and the
goal remains paused. Reproduction and timing contracts:
[demo README](../../demos/pi0/README.md).
