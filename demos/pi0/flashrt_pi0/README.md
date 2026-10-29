# FlashRT-style PiZero on the static Thor VM

This is the second BF16 implementation. `../native_pi0/` and
`local/pi0/model.so` remain the qualified reference. This package has separate
model compilation, projection operators and binary output. It executes the
complete model through the original cooperative static VM in one launch.

The implementation transfers FlashRT's compute organization into native VM
operators; it does not call FlashRT, launch its standalone CUTLASS kernels, or
replace the VM with CUDA graphs. The starting source audit is FlashRT revision
`86ef457d16ce5ea490d94493991e76078fcaa97d`, especially
`csrc/gemm/mega/flashrt_megakernel_geglu*.{cu,hpp}` and
`flash_rt/hardware/thor/shared_primitives.py`.

## Projection pipeline

| Property | Reference | FlashRT-style implementation |
|---|---|---|
| Tensor-core output tile | 128×128; 64×128 expert | Same output coverage initially |
| Reduction tile | K=64 | K=128 |
| Input stages | Two | Three |
| MMA synchronization | Wait after every K tile | Issue across stages; drain at output handoff |
| Shared-page map | Rotating logical pages | Fixed identity mapping for contiguous operands |
| Output epilogues | BF16 residual, bias/GELU, interleaved GeGLU, optional QKV/RoPE/cache | Retained in native consumers |
| Gemma attention | Two-pass eager-compatible scores/probabilities | Single-pass fused attention; separate state/action tiles |

Each stage reserves four contiguous 16-KiB VM pages: two for A and two for B.
Three stages occupy 192 KiB. The loader waits for stage retirement before
overwriting it. The launcher waits for TMA readiness, issues ordered MMA work,
and waits for the final completion before publishing the accumulator to the
consumer. Output consumption acknowledges TMEM reuse before the next tile.
Instruction counters continue to describe produced row bands.

The original static stream scheduler owns CTA work. Unlike FlashRT's standalone
2×2 cluster kernels, this first implementation retains the VM's one-CTA cluster
and checked one-CTA-per-SM residency. BF16 rounding at residual/activation
boundaries and the FP32 patch/action heads follow the qualified reference.
FlashRT's current PiZero FP16/FP8 precision policy is not imported.

Gemma prefix and action attention use a 64-query, 64-key, HD256 BF16
FlashAttention device collective, with GQA reading the resident KV cache.
The state query gets a separate tile restricted to prefix+state; action queries
see all keys. No full attention-score tensor is materialized. For action
sequences of at most 16 rows, a one-warp eager-compatible instruction avoids
the wasted 64-query tile and preserves the small-workload numerical gate.

The model retains setup-time action/time composition, fixed timestep biases,
last-prefix K/V pruning, resident prefix caches, merged QKV and GeGLU epilogues,
and final Euler-update epilogues. Normalization and vision attention reuse the
qualified reference device implementations. The small-query Gemma path also
reuses its numerical algorithm with a narrower warp group. Additional GEMM
and cluster tactics can be introduced here independently of the reference.

## Build and compare

From the Megakernels root:

```bash
source demos/pi0/env.sh
bash demos/pi0/build_flashrt.sh > local/pi0/flashrt-build.log 2>&1
python demos/pi0/scripts/validate_native_pi0.py --implementation flashrt \
  --report local/pi0/flashrt/full.json
python demos/pi0/benchmarks/matched_native.py --implementation flashrt \
  --library local/pi0/flashrt/model.so --backends native eager graph \
  --report local/pi0/flashrt/matched.json
```

The build manifest records both backend and shared reference sources. All
reports identify the selected implementation. Use the same checkpoint, seed,
shapes, power profile and measurement boundary for comparisons.
