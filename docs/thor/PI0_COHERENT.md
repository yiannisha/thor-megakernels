# One- and three-megakernel BF16 PiZero experiments

The source-matched multi-launch backend stays available. This experiment changes
incompatible device collectives so all model operations can run inside one native
VM invocation, and compares that with exactly three native VM invocations:

1. Vision: SigLIP plus the image projector.
2. VLM: visual/language prefix and KV cache production.
3. Action: state/action/time heads and every denoising step.

Both variants use the same operands, operation order, reused model scratch and
fixed 2x2 physical cluster. The first comparison deliberately holds operators and
resource configuration fixed between one and three launches. It does not claim
that independently tuned phase-specific kernels have been implemented.

## Device adaptations

- BF16 projections use TCGEN05 CTA group 2 with FP32 accumulation.
- Vision attention uses the existing pinned FlashAttention-2 device body with
  BF16 `mma.sync`, avoiding group-1 TCGEN05 instructions in the same kernel.
- Prefix down projection keeps the source tile/epilogue family but changes its
  logical cluster from 2x1 to 2x2.
- The source fused GeGLU-to-down handoff still hangs even after fixing cluster
  geometry. This variant instead uses two device GEMM collectives inside the
  opcode: gate FP32 accumulation -> GELU -> BF16 gate materialization, followed
  by up FP32 accumulation multiplied by the BF16 auxiliary in its epilogue.
  The gate and up remain inside the megakernel; no host GEMM is invoked.

These are intentional device-implementation changes, not an unchanged FlashRT
collective port. Numerical comparison with the BF16 source-policy shadow is a
required check. The existing reference and multi-launch backends remain separate.

## Build and check

```
source demos/pi0/env.sh
bash demos/pi0/build_coherent.sh
python demos/pi0/scripts/validate_coherent_pi0.py \
  --report local/pi0/faithful/coherent-full.json
```

Use `--views 1 --chunk 10 --steps 1` for a small qualification run. `--partition`
accepts `single`, `three`, or `both`. The default checks both and compares their
outputs. It also checks repeatability, graph replay, and CUPTI driver launch counts
of exactly one and three for the model execution (resident input copies excluded).

Build logs and JSON measurements live under `local/pi0/faithful/`. Until those
checks pass, the corresponding variant is not qualified. The earlier 240 ms
native / 93 ms shadow result belongs to the 89-region backend, not this experiment.
