# Faithful BF16 PiZero port — implementation progress

Workstream started 2026-10-28. This backend lives in
`demos/pi0/faithful_pi0`; the reference and partial `flashrt_pi0` backend remain
separate. The initial multi-launch native model is implemented; its performance
is not qualified against the shadow.

The one-VM-launch requirement conflicts with unchanged source collectives:
NVIDIA forbids mixing `tcgen05.cta_group::1` (SigLIP FMHA) and group 2
(selected prefix GEMMs/GeGLU) in one kernel. The compiler rejected the combined
512-thread prototype. The faithful path now uses separate native VM kernel
types joined in a CUDA graph; a single-megakernel variant requires a documented
collective adaptation. This is an ISA constraint, not a scheduler flag.

## Implemented and exercised

- Isolated FlashRT revision `86ef457d16ce5ea490d94493991e76078fcaa97d` and
  CUTLASS v4.4.2 `da5e086dab31d63815acafdac9a9c5893b1c69e2`.
- BF16 adaptations of source normalization, softmax/state masking, NHWC patch
  packing, adjacent-pair RoPE, strict vision rounding, activations, SigLIP FMHA,
  selected prefix GEMMs and fused two-phase shared-auxiliary GeGLU.
- Standalone complete BF16 shadow: three views, 768 visual tokens, 50 actions,
  ten denoising steps; resident source GEMM/GeGLU plans, reused transformer
  scratch, CUDA graph capture and identical replay outputs. About 95 MB scratch.
- Static cluster scheduler replacing CLC work discovery while retaining the
  source GEMM mainloop and epilogue. Native QKV and O execute through the original
  VM fetch/controller/worker/page lifecycle, including controller participation.
- Compact VM metadata, explicitly aligned shared arena, flattened multidimensional
  worker IDs, 2x2 cluster launch and lifetime group-2 TMEM ownership. A probe
  admitted four resident clusters / 16 CTAs, rather than assuming 20 workers.
- QKV and O tests: one, two and eight consecutive instructions match cuBLAS BF16
  exactly. Graph replay tests overwrite output with NaNs before replay.
- Fused GeGLU is source-output exact for one, two and eight consecutive VM
  instructions. A 512-thread variant also passes QKV, O and fused GeGLU.

## Measurements and acceptance limits

The full three-view / 50-action / ten-step native model measured 240.06 ms
versus 93.01 ms for the BF16 shadow, with 0.01273 action NRMSE and exact
repeated/captured replay. This used 89 VM regions; it is not a single megakernel.
See `local/pi0/faithful/native-full.json`. The one- and three-megakernel
experiments are described in [PI0_COHERENT.md](PI0_COHERENT.md).


`local/pi0/faithful/` contains build logs and JSON results. Early direct-call
source down projection measured 0.709 ms versus the older partial backend's
1.960 ms. These separate measurements are preliminary; prefer the matched warm
CUDA-graph benchmark in `scripts/benchmark_faithful_vm.py` for engine overhead.

The complete standalone shadow initially measured about 93 ms with graph replay
and resident input copies. That is not native-model latency. Separate comparison
with OpenPI gave 0.04615 action NRMSE on random weights; the BF16 head/update
policy differs from OpenPI. Source-policy fidelity and OpenPI agreement remain
separate checks. No pretrained-checkpoint qualification has been performed.

## In progress / remaining

- Logical 2x1 pair execution is implemented and source-output exact for the
  down collective; QKV, O and down have matched warm graph timing.
  Standalone/native milliseconds: QKV 0.0628/0.0571, O 0.0466/0.0700,
  down 0.6864/0.6749. O remains a measured regression to optimize.
- The source-policy program now records operations with tensor shapes, strides,
  dtype, scalars and tactic choices; cuBLAS algorithm IDs still need tracing.
- Native source SigLIP FMHA now passes exactly, including repeated instructions
  and captured replay, in its separate group-1 VM.
- Source pointwise functions and two generic TCGEN projection layouts are being
  integrated into a merged group-2 program; the full native runner is written
  but not yet qualified.
- SigLIP FMHA uses 16 warps, while the qualified GEMM-only VM currently uses
  eight. The complete VM needs explicit participation and register policies for
  both; the eight-warp GEMM proof is not sufficient for full-model integration.
- One reused-buffer model program, dependency/last-reader handoffs, source-policy
  intermediate checks, full-model numerical and performance qualification.

## Commands

Source `demos/pi0/env.sh` first for Python/OpenPI. Build with
`bash demos/pi0/build_faithful.sh` after provisioning the pinned CUTLASS checkout
under `local/pi0/faithful/cutlass` (include, tools/util/include and
examples/77_blackwell_fmha sparse paths).

Run:

```
python demos/pi0/scripts/validate_faithful_operators.py
python demos/pi0/scripts/validate_faithful_shadow.py --capture --compare-openpi
python demos/pi0/scripts/validate_faithful_vm.py
python demos/pi0/scripts/benchmark_faithful_vm.py
```
