# Native PiZero on Jetson Thor

This demo executes the complete ordinary π0 model in one cooperative launch of
this repository's static Megakernels VM: 27 SigLIP layers, Gemma prefix/cache,
and all ten 18-layer action-expert/Euler passes. Python prepares fixed-shape
programs and resident weights/buffers; forward supplies input pointers and
launches the VM. There is no per-stage host loop or generic device task queue.

There are two independent implementations:

- `native_pi0/`: the numerically qualified reference and `build.sh`.
- [flashrt_pi0/](flashrt_pi0/README.md): FlashRT-style BF16 compute operators,
  built with `build_flashrt.sh` into `local/pi0/flashrt/model.so`. Select it
  with `--implementation flashrt` in the validator or matched benchmark runner.

The second implementation keeps the same one-launch static VM contract while
using K=128 projection tiles, three TMA stages, asynchronous MMA retirement and
fused Gemma attention. Its build and results are separate from the reference.
See [the second implementation report](../../docs/thor/PI0_FLASHRT.md) for
numerical gates, sanitizer checks and matched measurements.

The default full workload, 16-token prompt and small diagnostic shape pass the
unchanged random-weight action smoke gate, including identical replay and changed
inputs. See the status report for final-build checks. Trained-policy evaluation
and further optimization are deferred at the user's request.

## Build and run

From the Megakernels root, using the existing local Thor toolkit and PyTorch:

```bash
bash demos/pi0/setup.sh
source demos/pi0/env.sh
bash demos/pi0/build.sh > local/pi0/build.log 2>&1
python demos/pi0/scripts/validate_native_pi0.py --benchmark --report local/pi0/full.json
python demos/pi0/benchmarks/matched_native.py --library local/pi0/model.so \
  --backends native eager graph --report local/pi0/matched.json
```

Setup creates an isolated environment for OpenPI's pinned Transformers overlay,
reusing the already installed Thor PyTorch. It fetches OpenPI at
`215abfb217dbac7d5f1273282331b9b1866c0479`, the pinned FlashAttention arithmetic,
and CUTLASS headers if missing. It does not download model weights or replace
the JetPack stack. Supply `--checkpoint /path/model.safetensors` for
same-checkpoint tests. Random weights are the default.

The default workload is batch one, three 224×224 views, zero prompt tokens,
50 actions, 32 action dimensions, and ten denoising steps. Validation accepts
`--views`, `--prompt`, `--chunk`, `--steps`, `--seed`, and `--replays`. These
shape controls do not imply every shape is numerically qualified. Inputs are
GPU-resident normalized images, state, token IDs and supplied diffusion noise.
Preprocessing, transfers, model loading and setup are outside inference timing.

Native inference is not CUDA-graph-capture compatible: host-side replay epochs
advance once per launch. Calls on an instance must be serialized on one stream;
use separate instances for concurrent work. Fixed-shape programs own their
output/scratch buffers, so clone outputs before another forward if needed.

## Runtime and tuning

The Thor target is explicitly `compute_110a → sm_110a`. Unlike the unchanged
Llama enabling path, this demo opts into TK's native TCGen05/TMEM operations.
It uses four consumer warps and four service warps, thirteen 16-KiB pages,
logical CTA instruction streams, cooperative entry synchronization, and checked
one-CTA-per-SM occupancy. The default worker count is 20; the binding checks the
actual device capacity. The Llama register limit remains 96 unless overridden.

Build experiments can set `PI0_REGISTERS`, `PI0_LINEAR_STAGES` (2 or 4),
`PI0_VISION_KEY_TILE` (64 or 128), `PI0_WEIGHT_LOCALITY` (0 or 1), and
`PI0_OUTPUT`. The JSON beside a binary records these settings and source hashes.
The matched runner supports fusion ablations, explicit/folded RMS normalization,
and diagnostic component launches. Its `--expert-qtile 32` path is experimental;
64 rows is the selected default. Component launches are profiling controls;
ordinary inference retains one kernel.

Keep CUPTI tracing separate from Compute Sanitizer. For example:

```bash
compute-sanitizer --tool memcheck --kernel-name kns=NativeModelGlobals \
  --error-exitcode 99 python demos/pi0/scripts/validate_native_pi0.py \
  --skip-trace --report local/pi0/memcheck.json
```

The kernel filter avoids instrumenting unsupported standalone cuBLAS kernels
used by the OpenPI oracle. Without `--launch-count`, every matching VM launch,
including replays, is checked. Report the actual filter and launch count with
any sanitizer claim. A numerical gate failure is separate from sanitizer errors.

See [the plan](../../docs/thor/PI0_PLAN.md) and
[implementation status](../../docs/thor/PI0_STATUS.md).
