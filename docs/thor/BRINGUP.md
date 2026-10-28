# Native Jetson Thor enabling port

This branch retains the upstream static megakernel and Llama operator arithmetic. It removes the experimental tuning variants and model-specific optimizations. The previous implementation and all historical measurements remain on `archive/thor-experiments-20261027`.

Retained changes:

- Explicit native SM110a build using the pinned ThunderKittens Hopper/TMA compatibility path, without B200 tensor-memory code.
- ARM64 signed-character conversion and semaphore-wait compiler memory clobbers in a project-local ThunderKittens overlay; the submodule remains unchanged.
- Logical `blockIdx.x` worker indexing, bounded instruction serialization, and splitting oversized up/gate instruction payloads for Thor's smaller SM count.
- Checked cooperative launch and entry grid synchronization; actual compiled occupancy and shared-memory admission checks prevent oversubscribed persistent dependency waits.
- Fixed static 96-register cap to fit the original 640-thread CTA; upstream H100/B200 register behavior remains available on their targets.
- Per-thread instruction and pipeline semaphore participation, explicit global acquire/release publication, integer nanosleep delay, and TMA completion/page-reuse fixes.
- Attention partition ring phases, empty partitions and negative-maximum initialization correctness fixes. These prevent invalid/nonfinite execution and do not add the compensated-attention specialization.

The retained execution is 16 consumer warps, three weight stages, thirteen 16-KiB pages, one persistent CTA per SM, original static round-robin instruction queues and original SIMT matvec. There is no ready queue, multi-CTA mode, graph segmentation, activation grouping, alternative register policy, compensated attention, FP32 down-projection finalization or BF16 matching specialization.

## Local build

```sh
source tools/thor/env.sh
python -m tools.thor.inspect_device --output local/thor-inventory.json
python -m tools.thor.prepare_tk
make -B -C demos/low-latency-llama GPU=THOR
python -m tools.thor.smoke --execute --output local/thor-smoke.json
```

The existing project-local CUDA toolkit and Python environments are used. No system CUDA, driver, firmware or power settings are changed. `tools/thor/setup_local.py` and `setup_vllm.py` retain the local setup commands. The model is the user's local `local/models/Llama-3.2-1B-Instruct` checkpoint; benchmarking does not download weights.

The default worker count is 20 for this T5000. `THOR_WORKERS` selects the physical worker count when building for a different Thor device, and `THOR_CONFIG='{"workers":20}'` must match at runtime. The binding verifies the actual device, queue dimensions, occupancy, allocation and source/binary fingerprints.

Run host checks with `python -m unittest discover -s tests -p 'test_thor*.py'`. Compile original data-center paths with `GPU=H100` or `GPU=B200` and a separate `TARGET`; execution on those GPUs is not tested here.

See [the benchmark report](BENCHMARKS.md) for the stripped port's fresh comparison against vLLM. Numerical metrics are retained for inspection; at the user's direction, small reference tolerance differences do not block timing. Nonfinite output and execution/memory errors remain failures.
