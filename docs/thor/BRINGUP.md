# Thor bring-up

## Actual platform

- Upstream baseline: `7309cec801537b61fea3b50d7dfe454a6cde578e`, initially clean.
- Pinned ThunderKittens: `664c108d16f12707a73d3072ab525f26fb2b4f62`; submodule remains unchanged.
- NVIDIA Thor, CC 11.0, 20 SMs, 125809 MiB reported by PyTorch; opt-in shared memory/block 232448 bytes, shared memory/SM 233472 bytes.
- Ubuntu 24.04 ARM64, R39 revision 2.1, Python 3.12.3, driver 595.78.
- Initial environment lacked CUDA toolkit, Python development headers, g++, PyTorch and profilers. See `inventory.json` for initial capture and `inventory-configured.json` for configured capture.
- Power mode observed: 120W (mode 1), unchanged. Desktop and a competing compute process were observed; performance isolation is not established.

## Installed project-local environment

After the user requested dependency installation, packages were downloaded from configured signed Ubuntu/JetPack r39.2 APT repositories and extracted into `local/toolchain`; nothing was installed into `/usr`, no sudo was used, and the driver was unchanged. The official PyTorch ARM64 CUDA 13.2 wheel and project Python dependencies are in `local/thor-venv`. Exact APT package versions/hashes are recorded in `apt-lock.json`; Python versions in `python-lock.txt`.

Verified versions: nvcc 13.2.86 (JetPack CUDA toolkit 13.2.2 components), GCC/g++ 13.3, torch 2.13.0+cu132, Nsight Systems 2025.6.3, Nsight Compute 2026.1.1, Compute Sanitizer 13.2.87. PyTorch's isolated bundled runtime dependencies are distinct from the toolkit used to compile the extension. BF16 CUDA matmul completed with finite output; PyTorch's compiled architecture list includes sm_110.

[NVIDIA's current PyTorch installation guidance](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/additional-setup/pytorch) identifies JetPack 7+ ARM64 support and the official CUDA 13.2 PyTorch wheel channel. The recorded version was chosen explicitly rather than replacing the JetPack driver stack.

```bash
# From repository root; read-only setup check:
python tools/thor/setup_local.py
# Reproduce missing local packages from exact recorded versions (downloads ~GB):
python tools/thor/setup_local.py --install
source tools/thor/env.sh
python tools/thor/inspect_device.py
make -C demos/low-latency-llama GPU=THOR THOR_ARCH=sm_110a THOR_WORKERS=20
python tools/thor/smoke.py --execute --sanitizer memcheck
python -m unittest discover -s tests -v
```

The Llama extension build command above succeeded and its Python module imported. See `native-build.txt`. It does not establish numerical correctness of the Llama operators or full decode. No model weights have been downloaded.

## Required SM110 adaptations

The explicit THOR target defines `MK_THOR` and selects the pinned TK TMA/SIMT path (`KITTENS_HOPPER`), excluding the unvalidated B200 tensor-memory path. A local TK include overlay fixes two ARM64 unsigned-char to CUDA signed-char2 list initializations and adds missing compiler memory clobbers to semaphore wait/test-wait inline assembly; `prepare_tk.py` asserts the expected source pattern, records source/output hashes, and leaves the pinned submodule untouched.

The initial `-arch=sm_110a` build failed while validating additional generic compute_110 PTX, not while assembling native sm_110a. Treating that failure as proof that SM110a lacks setmaxnreg was incorrect. Explicit `-gencode=arch=compute_110a,code=sm_110a` fixes this, and dynamic register redistribution now compiles and runs on Thor. H100/B200 builds likewise succeed with explicit compute_90a/sm_90a and compute_100a/sm_100a codegen; hardware execution on those GPUs remains NOT RUN. Raw initial failures and corrected build logs are retained.

Thor supports `THOR_REGISTER_MODE=static` (whole-CTA cap, default 96) and `THOR_REGISTER_MODE=dynamic` (separate consumer/service targets). The static build reported zero spill load/store bytes; the first dynamic build reported 20 spill-store bytes, 36 spill-load bytes and a 176-byte stack. These are resource observations, not latency conclusions. NoOp, QKV, whole/partial attention and reduction checks passed with dynamic registers. Other configurations still require validation.

`sm_110a` compiled and executed a simple native CUDA kernel successfully. `nvcc --list-gpu-code` reports base sm_110, not its architecture-specific suffix. SM110 native op support is established through compilation, not just that listing. `THOR_ARCH=sm_110` remains selectable but has not been compiled/tested here.

The checked Llama binding queries actual compiled resources, validates queue/grid sizes and CC, accounts for alignment/static shared memory, and rejects an oversubscribed initial grid. `dry_run=True` returns resource metadata without launching after argument conversion. Native validation of the Python argument path remains to test.

## Tool checks and remaining gates

The model-free VM passed 36 NoOp/stamp cases with lengths 0/1/2/3/17/257 and worker counts 1/3/20. Its initial 16-consumer-warp memcheck and 4-warp synccheck passed. The original warp-aggregated instruction-barrier path reported shared-memory hazards under 8-warp racecheck. Warp synchronization and wait compiler clobbers alone did not eliminate them. Thor now uses per-thread instruction-barrier arrivals; racecheck passes at 4/8/16 consumer warps and memcheck passes at 16. Preserve both failing and successful `smoke-*.json` reports. These results cover model-free NoOp/stamp instructions, not all Llama operation pipelines.

Nsight Systems CUDA tracing completed (`nsys-smoke-check.json`; trace artifact under `local/`). Nsight Compute runs but reports `ERR_NVGPUCTRPERM` (`ncu-smoke-check.json`). Counter access requires an administrator-controlled permission change; no driver settings have been changed. Missing model access and outstanding correctness gates, rather than absent build dependencies, now limit full decode benchmarking.

The synthetic fixture subsequently exercised five registered matvec opcodes on BF16 random tensors and all five passed numerical comparison and racecheck after per-thread matvec ring semaphore signaling was added. See THOR_RESULTS.md for the deliberately limited coverage. This does not establish full Llama correctness.

## Correctness experiments and cooperative launch

Thor checked bindings now require cooperative-launch support, validate compiled occupancy, and use cooperative launch with an entry grid synchronization. `inventory-cooperative.json` records the observed capability. The ordinary H100/B200 launch path is retained. The generated-project template inherits the matching base config and obtains its worker count from the actual device rather than assuming 148 SMs.

The historical BF16 atomic down-projection and fused-float epilogues remain the default numerical configuration for comparison. A separately named experiment uses deterministic FP32 column partials and reference BF16 rounding points:

```bash
source tools/thor/env.sh
make -C demos/low-latency-llama GPU=THOR 'EXTRA_NVCCFLAGS=-DMK_FP32_DOWNPROJ=1 -DMK_MATCH_TORCH_BF16_EPILOGUES=1'
python -m tools.thor.model_check --context 1024 --fp32-downproj --match-torch-bf16-epilogues --correctness-repeats 5 --output local/reference-rounding-check.json
THOR_CONFIG='{"name":"reference-rounding","fp32_downproj":true,"match_torch_bf16_epilogues":true}' python -m tools.thor.teacher_forced --context 1024 --steps 100 --output local/reference-rounding-teacher-forced.json
```

This is a reproducible experiment, not an accepted replacement: seed 2 at context 128 and two steps of the 100-step context-1024 teacher-forced run fail their declared numerical gates. Rebuild after any source/tool edit; source fingerprint validation intentionally rejects stale extensions. Layer-prefix diagnostics are available through `tools.thor.layer_check` with an external `timeout` watchdog.

## Weight-stage layout selection

`configs/thor/weight-stage-{1,2,3}.json` select 1/2/3 input stages with minimum 5/9/13 pages; `small-smem.json` aliases the one-stage footprint. Compile-time overrides are `MK_WEIGHT_PIPELINE_STAGES` and `MK_PAGE_COUNT`. Output staging stays three. For example:

```bash
make -C demos/low-latency-llama GPU=THOR 'EXTRA_NVCCFLAGS=-DMK_WEIGHT_PIPELINE_STAGES=1 -DMK_PAGE_COUNT=5'
THOR_CONFIG='{"name":"weight-stage-1","weight_stages":1,"page_count":5}' python -m tools.thor.synthetic --op qkv --iterations 8 --rotary --output local/stage1-qkv.json
```

The full `stage_matrix` command in BENCHMARKS.md applies matching build/host flags and checks actual resource metadata. The historical BF16 atomic math can fail repeated model gates; the qualified stage comparison uses the same reference-rounding numerical switches in every candidate. Neither layout selection nor that experiment closes the model-wide correctness gate.

The qualified stage, fixed-page, racecheck and memcheck datasets were captured before the first local commit and therefore record the upstream Git SHA plus a dirty worktree. Commit `859ac01` preserves their exact source fingerprint `61f4be3bdb497ebd4ba3dc3d15a75edda184bfc6a80fb54ab781a0335d4bfdee` for reproduction. Later plot-label changes do not alter kernel mathematics. Earlier bring-up failures retain their recorded hashes/logs but do not all have independently recoverable source snapshots.

Decode-oracle choice is explicit through `--reference-attention flash|math` in `model_check`, `--reference-attention` in `teacher_forced`, and `THOR_REFERENCE_ATTENTION` for benchmark adapters. Shared prefill remains FlashAttention. Math mode requires highest FP32 matmul precision and disallows reduced-precision math SDPA. `teacher_forced --native-past` propagates native KV independently rather than restoring reference past at every step. It remains a matched-input state test, not free-running sequence equivalence. Historical optimized-backend failures remain failures.
