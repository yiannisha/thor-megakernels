# Native π0 on the Jetson Thor VM

Implementation update: see [PI0_STATUS.md](PI0_STATUS.md). Local experimental
Thor code provides measured native TCGen05/TMEM support, and the PiZero demo
now uses it with explicit SM110a code generation and cooperative admission.
The original recommendation below to replace TCGen05 was superseded by that
implementation evidence; the Llama enabling path remains unchanged.

Original planning audit: 2026-10-28. Current results are in PI0_STATUS.md.
At the user's direction, work stops after numerical fixes and final validation;
the later optimization and baseline phases below are deferred. This document
does not change the completed Llama enabling port or historical workspace plan.

## Decision

Use this repository's Thor static VM as the runtime. Transfer the complete π0
model compiler, packing rules, numerical semantics and useful fusions from
`../../../pizero-megakernel/native_pi0/`. Replace its B200 projection pipeline
with Thor-native tensor-core range operators. Keep vision, prefix construction,
all ten expert/Euler steps and output in one cooperative VM launch.

Do not implement π0 as repeated Llama matvec instructions: its hundreds of
vision/prefix rows and 51 expert rows require GEMM and tiled attention. Do not
import the retired DAG executors or use a host loop over model stages.

## Evidence and starting points

Audited revisions:

| Repository | Commit | Relevant state |
|---|---|---|
| Megakernels | `8d73fcb1f8453c5d69055a9c9b58503c517e4e90` | Native Thor runtime; clean worktree before this document |
| pizero-megakernel | `9affe7a466263cd973d1e20112a0f74a9e5fa704` | Full static π0, BF16 model and integrated B200 fusions; clean worktree |

The recorded Thor inventory in `enabling-port/inventory.json` reports 20 SMs,
CC 11.0, 232,448 bytes opt-in shared memory per block and a local CUDA 13.2
toolchain. Refresh this inventory before implementation. These are recorded
device facts, not a new probe in this audit.

The source repository's README and early IMPLEMENTATION.md entries are stale
about parity. Later entries and `results/native_pi0_realtime_fusions.json`
record one launch, action NRMSE 0.00848 and a reported B200 median of 174.4 ms.
That is random-weight evidence from B200, not Thor performance or trained-policy
qualification. Intermediate prefix discrepancies are substantially larger than
the final action discrepancy, so final actions alone are insufficient diagnosis.

Read these sources during implementation:

| Source | Transfer or constraint |
|---|---|
| Thor `include/{megakernel,config,checked_binding}.cuh` | Cooperative residency, static resource checks, synchronization fixes |
| Thor `include/controller/` and `include/util.cuh` | Original static fetch and logical worker ownership |
| Thor `tools/thor/prepare_tk.py`, demo Makefile | Local TK overlay and explicit SM110a build |
| π0 `model.py`, `program.py`, `model.cu` | Complete wiring, static ranges, producer counters and replay epochs |
| π0 `native.cu`, `dense.cu`, `tcgen_probe.cu` | Numerical epilogues to transfer; TCGen05/TMEM pipeline to replace |
| π0 `attention.cu`, `model_attention.cuh`, `norm.cu` | Warp MMA attention and norm algorithms; participation assumptions to audit |
| π0 `scripts/validate_native_pi0.py`, `benchmarks/` | Same-weight oracle, landmarks, workload/timing boundary |
| π0 `FLASHRT_TRANSFER_AUDIT.md`, `FLASHRT_PORTING_MATRIX.md` | Optimization inventory; distinguish π0 from π0.5 |

## Architecture boundaries

Keep reusable VM changes in Megakernels. Add `demos/pi0/` for the native CUDA
entry point and operators, `megakernels/pi0/` for model compilation and packing,
and `tools/thor/pi0/` for validation, profiling and benchmark reports. These are
proposed paths. Initially preserve the B200 implementation as the comparison
source; extract shared model descriptions only when both backends are stable.
Pin source revisions and retain attribution when copying implementations.

The Python compiler owns weights, layout transforms, tensors, masks, rotary
tables, static instructions and scratch allocation. Forward supplies root input
pointers and invokes the checked VM launcher on the caller's CUDA stream. All
model computation remains in native operators, without runtime allocations or
per-stage host dispatch. Fixed-shape programs are cached by shape/configuration.

Use `thor_config` as the base for a model-specific `Pi0ThorConfig`, retaining
cooperative launch and per-thread semaphore participation. First attempt four
consumer warps plus four service warps (256 threads): this matches the π0
operators' existing participation and avoids running their 128-thread loops on
512 consumers. This is a candidate configuration, not an already validated Thor
mode. Audit every VM role, page-release count and named barrier before using it.
Keep the existing 640-thread Llama configuration unchanged.

The current Thor 96-register cap was chosen for 640 threads. Recalculate the
register/resource budget for 256 threads rather than copying either that cap or
B200's 200-register target. All config-derived thread/shared-memory values must
be recomputed explicitly. Retain thirteen 16-KiB pages initially to avoid mixing
allocator changes with model bring-up; reduce pages only after qualification.

The Thor path currently compiles with `KITTENS_HOPPER` and no
`KITTENS_BLACKWELL`. Do not enable the latter to make TCGen05 code compile.
Prototype warp-level BF16 `mma.sync` with FP32 register accumulators first;
existing attention provides an example of that style. Compile/run a separate
WGMMA probe through the pinned compatibility path before considering WGMMA.
Its presence in TK source is not proof that a particular Thor instruction works.
Neither choice uses TMEM. Remove tensor allocation/waits from projection hooks;
inactive launcher roles use the VM NoOp contract, with consumer-driven MMA.

## Transfer matrix

| Component | Initial Thor treatment | Later work |
|---|---|---|
| Full model wiring and ten-step integration | Reuse exact π0 semantics | Buffer-lifetime reuse and reduced instruction traffic |
| Static range streams / data counters | Recompile for actual resident workers | Thor-measured costs and row-band pipelining |
| TCGen05 BF16 GEMMs, 64-row variant | Replace with register-accumulator tensor-core GEMMs | Tune 32/64/128-row and 64/128-column candidates |
| Vision HD72 / Gemma HD256 attention | Port warp-MMA algorithms and rounding boundaries | Query/key tiles, GQA KV reuse and TMA staging |
| Residual, GELU, GeGLU epilogues | Preserve BF16 projection/activation rounding | Integrate individually with full-model ablations |
| Offline action/time fold, FP32 heads | Reuse fold and FP32 semantics | Tune small heads only if material in profile |
| Model-derived RoPE tables | Reuse actual model tables | Keep setup precision identical to oracle |
| Last prefix layer K/V-only | Reuse dead-output pruning | Verify cache dependencies and prompt offsets |
| Compact RMS factors / folded RMS weights | Port behind an explicit option | Enable after fold/rounding qualification |
| QKV/RoPE/cache epilogue | Start with a separate native RoPE/cache instruction | Replace unsafe cross-worker rendezvous with owned Q/K pairs |
| FP8 / NVFP4 | Exclude from initial BF16 path | Probe native instruction support and calibrate separately |
| π0.5 conditioning / AdaRMSNorm | Separate model variant | Only after ordinary π0 is qualified |

## Scheduling and synchronization requirements

1. Use logical CTA IDs and device-derived worker capacity, initially at most one
   resident CTA per physical SM. Use a cooperative launch with entry grid sync;
   do not copy `model.cu`'s ordinary triple-chevron launch. Adapt the checked
   binding to the new globals and include compiled resource/build fingerprints.
2. Preserve monotonically ordered model stages within every worker stream and
   exact tile coverage. Reject missing, duplicated, out-of-range and unsupported
   instructions before launch. Recompute producer expectations for every tile
   geometry; the source hardcodes 128-row counter bands and 64-query attention
   tasks in several places.
3. Residency alone does not prove progress. Validate a wait-for graph including
   worker stream order and intra-instruction waits. No instruction may block
   waiting for work later in its own stream or a cycle of blocked workers.
4. The B200 QKV fusion explicitly requires `tasks <= workers` and rendezvouses
   after raw tile publication. On Thor many projections exceed 20 tiles. Start
   with the existing unfused fallback. Later assign both rotary halves of a Q/K
   head to one instruction/worker, or finish all projection work before a
   separate RoPE range. Never relax the guard without a progress proof.
5. Producer completion must follow all writer stores, appropriate async proxy
   fences and consumer-group synchronization. Consumers acquire the precise
   row/head counters for data they use. Keep page lifetime distinct from output
   visibility, and audit phase reuse across tile ranges and successive opcodes.
6. Reuse epoch counters only with serialized calls per model context on one
   stream. Explicitly reject concurrent use or provide independent contexts.
   Test changed images, state, tokens and noise, not just identical replay.
   Define overflow/reset behavior outside the timed forward. The source rejects
   CUDA graph capture; keep that explicit initially rather than capturing a
   host-incremented epoch value incorrectly.

## Implementation sequence and acceptance gates

### P0 — Reproducible workload and build (first change)

Refresh device/toolchain inventory; record both SHAs, TK revision/overlay,
compiler flags and Python versions. Reuse the existing local Thor environment;
create an isolated π0 environment if the pinned OpenPI Transformers overlay
conflicts with Llama. Do not run the B200 setup script unchanged or replace the
JetPack stack. Audit available checkpoints and memory before loading models.

Add an explicit Thor build target and backend parameter; remove hardcoded
`sm_100a`, `workers=148`, third-party runtime include paths and the shared
`/tmp/pi0_native_model.so` output. Cache binaries by source/config/toolchain hash.
Make CUPTI optional and use local library paths. Add worker selection to the
validator, which currently constructs NativePi0 with its B200 default.

Gate: host program/packing tests, capability report and a checked cooperative
NoOp/range smoke with four consumers. Existing Llama host checks/build still
pass. New configs report actual registers, spills, static/dynamic SMEM and
admitted worker capacity.

### P1 — Thor BF16 compute pipelines

Implement a TMA/shared-memory double-buffered range GEMM with BF16 operands and
FP32 register accumulation. Start with 64-row tiles for the 51-row expert and
64-column tiles if accumulator pressure requires it. Mask tails and pad only
as required by tensor instructions; benchmark 128-row vision/prefix alternatives
later. Keep instruction encoding independent of physical tile dimensions.

Port attention/norm/pointwise hooks with exactly one four-warp consumer group,
explicit barrier IDs and page ownership. Retain FP32 patch/head operations in
the first correct model even if their initial SIMT implementation is slow.

Gate: representative actual model shapes, partial rows, HD72 padding, HD256,
GQA/cache masking, rotary pairs, biased projections and residual rounding pass
the numerical oracle. Run memcheck and synccheck on mixed range sequences,
including empty worker streams and ranges longer than the pipeline depth.
Keep this operator phase short: integrate the full model immediately afterward.

### P2 — Complete one-launch model

Transfer 27 vision layers, 18 prefix layers and 18 expert layers per Euler step,
state/token assembly, cache construction, FP32 heads and all ten updates.
Keep data on-device, build prefix once, reuse its cache across steps, and prune
the final prefix layer to K/V. Use separate native RoPE/cache instructions until
the ownership-safe fusion is ready. Start without algebraic RMS folding if it
obscures diagnosis; retain selectable fused/unfused references.

Gate: full three-view, 224×224, chunk-50, ten-step model; zero and 16-token
prompts; smaller diagnostic shapes; multiple seeds; identical replay and changed
inputs; one traced VM launch. Require finite outputs, correct shapes and the
existing action smoke gates: NRMSE <= 0.015625, normalized max error <= 0.0625,
cosine >= 0.999. Record layer landmarks and investigate new divergence even
when the final action passes. Full-shape memcheck and synccheck must pass;
targeted racecheck and repeated replay stress address publication/page reuse.
Random-weight success precedes same-weight trained-checkpoint evaluation with
real normalized observations/tokens; neither proves robot task success alone.

### P3 — Matched Thor baseline and bottleneck attribution

Run native, OpenPI eager, compiled without implicit CUDA graphs, explicit graph
and supported realtime-vla variants on this Thor with shared weights/inputs.
Record unsupported baselines explicitly. Use the existing timing boundary:
GPU-resident normalized inputs/noise through complete actions, including vision,
prefix and ten steps; exclude setup, packing, capture and transfers.

Measure isolated subprocesses, warmups, raw samples, event and wall latency,
median/p90/p99, memory footprint and current power/thermal/clocks. Keep the
existing power mode. Avoid reporting B200 ratios or Llama decode ratios as a
Thor π0 prediction. Report at least three independent timed runs for decisions.

Use Nsight Systems for launch/idle behavior, selective VM timing for pipeline
and dependency waits, and Nsight Compute where available for tensor utilization,
register spills, L2/DRAM traffic and stalls. Measure instrumentation overhead.
Attribute cost to vision, prefix, repeated expert, heads and instruction/wait
overhead; component-launch diagnostics are not the accepted inference path.

Gate: numerically qualified full-model baseline and a measured ranked cost
table. Set latency targets from this evidence rather than guessing a deadline
or a speedup from the B200 results.

### P4 — Optimize in measured order

Default experiment order, revised if P3 identifies a different dominant cost:

1. Port residual, vision bias/GELU and interleaved gate/up GeGLU epilogues;
   preserve BF16 rounding boundaries. Fold action/time projections at setup and
   fuse SiLU/Euler where semantics permit. Measure each fusion separately.
2. Tune GEMM shapes, K blocking, register cap, range length and two/three-stage
   staging by vision/prefix/expert shape family. Favor useful work per byte and
   avoid padding 51 expert rows to 128. Compare warp MMA with WGMMA only after
   the latter passes the native feature/accuracy probe.
3. Replace `count*K` static costs with measured Thor operator costs. Balance
   bounded tile ranges across 20 workers; avoid long ranges monopolizing
   critical producers. Narrow whole-tensor waits to required row/head bands.
   Validate the expanded wait-for graph for every scheduling change.
4. Implement ownership-safe QKV/RoPE/cache fusion; exploit GQA KV reuse and tune
   attention query/key blocks. Keep the eager-compatible attention path as the
   oracle; faster softmax/rounding paths require full-model qualification.
5. Reduce scratch/live buffers and padded traffic, eliminate unused timing
   storage when disabled, and shrink page count only with resource evidence.
   Prefer reuse across the ten expert passes where caches permit; quantify
   actual bandwidth/L2 effects rather than assuming weights remain resident.
6. Evaluate calibrated FP8 only at proven native-supported sites. Include
   scale computation/conversion in timing and test trained observations across
   diffusion timesteps. NVFP4 requires separate hardware/backend evidence;
   retain BF16 for sensitive sites. The B200 TCGen05/NVFP4 code is not a Thor
   implementation merely because the GPU is called Blackwell.

Every experiment records parent/config/SHA, action and landmark errors, latency
distribution, resource usage and full-forward ablation. Keep only repeatable
end-to-end improvements. Multi-CTA residency and segmented graphs are later
optional experiments with separate acceptance; they are not prerequisites for
the requested one-launch native π0.

## First implementation slice

Deliver P0 and the smallest P1 vertical slice together: a new π0 Thor config,
checked cooperative binding, logical static range streams and BF16 projection
for a real expert shape, followed by residual/GeGLU-compatible output layout.
Validate mixed dependent ranges and repeat calls. Then wire the complete BF16
model before broader tuning. The highest initial uncertainty is viable
register-accumulator GEMM performance/resources on this VM, followed by safe
20-worker scheduling and attention numerical behavior.
