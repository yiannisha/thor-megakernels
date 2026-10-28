# O-projection activation reuse on Thor

`o_proj_grouped=true` coalesces the existing O-projection output-block ranges. At 20 workers, each layer has twenty balanced six/seven-tile instructions instead of 128 single-tile instructions. The matvec consumer loads its activation vector once outside the tile loop and retains it in registers. Activation loading uses the existing global/shared/register path; it is not a new TMA activation transfer or precision implementation.

For Llama-3.2-1B, this changes 3,156 tasks / 70,784 dependency edges to 1,428 tasks / 22,400 edges. O-projection instructions fall from 2,048 to 320, and logical activation input bytes from 8,388,608 to 1,310,720 per decode. These counts do not measure physical LPDDR traffic. Task placement, instruction setup and fan-in costs change together, so timing cannot isolate activation traffic alone.

## Safety and progress

All 128 output tiles per layer remain present exactly once. Their original arithmetic and completion-counter increments are unchanged. The host coverage audit checks ranges and all producer dependencies before native ready-queue execution; scheduler tests cover balanced ranges across several worker counts and policies.

The first grouped static implementation passed frozen numerical checks but failed racecheck with eight shared-memory hazards. The reported conflicting accesses were attention's BF16 shared output conversion and a later matvec's weight read after page reuse. Those records remain in `activation-reuse-qualified/`; their timings are excluded from the qualified report.

The safe static configuration sets `handoff_after_opcode=2`: all execution roles join after partial attention through the existing counted non-aligned barrier 6 before advancing. Controller participation is excluded, preserving its prefetch role. The branch uses the current instruction's uniform opcode within each worker. This local instruction join adds no cross-CTA wait or new residency requirement. Full serial handoff also satisfies the grouping guard. Static grouping without either safeguard is rejected. The ready queue already joins after every instruction and rejects a redundant selective-handoff setting.

`handoff_after_opcode=-1` preserves the existing default. Grouping is host-only and changes the configuration cache key without adding a compiler macro; the matched grouped/ungrouped pair uses the same binary. The static comparison includes a separate original baseline to distinguish the attention join from grouping.

## Qualification

See [the measured report](benchmark-activation-reuse.md) for the repeated comparison. All nine context/seed pairs per backend (128/1024/8192, seeds 1/2/3) have bitwise identical logits, final hidden state and appended K/V when grouping is toggled. Both grouped backends pass memcheck, synccheck and racecheck. Static sanitizer evidence is in `activation-reuse-handoff-pilot/`, with its exact-source reference recorded in `activation-reuse-safe/`; ready evidence is directly in the latter directory.

The accumulated-native-cache test remains **FAIL** at steps 10/11/21 for both grouped backends, exactly reproducing the baseline failures. Worst logit NRMSE is 3.0852%, with 100/100 greedy matches. Grouping does not resolve the numerical qualification issue and is not promoted to the default. A separate 100-step baseline test supplying reference past K/V fails only step 44 (position 1068): logit NRMSE 3.7640%, hidden NRMSE 0.9424%, K/V gates passing and all greedy tokens matching. This isolates a per-step mismatch in addition to the accumulated-cache failures; a follow-up 45-step head probe rules against the native head as the main source: at step 44 native versus PyTorch head on identical native hidden state differs by only 0.0112% NRMSE, while the PyTorch head on that hidden state still differs from the full reference by 3.7640%. The decoder hidden-state difference is amplified by the head. Across the probe, native versus same-input PyTorch head stays below 0.0129% NRMSE. Evidence: `activation-reuse-safe/reference-past-temporal.json` and `head-error-probe.json`; reproduce the latter with `python -m tools.experiments.head_error_probe --output <fresh-output.json>`.

## Reproduction

Source `tools/thor/env.sh`. Build with the recorded Makefile command using `configs/thor/attention-handoff.json` for static control and `configs/thor/activation-reuse.json` for grouped execution. Use `configs/thor/ready-queue.json` and `configs/thor/ready-activation-reuse.json` for the ready pair. Set `THOR_CONFIG` to the selected JSON and `THOR_BINARY_DIR` to its matching build directory. The binding enforces source/configuration identity.

Run `python -m tools.experiments.activation_identity --output <fresh-output.json>` for each ungrouped backend configuration. Run `tools.thor.bench` with the native model adapter, five warmups and thirty repeats, in five independently loaded batches per variant. The saved process record gives the exact build, sanitizer, temporal and benchmark commands. The seeded Latin-square order balances position, but not every adjacent ordered pair. Existing unrelated GPU workloads are uncontrolled and remain running.

Generate the report with:

```sh
python -m tools.experiments.activation_report docs/thor/activation-reuse-safe --output docs/thor/benchmark-activation-reuse.md
```

The report requires all five variants, complete identity matrices and matching workload/source metadata. A final quality-matched autoregressive/vLLM comparison remains separate.

The failing reference-cache step also has 64 layer-prefix comparisons against the explicit math attention oracle, each starting from the reference layer input. All pass their gates; worst prefix NRMSE is 0.1355% for partial attention, 0.1349% after O projection, 0.3585% after up/gate and 0.4126% after down projection. This supports investigating propagation across layers rather than attributing the mismatch to one isolated opcode. Evidence: `activation-reuse-safe/step44-math-layer-prefix.json`. The wrapper selects math SDPA for the entire check, highest FP32 matmul precision, and disables reduced-precision math-SDPA reductions.

Final delivery checks: H100/B200 compile successfully (hardware execution pending), the primary Thor static module is rebuilt from current sources and passes five native model smoke checks, and the host suite passes 60 tests with one optional GPU skip. Records are in `activation-reuse-build-refresh/`.
