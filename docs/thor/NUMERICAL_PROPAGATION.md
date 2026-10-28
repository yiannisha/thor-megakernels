# Decode numerical propagation diagnostics

These experiments investigate the unresolved acceptance gates. They do not change error tolerances or establish generation quality. The control uses compensated BF16 attention, FP32 down-projection partial accumulation and matching BF16 normalization/epilogue rounding. All use local Llama-3.2-1B-Instruct, context 1024, seed 1 and teacher-forced greedy reference tokens.

| Independent-cache diagnostic | Failed steps (any gate) | Worst logit NRMSE |
|---|---|---:|
| Native control | [10, 11, 21] | 3.0852% |
| No FMA contraction | [10, 21, 33, 44] | 5.3347% |
| No FMA + precise math | [10, 21, 32, 44, 55, 88] | 4.4565% |
| PyTorch attention substitution | [9, 10, 20, 21, 32, 43, 44, 88] | 4.3944% |
| PyTorch QKV/RoPE substitution | [10, 21] | 2.6925% |
| PyTorch up/gate/SiLU substitution | [9, 10, 21, 44, 64, 88] | 5.5230% |
| PyTorch down/residual substitution | [10, 11, 21, 32, 44] | 7.1015% |

All rows retain the original error gates and match all 100 greedy choices. Substitutions are diagnostic Python/native hybrids, not native implementations or performance candidates.

## Reference-cache failure

The control with independent native K/V histories fails steps 10/11/21 out of 100. Supplying reference past K/V instead leaves one failure at step 44: logit NRMSE 3.7640%, hidden NRMSE 0.9424%, K/V gates passing and all greedy choices matching. This shows both per-step arithmetic and accumulated-cache effects matter.

At step 44, applying the PyTorch output head to the native hidden state retains the 3.7640% logit mismatch. Native versus PyTorch head on the identical hidden state differs by only 0.0112%; over the 45-step diagnostic the worst same-input head error is below 0.0129%. The head arithmetic is therefore not the main source of this observed error.

All 64 native layer-prefix comparisons starting from reference layer inputs at step 44 pass the gates. Worst NRMSE by endpoint: partial attention 0.1355%, O projection 0.1349%, up/gate 0.3585%, down projection 0.4126%.

A separate cumulative-prefix probe starts each prefix from the original embedding and reference past K/V. Its last hidden state is **bitwise identical** to full-model native execution. Hidden NRMSE by layer (percent):

| Layer | NRMSE | Layer | NRMSE |
|---:|---:|---:|---:|
| 0 | 0.0383 | 8 | 1.0949 |
| 1 | 0.4610 | 9 | 0.9740 |
| 2 | 0.8079 | 10 | 0.9463 |
| 3 | 1.0555 | 11 | 0.9571 |
| 4 | 1.2589 | 12 | 1.0372 |
| 5 | 1.2945 | 13 | 1.0256 |
| 6 | 1.2995 | 14 | 1.2373 |
| 7 | 1.1720 | 15 | 0.9424 |

This supports tracing propagation through multiple layers. It does not prove every opcode is free of implementation error or that BF16 rounding explains every mismatch.

## FMA contraction diagnostic

A separate native module built with the same configuration plus `--fmad=false` passes all 100 reference-cache steps (worst logit NRMSE 2.5237%). Its independent-native-cache run still fails steps 10/21/33/44, worst logit NRMSE 5.3347%, with all 100 greedy choices matching. Disabling contraction is therefore not a qualified fix and is not promoted. This is a compiler-flag diagnostic, not a new named runtime configuration or a performance result. The main native module remains the control build.

The additional `--prec-div=true --prec-sqrt=true --ftz=false` build also passes all reference-cache steps (same worst logit NRMSE), but fails native-cache steps 10/21/32/44/55/88, worst logit NRMSE 4.4565%. Precise division/square root therefore do not close the native-cache gate. Commands and results are in `precise-math-probe/`.

Disabling PyTorch BF16 reduced-precision GEMM reductions reproduces the control's exact failed steps and worst logit NRMSE (10/11/21, 3.0852%). That backend setting does not explain this control failure. The wrapper's explicit flag and command are recorded in `reference-accumulation-probe/processes.json`.

## Attention substitution

A diagnostic decoder splits execution after QKV and at layer boundaries. Its all-native control reproduces every numerical metric of the original 100-step native-cache control exactly, including failures at 10/11/21. Replacing only partial attention with PyTorch math SDPA on the native Q/K/V fails steps 9/10/20/21/32/43/44/88, worst logit NRMSE 4.3944%, with 100/100 greedy matches. That substitution does not resolve accumulated decode drift. This is an attribution experiment, not a proposed runtime or a latency comparison. It does not rule out attention improvements, but does not support treating attention probability accumulation as the sole remaining cause. Raw records: `attention-oracle-probe/{control,oracle}.json`.

The up/gate/SiLU substitution has its own split native control, which also reproduces all 100 original metrics exactly. Replacing this whole opcode with PyTorch normalization, projections and SiLU fails steps 9/10/21/44/64/88, worst logit NRMSE 5.5230%, with 100/100 greedy matches. It does not support treating the approximate SiLU exponential alone as a sufficient explanation. Raw records: `upgate-oracle-probe/{control,oracle}.json`.

The QKV/RoPE substitution improves this cohort: all 100 logit checks pass (worst logit NRMSE 2.6925%). Hidden-state checks still fail at steps 10/21, NRMSE 2.3632%/2.0992%; all K/V checks and greedy choices pass. The native split control again reproduces every original metric exactly. This makes QKV/RoPE arithmetic worth further native investigation, but the PyTorch substitution is only an oracle diagnostic and does not qualify a native fix. Raw records: `qkv-oracle-probe/{control,oracle}.json`.

Replacing down projection and its residual update with the full PyTorch BF16 linear result also fails (steps 10/11/21/32/44, worst logit NRMSE 7.1015%, all greedy choices matching). Its split native control reproduces all original metrics exactly. This does not support promoting the existing four-part native reduction to a different summation order without full temporal qualification. Raw records: `down-oracle-probe/{control,oracle}.json`.

## Targeted native RoPE experiment

A native source experiment applies explicit `__fmul_rn`/`__fadd_rn` to the two RoPE products and their sum only when matching BF16 epilogues are enabled. Unlike the global no-contraction build, it does not clear the reference-cache step-44 failure (same 3.7640% worst logit NRMSE). Native-cache checks regress to nine failed steps (5/10/20/21/31/32/44/66/88), worst logit NRMSE 4.9778%, with all greedy choices matching. The source change is **reverted**. The exact patch, compiler invocation and both test outputs remain in `nonfused-rope-probe/`. Applying `source.json`'s patch to base commit `1bb8d61` reproduces the candidate source; it is not part of the selected native implementation.

These results do not identify a sufficient native accuracy fix. In particular, improved isolated/reference-cache checks cannot substitute for independent-native-cache qualification.

## Evidence and reproduction

- `activation-reuse-safe/reference-past-temporal.json`: 100-step reference-cache control.
- `activation-reuse-safe/head-error-probe.json`: same-hidden head attribution.
- `activation-reuse-safe/step44-math-layer-prefix.json`: reference-input layer prefixes under explicit math SDPA.
- `activation-reuse-safe/step44-layer-propagation.json`: cumulative native prefixes and full-kernel identity.
- `no-fma-probe/processes.json`: exact compiler invocation and both temporal commands; failed checks remain failures.

Source `tools/thor/env.sh`, set `THOR_CONFIG` to `configs/thor/compensated-attention.json` contents and `THOR_BINARY_DIR` to the matching extension. Reproduce head attribution with `python -m tools.experiments.head_error_probe --output <fresh-file.json>`, and cumulative prefixes with `python -m tools.experiments.layer_propagation --step 44 --output <fresh-file.json>`. The latter selects math SDPA explicitly throughout reference generation. These are diagnostic harnesses without latency measurements.

## Further separation of QKV arithmetic

The RoPE-only oracle executes native QKV with identity rotary tables, then applies the reference eager rotary operation to the native BF16 Q/K. It reproduces every metric of the rejected explicit nonfused native RoPE candidate over all 100 steps. That agreement validates this separation method for the measured trajectory; it does not provide an accuracy improvement.

A temporary sixteen-consumer RMS implementation follows the installed PyTorch FP32 mean's 512-thread/four-adjacent-values reduction tree. Output scratch is reused only before matvec outputs exist, with consumer barriers protecting publication and all readers before reuse. The native-cache test reproduces every original metric exactly; the reference-cache test changes metrics at steps 27/62 without changing any gate outcomes. Failed steps and worst logit NRMSE remain unchanged in both tests. The extra synchronization has no demonstrated qualification benefit, so this candidate is reverted. Exact patch and both test records are in `torch-rms-tree-probe/`.
