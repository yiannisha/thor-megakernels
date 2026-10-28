# Experimental weight-only NVFP4 VM

This path preserves BF16 activations and uses BF16 `tcgen05` MMA. Only weights
are quantized to NVFP4. The existing native W4A4 path remains the default;
select weight-only operation with `--weight-only`.

Packed weights and block scales remain resident in device memory. The VM loader
stages both through TMA. Consumer warps decode FP4 pairs and apply block scales
in shared memory; the launcher consumes the resulting BF16 tiles. Two expanded
BF16 buffers permit conversion/MMA overlap. Both physical 64 KiB pages are owned
by the operator, and TMEM is retained for the VM launch. No full BF16 weight copy
is materialized in global memory. Activation preparation retains the existing
RMS/gating operations but skips FP4 activation quantization.

## Current performance

Llama-3.2-1B-Instruct, batch one, context 1024, 100 autoregressive decode tokens:

| Path | ms/token |
| --- | ---: |
| W4A16, global scale reads, single expanded buffer | 93.44 |
| W4A16, TMA scales, single expanded buffer, 16-token pilot | 48.87 |
| W4A16, TMA scales, double expanded buffer, 5 requests | 53.22 |
| Previously measured native BF16 | 10.40 |
| Previously measured native W4A4 | 5.13 |
| Previously measured vLLM BF16 with graphs | 11.37 |

The short pilot and longer run are different protocols. Double buffering has
not demonstrated an improvement. BF16/W4A4/vLLM entries are previous runs,
not contemporaneous controls. W4A16 is currently a substantial slowdown;
reduced off-chip bytes do not compensate for this conversion implementation.
W4A4 and W4A16 have different quality and cannot be compared as equivalent
precision configurations.

## Numerical checks

On one frozen decode, all 65 prepared BF16 activation vectors match the
independent preparation reference exactly. Maximum per-linear output NRMSE
against FP32 multiplication of the same dequantized weights is 0.00032051
(0.0321%), including residual addition. This check holds each actual input fixed
and therefore isolates operator error from accumulated model divergence.

The full-model dequantized FP32 oracle with shared VM attention has logits
NRMSE 2.6767%, hidden 0.9466%, K-cache 0.5321%, V-cache 1.3550%. Outputs are
identical before and after the buffering change. This remaining aggregate
reference difference is not fully explained by the frozen-input checks.
Against original BF16 weights, logits NRMSE is 40.905%; weight-only quantization
still changes model outputs. These checks do not establish quality equivalence.

Full-model compute-sanitizer memcheck, synccheck, and racecheck all pass with
zero errors or warnings. Raw checks and timing are in [w4a16-vm](w4a16-vm/).

## Quality

On the same 1,024 teacher-forced WikiText test tokens used for the W4A4 check
(16 windows, 128-token BF16 prefill, 64 scored decode tokens per window), BF16
conditional perplexity is 18.4948 and W4A16 is 20.4661: **+10.66%**. W4A4
previously measured +20.16%. Top-token agreement is 85.45% for W4A16 versus
78.42% for W4A4. This is a fixed subset with BF16 prefill, not standard
whole-corpus perplexity or evidence of quality equivalence.

The eight authored prompts retain the grammar regression ("go" versus BF16
"She goes.") and change the ice explanation; this is not a quality-equivalent
replacement. The W4A4 regression check still matches its quantized oracle
(logits NRMSE about 9.1e-6).

## Reproduction

```sh
source tools/thor/env.sh
bash experimental/native_fp4_vm/build_model.sh
python -m tools.analysis.nvfp4.model_bench --weight-only --check-only --quantized-reference --shared-attention-reference --output local/w4a16-check.json
python -m tools.analysis.nvfp4.model_trace --weight-only --output local/w4a16-trace.json
python -m tools.analysis.nvfp4.model_bench --weight-only --context 1024 --decode-steps 100 --warmups 2 --repeats 5 --output local/w4a16-e2e.json
python -m tools.analysis.nvfp4.qualify_model --weight-only --output local/w4a16-sanitizers
python -m tools.analysis.nvfp4.model_corpus --weight-only --output local/w4a16-corpus.json
python -m tools.analysis.nvfp4.model_quality --weight-only --output local/w4a16-quality.json
```

The separate `experimental/weight_only_fp4` primitive is an initial unpipelined
control. It is correct but slower than Torch BF16 and is not used by the VM.
