# Native FP4 VM and model integration (W4A4)

This path remains experimental and selectable. It uses packed NVFP4 weights and
FP4 activations, not weight-only quantization. The enabling-only BF16 model entry
point remains available. W4A16 will be implemented alongside this path.

On Thor, batch one, context 1024, 100 autoregressive decode steps, 5 warmup
requests and 3 batches of 10 timed requests:

| Engine | Linear precision | Median ms/token |
|---|---|---:|
| Native enabling-only | BF16 | 10.3952 |
| Native FP4 VM | W4A4 | 5.1297 |
| vLLM | BF16 | 11.3656 |

Prefill, loading and packing are excluded; embedding, decode, selection and final
output-ID copy are included. Clocks and unrelated system load are uncontrolled.
This compares different precision/quality; it is not a pure framework speedup.
Raw records are in `native-fp4-vm/`.

Correctness checks distinguish execution from quantization quality:

- All 65 activation conversions match bit-for-bit on frozen inputs.
- Linear operations agree with independently decoded FP32 products within
  0.0055% relative RMS error on frozen inputs.
- The unchanged attention path differs from FlashAttention by up to 0.20% on
  identical inputs. FP4 rounding boundaries amplify those differences, producing
  an approximately 11% full-model difference with that oracle.
- With the same unchanged attention backend and independently dequantized FP32
  linear products, full-model logit error is 0.00091%, hidden-state error 0.00192%,
  K-cache error zero and V-cache error 0.000109%. See
  `native-fp4-vm/model-shared-attention-reference.json`.
- Full-model memcheck, synccheck and racecheck passed before diagnostic tracing
  was added; subsequent diagnostic module changes require renewed qualification.

Quality is not equivalent to BF16. On the deterministic WikiText-2 test subset
of 16 windows (128-token BF16 prefixes and 64 teacher-forced decode targets each,
1,024 scored tokens), conditional perplexity rises from 18.49 to 22.22 (+20.2%).
This is not the standard full-corpus perplexity protocol. Both paths use BF16
prefill and separate decode caches. Dataset: [Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
`wikitext-2-raw-v1`, test; downloaded rows checksum is in the raw result.

Eight authored sanity prompts produce five identical short answers, two changed
explanations and a grammar regression (BF16: "She goes."; W4A4: "go"). This tiny
set does not establish general accuracy.

Implementation: optional VM-lifetime TMEM ownership; real VM loader/launcher/
consumer/storer loops; 128-row native FP4 MMA; packed weight/scales staging;
BF16/FP32 attention, RMS, RoPE and residual arithmetic. The generic fused-output
quantization path is preserved, but the full model currently uses explicit
activation preparation around its nonlinear operations.

Build and run:

```sh
source tools/thor/env.sh
bash experimental/native_fp4_vm/build_model.sh
python -m tools.analysis.nvfp4.model_bench --check-only --quantized-reference --shared-attention-reference --output local/fp4-check.json
python -m tools.analysis.nvfp4.model_bench --context 1024 --decode-steps 100 --warmups 5 --repeats 10 --batches 3 --output local/fp4-e2e.json
```
