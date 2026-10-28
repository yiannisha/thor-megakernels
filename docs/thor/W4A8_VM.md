# Experimental mixed MXFP4 / MXFP8 VM

This backend uses `tcgen05.mma.kind::mxf8f6f4` with E2M1 weights,
E4M3 activations, FP32 accumulation, and block-32 UE8M0 scales. It does not
reuse the NVFP4 block-16 UE4M3 weight format: weights are independently
quantized once at setup. Consequently quality changes include both the new
weight format and the activation precision.

NVIDIA's PTX ISA documents the mixed types, scale requirements and TMA padding:
https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-mma

The existing VM loader stages packed FP4 weights, FP8 activations and scales
using TMA. TMA supplies the padding required by the K32 mixed instruction.
The launcher copies scale factors to TMEM and issues mixed MMA directly.
No software weight dequantization to BF16 is used. Two 64 KiB shared pages
form a single operator-owned pool for two pipeline stages. Consumer warps
publish BF16 outputs and optional residuals through the existing protocol.
RMS, gating, RoPE, attention and residual boundaries retain BF16/FP32 arithmetic;
only prepared linear inputs are quantized to FP8.

This is a reusable precision backend, exercised with the Llama integration.
It is not a Thor-exclusive instruction: B200 also supports mixed narrow MMA.
Any Thor benefit must be demonstrated by measured reduction in memory costs
and complete decode latency, with quality measured separately.

Build and check:

```sh
source tools/thor/env.sh
bash experimental/native_fp4_vm/build_model.sh
python -m tools.analysis.nvfp4.model_bench --mixed-fp8 --check-only --quantized-reference --shared-attention-reference --output local/w4a8-check.json
python -m tools.analysis.nvfp4.model_trace --mixed-fp8 --output local/w4a8-trace.json
python -m tools.analysis.nvfp4.model_bench --mixed-fp8 --context 1024 --decode-steps 100 --warmups 2 --repeats 5 --output local/w4a8-e2e.json
python -m tools.analysis.nvfp4.qualify_model --mixed-fp8 --output local/w4a8-sanitizers
python -m tools.analysis.nvfp4.model_corpus --mixed-fp8 --output local/w4a8-corpus.json
```

Omitting `--mixed-fp8` retains W4A4; `--weight-only` selects W4A16. The two
flags cannot be combined.

## Measured results

Llama-3.2-1B-Instruct, batch one, context 1024, 100 autoregressive decode
steps, two warmup requests and five measured requests. All compared prompts
have the same recorded input hash. Packing/prefill are excluded from decode
latency; native timing includes embedding, VM execution, barriers and argmax.
vLLM uses engine first/last-token timestamps, including engine scheduling.

| Backend | Median ms/token |
| --- | ---: |
| Mixed MXFP4 / MXFP8, initial run | 5.0404 |
| Mixed MXFP4 / MXFP8, repeat after controls | 5.0397 |
| Native BF16, fresh control | 10.9376 |
| Native W4A4, fresh control | 5.6837 |
| vLLM BF16 with graphs, fresh control | 11.9377 |
| W4A16, earlier run | 53.2178 |

The mixed path is 2.17x faster than native BF16 and 2.37x faster than vLLM
BF16 in these runs. These are different precision configurations, not a
precision-matched framework speedup. No clocks, power modes, or other GPU
processes were changed. The small request count and uncontrolled existing
workload limit performance comparisons.

vLLM uses the previous working configuration: FlashInfer autotuning disabled,
CUDA graphs enabled. The attempted autotuning-enabled startup failed with
an illegal-memory-access error; that failure is archived separately.

## Correctness and quality

With identical quantized formats and VM attention, the independent FP32 linear
reference produces exactly the same BF16 logits, hidden state and KV cache
on both seed 1/context 1024 and seed 2/context 128. Frozen-input checks cover
all 65 linear operators: all prepared activation vectors match exactly;
maximum linear-output NRMSE is 0.00002548 (0.00255%). Full-model memcheck,
synccheck and racecheck pass with zero errors/hazards. W4A16's reference check
is unchanged. W4A4 remains selectable, was benchmarked with the same binary, and retains
its reference logits NRMSE of 9.1e-6.

Against original BF16, the first decode has logits NRMSE 19.04%. On the same
1,024 WikiText teacher-forced tokens as the other precision paths (16 windows,
128-token BF16 prefill and 64 scored tokens per window), conditional perplexity
is 21.4314 versus BF16 18.4948: **+15.88%**. Top-token agreement is 80.27%.
Previously W4A4 measured +20.16% perplexity and W4A16 +10.66%. This subset is
not standard whole-corpus perplexity and does not establish quality equivalence.

The eight authored sanity prompts select the correct grammar word, but the
ice explanation still changes and contains questionable claims. The tiny
prompt set is not an accuracy benchmark. Raw measurements and checks are
in [w4a8-vm](w4a8-vm/).
