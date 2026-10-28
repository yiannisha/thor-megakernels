# OpenPI integration scope and operator gap

Full OpenPI integration: **BLOCKED**. No robotics checkpoint is present under this project's `local/models`; only the user-supplied Llama checkpoint is installed. OpenPI/JAX/Flax are absent in the Thor environment. The pinned package requirements include Torch 2.7.1, Transformers 4.53.2 and JAX CUDA 12; the working Thor stack uses a separate Torch 2.13/CUDA 13 environment. Compatibility requires an isolated, explicitly validated setup. The [source inventory](openpi-source-inventory.json) records exact file hashes for commit `215abfb217dbac7d5f1273282331b9b1866c0479`. No robotics weights were downloaded. [Pinned dependency contract](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/pyproject.toml).

## Chosen subgraph and frozen shapes

The first integration target is one action-expert transformer block inside a denoise step, with prefix K/V supplied from the conditioning pass. Use batch 1 and action horizon 50, padded action dimension 32. For pi0, the continuous state contributes one suffix token (51 total); pi0.5 puts state in the language prefix and conditions expert norms on time (50 suffix tokens). Default prompt limits are 48/pi0 and 200/pi0.5. Images, masks and prefix length must be fixed in the checkpoint-specific run manifest; a synthetic prefix length must not be described as a deployed policy. [Configuration](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py).

The default action expert has width 1024, MLP width 4096, 18 layers, eight Q heads, one KV head and head dimension 256. The PaliGemma language backbone has width 2048 and MLP width 16384. These differ from the current Llama interpreter's width 2048, MLP width 8192, 32/8 Q/KV heads and head dimension 64. Gemma uses RMS epsilon 1e-6 with a learned 1+scale and a GELU-gated MLP; the current Llama kernels use SiLU and a different RMS contract. [Pinned Gemma implementation](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py).

## Proposed block interface

Inputs: BF16 hidden suffix `[B,S,1024]`; per-layer prefix K/V `[B,1,P,256]`; suffix validity and block-attention mask `[B,S,P+S]`; integer positions `[B,S]`; norm/projection weights; optional timestep conditioning and adaptive norm modulation. Outputs: BF16 hidden suffix with the same shape and optional suffix K/V for diagnostics. Prefix K/V are immutable during a denoise step. The manifest must record checkpoint SHA, source commit, norm format, RoPE convention, head scaling, camera slots, image transform, prompt tokens/masks, action normalization, horizon, dtype and timestep schedule.

Dense suffix attention is required: actions within their block attend one another, plus eligible prefix tokens. A single-query autoregressive decode instruction cannot reproduce this mask or compute the full suffix. OpenPI's PyTorch path defaults to ten flow steps and projects the final suffix into an FP32 action-velocity tensor. Block latency must be reported separately from conditioning, all flow steps, action integration and total observation-to-action latency. [Denoise implementation](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py).

| Operation | Current native coverage | Required extension / evidence |
|---|---|---|
| Q/K/V and output projections | Llama fixed-shape GEMV | Multi-token GEMM, width/head specialization, per-layer checkpoint layout |
| Attention and RoPE | Single-query Llama GQA, head 64 | Masked multi-query attention, head 256, immutable prefix plus suffix KV |
| Normalization | Llama RMS and optional sum precision | Gemma offset scale, epsilon, adaptive scale/shift/gate for pi0.5 |
| MLP | SiLU-gated vector projections | GELU tanh, GEMM, 1024/4096 dimensions |
| Residual | BF16 residual operations | Match adaptive residual gating and exact rounding contract |
| Vision, timestep, action head, flow integration | Absent | Independent preprocessing/conditioning and FP32 action reference |

Do not assign speculative opcodes to the gaps. Introduce a reusable dense-tile projection descriptor only after matching weights, strides and accumulation against the pinned reference. Treat arbitrary masking, adaptive normalization and flow integration as explicit operator contracts.

## Focused microbenchmark

`tools/robotics/mlp_probe.py` exercises a synthetic `[1,50,1024]` RMS/GELU-gated MLP/residual with width 4096, repeated ten times with shared random weights. This isolates a reusable GEMM-heavy component; it omits attention and adaptive normalization. It is neither a full transformer block nor checkpoint/action parity. The PyTorch activation setting is `gelu_pytorch_tanh`. [PyTorch expert configuration](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/gemma_pytorch.py).

```bash
source tools/thor/env.sh
THOR_ROBOTICS_MODE=eager python -m tools.thor.bench --adapter tools.robotics.mlp_probe:make_adapter --output docs/thor/robotics-mlp-eager.json
THOR_ROBOTICS_MODE=graph python -m tools.thor.bench --adapter tools.robotics.mlp_probe:make_adapter --output docs/thor/robotics-mlp-graph.json
```

Each timed invocation starts from the same immutable input. Reference linear arithmetic is FP32 with matched BF16 intermediate rounding; gates are finite output, NRMSE <=2%, and pointwise atol 0.1 / rtol 0.05. Metadata includes synthetic tensor hash, probe-source hash, pinned OpenPI commit, shape, shared-weight repetition count and exclusions. Five warmups and three batches of thirty samples use the existing CUDA-event runner. Measurements must wait for the other project GPU tests, and retain the existing external-load disclosure. No native runtime speedup or policy quality claim follows from this probe.

## Checkpoint-level acceptance

Once a checkpoint is supplied, freeze camera/text/state transforms and tensors, export the chosen block's inputs/weights/reference intermediates, and compare projections, norms, attention probabilities, residuals and final hidden output before timing. Then verify one denoise step, ten-step action tensors and action normalization against the pinned OpenPI reference. Compare eager/graph baselines at identical precision and conditioning; add TensorRT/Triton only when their exact contract is validated. Record block, denoise and full observation-to-action latency separately. Action-tensor agreement does not establish closed-loop robustness; simulation or robot evaluation remains a separate gate.
