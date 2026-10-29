# PiZero Thor backend provenance

Initial model and native operators copied from the sibling pizero-megakernel
repository at `9affe7a466263cd973d1e20112a0f74a9e5fa704` on 2026-10-28.
The static VM is supplied by this Megakernels repository, not its B200 checkout.
Attention retains its original FlashRT attribution and Apache-2.0 notice.
OpenPI is fetched separately at `215abfb217dbac7d5f1273282331b9b1866c0479`;
its files and Transformers overlay retain upstream notices.

Thor uses explicit `compute_110a` code generation and cooperative launch.
The opt-in `KITTENS_BLACKWELL` compilation enables native tensor-core/TMEM
operators only in this demo; the Llama build remains unchanged.

Vision attention uses FlashAttention's device arithmetic at
`6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15`, the submodule pinned by the installed
PyTorch `cf30153c4c131c8164ee7798e5022d810682e2cb`. The checkout and BSD license
remain under `local/pi0/flash-attention`. A generated include overlay replaces
CTA-wide synchronization with the VM's 128-thread consumer barrier and uses
VM-owned shared pages; it does not launch the standalone FlashAttention kernel.
Its selected traits are the oracle's HD96/M128/N64/four-warp BF16 traits.
CUTLASS headers are at `91843b71d90737e0a9b86f7d9df213cff5a3526f`.

LayerNorm statistics follow the pinned PyTorch vectorized Welford order.
Arithmetic references:

- [PyTorch LayerNorm](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/layer_norm_kernel.cu)
- [PyTorch FlashAttention parameter scaling](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp)
- [Pinned FlashAttention softmax](https://github.com/Dao-AILab/flash-attention/blob/6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15/csrc/flash_attn/src/softmax.h)
