# W4A16 decoded-weight cache experiment

L2 caches decoded weight data, not unpacking instructions. This experiment
unpacks NVFP4 once into an unscaled BF16 buffer in device memory, then reads
that buffer using the same BF16 shared-memory layout, staging loop, and
`tcgen05` MMA as the original standalone W4A16 control. The tensor-wide scale
is applied at the output in both cases. Outputs are bit-identical.

Thor reports 32 MiB L2, a 24 MiB maximum persisting allocation, and a 128 MiB
maximum access-policy window. Persistence is a retention preference, not a
pinning guarantee. The CUDA graph's kernel node receives the access-policy
attribute explicitly before graph instantiation. Attaching it only to the
ordinary stream did not reproduce reliably because PyTorch uses a separate
capture stream. Those earlier exploratory timings are not the final result.

## Result

Medians of 40 single-operation CUDA graph replays, microseconds. Before each
replay, a separate 128 MiB read-modify-write pass intervenes; its time is
excluded. Packing and one-time BF16 cache creation are excluded.

| BF16 decoded bytes | Decode every use | BF16 cache, normal | BF16 cache, persisting |
| --- | ---: | ---: | ---: |
| 8 MiB (2048 x 2048) | 344.99 | 323.42 | 147.30 |
| 32 MiB (8192 x 2048) | 1347.30 | 1285.66 | 1230.90 |
| 128 MiB (8192 x 8192) | 5805.36 | 5483.36 | 5489.90 |

For the 8 MiB case, persisting decoded data reduces this primitive's time by
54.5% relative to an ordinary BF16 cache, or 57.3% relative to decode-each-use.
A second graph-policy run measured 147.31 us versus 323.34 us for that pair.
The 128 MiB case shows no persistence benefit. The 32 MiB case exceeds the
persisting budget and yields only a small difference in the evicted setup;
its immediate-reuse persisting measurement is slower than ordinary caching.

These are standalone feasibility measurements, **not VM or model E2E**. The
control is the original unpipelined W4A16 primitive, not the newer TMA VM
operator. GPU and memory clocks are not locked. Raw data also contains immediate
reuse timings; the first decode-each-use immediate-reuse sample is anomalously
slower than the evicted measurement in repeated runs. Do not derive a cache
hit rate or a model speedup from these timing-only measurements.

The useful model experiment would selectively predecode and persist a few
matrices within the budget. For example, three 2048 x 2048 BF16 matrices use
24 MiB. That is only about 1% of Llama 1B's decoded weight bytes. A full decoded
cache is roughly 2.5 GB and resides predominantly in LPDDR, so it trades away
compressed weight streaming. No full-model cache improvement is established.

CUDA's persistence API and limits are documented in the
[NVIDIA L2 cache guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html).

The final graph-policy path passes compute-sanitizer memcheck with zero errors.
All three sizes produce bit-identical cached and decode-each-use outputs.

## Reproduction

```sh
source tools/thor/env.sh
bash experimental/weight_only_fp4/build.sh
python -m tools.analysis.nvfp4.weight_cache_bench --shapes 2048x2048,8192x2048,8192x8192 --repeats 40 --output local/cache-final.json
compute-sanitizer --tool memcheck --error-exitcode 7 python -m tools.analysis.nvfp4.weight_cache_bench --shapes 2048x2048 --repeats 1 --output local/cache-memcheck.json
```

The policy is reset and the previous context's L2 set-aside restored after each
case. GPU-driver settings, clock settings, and other processes are untouched.
