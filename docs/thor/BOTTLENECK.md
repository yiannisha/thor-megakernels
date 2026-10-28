# Current Llama-1B bottleneck investigation

The runtime and native binary are unchanged from the enabling-only port. This is diagnostic measurement, not a new runtime optimization. All runs use the existing shared Thor; external load and automatic clocks remain uncontrolled.

## Measured steady-state execution

Batch one, context 1024, 100 successive greedy steps, matching cache capacity and five complete 100-step warmup requests. Model loading/prefill/cache restore are outside the marked trace. Native submission is annotated with NVTX; capture starts/stops with CUDA profiler APIs.

| Quantity | ms/token |
|---|---:|
| Unprofiled same-probe execution | 10.4002 |
| Profiled marked execution | 10.3988 |
| Persistent megakernel | 10.3329 |
| Other GPU activity | 0.0432 |
| Time without traced GPU activity | 0.0228 |

The megakernel occupies 99.37% of the marked elapsed interval. Host binding averages 0.2229 ms per call, but overlaps execution; it must not be added to GPU time. Blocking CUDA copy/API durations similarly overlap GPU execution. This rules against Python/launch setup as the dominant exposed bottleneck in this workload. It does not identify stalls inside the megakernel.

Initial short-warmup, smaller-capacity traces measured about 13.4–13.8 ms/token; they remain recorded but are excluded from steady-state conclusions. Matching the actual benchmark warmup/capacity reproduces its approximately 10.4 ms/token result.

## Independent memory-throughput reference

The checkpoint contains 1,235,814,400 stored BF16 parameters (2,471,628,800 bytes); the tied embedding matrix is also the output head. A separate CUDA probe reads a pseudorandom integer buffer (avoiding an all-zero test case) of that size, performs a checksum reduction, and writes a tiny result. The buffer exceeds the 32 MiB L2. Five warmups and thirty event-timed samples are used per launch geometry; checksum agrees across geometries. It is not a model execution or an exact reproduction of TMA access patterns.

| Streaming bytes | CTAs × threads | Median ms | Read GB/s |
|---:|---|---:|---:|
| 2,471,628,800 | 20 × 512 | 10.4497 | 236.53 |
| 2,471,628,800 | 80 × 512 | 9.6835 | 255.24 |
| 2,471,628,800 | 320 × 512 | 9.4899 | 260.45 |
| 6,000,000,000 | 20 × 512 | 25.4198 | 236.04 |
| 6,000,000,000 | 80 × 512 | 23.7424 | 252.71 |
| 6,000,000,000 | 320 × 512 | 23.1495 | 259.19 |

The full decoder is close to the 20-CTA read-only time and roughly 0.9 ms above the 320-CTA small-buffer time. This supports weight streaming/memory throughput as the leading bottleneck hypothesis. It is not proof of actual decoder DRAM traffic, and the difference is not measured removable overhead. The decoder also executes arithmetic, interpreter logic, synchronization and KV/intermediate traffic. More CTAs accelerate this simple streaming probe; earlier VM multi-CTA tests did not establish a gain.

Nsight Compute cannot collect the actual DRAM-byte, throughput and stall counters: it reports `ERR_NVGPUCTRPERM`. No administrator privileges or system settings were changed. The precise division between memory-interface saturation, insufficient in-flight memory work and internal scheduling stalls therefore remains unverified. NVIDIA documents this restriction in its [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

## Implications for the five proposals

1. Reducing external-memory traffic remains meaningful. For this batch-one model, however, most bytes are distinct weights, so generic activation reuse/fusion has limited leverage unless measurements reveal substantial redundant traffic.
2. Granularity and locality changes can help only if they improve the GPU data path or avoid internal idle time; the trace does not independently prove an imbalance.
3. Pipeline/concurrency changes remain a plausible GPU-side test, but the streaming microbenchmark does not overturn the prior negative VM results.
4. Coherent control/prepared launches are unlikely to substantially improve this particular steady-state decode: exposed GPU-idle time is only about 0.023 ms/token in the trace. They may help smaller tasks or other workloads.
5. Native backend work could help internal instruction/transfer efficiency or compute-heavy workloads. Its value for this batch-one streaming case remains unmeasured.

## Larger model

The 6 GB synthetic read maintains approximately the same bandwidth and takes approximately 23–25 ms, illustrating scaling of traffic rather than a new bottleneck class. This is not a 3B model benchmark. At batch one, a larger dense model generally adds weight bytes and matrix-vector work together; increasing parameter count alone does not increase weight reuse per fetch. It is useful for generalization tests, but is unlikely to expose more relative host/runtime overhead.

Only the 1B checkpoint is local. The native demo has fixed 16-layer/2048-hidden/64-head-dimension operator shapes, so a different checkpoint requires a real shape port; changing a model path is insufficient. No larger checkpoint was downloaded.

## Reproduction

```sh
source tools/thor/env.sh
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output=local/profile-new python -m tools.analysis.thor_bottleneck --capture --output local/trace-new.json
python -m tools.analysis.thor_bottleneck --output local/unprofiled-new.json
nvcc -O3 -std=c++17 -gencode=arch=compute_110a,code=sm_110a tools/analysis/stream_read.cu -o local/stream-read-new
local/stream-read-new
local/stream-read-new 6000000000
```

Raw profiler artifacts remain under `local/`. The marked GPU events, summary, throughput samples, counter error and binary/source provenance are saved alongside this report.
