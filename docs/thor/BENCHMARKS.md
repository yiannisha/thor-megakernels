# Thor enabling port versus vLLM

This is the native enabling-only port: upstream static scheduling, original SIMT matvec and Llama arithmetic, with Thor build/launch/synchronization correctness fixes. The experimental optimizations have been removed. See [retained changes](BRINGUP.md).

| Engine | Requests | Median ms/token | p95 | p99 |
|---|---:|---:|---:|---:|
| Native Thor port | 90 | 10.395 | 10.413 | 10.423 |
| vLLM 0.32.0 | 90 | 11.345 | 11.393 | 11.401 |

The native port has **8.37% lower observed median decode latency** than vLLM in this comparison. An earlier native run, performed before vLLM, measured 10.389 ms/token; the table uses the final native run after vLLM and validation.

Protocol: local Llama-3.2-1B-Instruct, identical BF16 weights and hashed 1024-token prompt, batch/concurrency one, greedy decoding, 100 tokens after the first prefill token. Each engine has five warmups and three batches of thirty requests. p95/p99 describe request-mean time per output token, not individual-token latency tails. Native and vLLM runs are sequential. Existing unrelated GPU workloads and automatic clocks remain uncontrolled; no driver or power settings were changed. These are observed runtime latencies, not a controlled optimization speedup.

Native timing excludes loading, prefill and KV restoration, and includes host dispatch, embedding, the megakernel, head, greedy selection and final token transfer. vLLM uses engine first/last-token timestamps, including scheduling and sampling; graphs remain enabled, prefix caching/speculative decode are disabled, and FlashInfer autotuning is disabled explicitly. These scopes differ at runtime boundaries. vLLM TTFT/end-to-end measurements are retained in the raw data.

Per user direction, numerical tolerance metrics are informational and do not block timing. The native first-step reference check passes; trajectory agreement is recorded without further numerical tuning.

Validation: fourteen host tests and thirty-six model-free VM cases pass. Full native decode passes kernel-filtered memcheck, synccheck and racecheck with zero errors/hazards. H100/B200 paths compile; execution on those devices is untested. Sanitizer/pilot timings are excluded from the performance table.

- [Final native measurements](enabling-port/native-final.json)
- [vLLM measurements](enabling-port/vllm.json)
- [Initial native measurements](enabling-port/native.json)
- [Comparison and trajectory details](enabling-port/comparison.md)
- [Benchmark process records](enabling-port/processes.json)
- [Build and sanitizer records](enabling-port/validation-processes.json)

The implementation cleanup is commit `8c09c7a`. Removed experiments and historical datasets remain on `archive/thor-experiments-20261027`. The initial native run precedes a terminal-blank-line cleanup in the source-fingerprint helper; the final native run and shipped binary match the final source fingerprint.

## Reproduction

```sh
source tools/thor/env.sh
make -B -C demos/low-latency-llama GPU=THOR
python -m tools.thor.autoregressive_bench --warmups 5 --repeats 30 --batches 3 --output local/native-port.json
source tools/thor/vllm_env.sh
python -m tools.thor.vllm_bench --disable-flashinfer-autotune --warmups 5 --repeats 30 --batches 3 --output local/vllm.json
```
