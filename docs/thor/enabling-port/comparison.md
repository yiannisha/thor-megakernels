# Autoregressive decode measurements

Batch/concurrency one, BF16, context 1024, 100 decode tokens after the first prefill token. Existing external load and automatic clocks are uncontrolled. Request-level p95/p99 describe mean TPOT across requests, not individual-token tails.

| Engine | Timing protocol | Requests | Median TPOT (ms) | p95 (ms) | p99 (ms) |
|---|---|---:|---:|---:|---:|
| native | PASS | 90 | 10.395 | 10.413 | 10.423 |
| vllm | PASS | 90 | 11.345 | 11.393 | 11.401 |

Native wall timing includes host dispatch, embedding, decoder, head, greedy selection and final token transfer; prefill and KV restoration are excluded. vLLM timing uses engine first/last-token timestamps and includes engine scheduling/sampling. TTFT/end-to-end latency is saved separately. These are application-level decode scopes with different runtime boundaries; they are not isolated kernel speedups.

Timing protocol completion alone does not establish numerical equivalence or a quality-matched speedup.

- native: raw data [native-final.json](native-final.json); 31/90 requests reproduce its first trajectory.
- Native/PyTorch greedy sequences match at all positions in 31/90 requests.
- vllm: raw data [vllm.json](vllm.json); 90/90 requests reproduce its first trajectory.

First measured trajectories agree at 13/101 token positions. Timing PASS means protocol completion; full numerical equivalence is not implied. Native first-step numerical checks and its PyTorch trajectory agreement are retained in the raw report.
