# Autoregressive decode measurements

Batch/concurrency one, BF16, context 1024, 100 decode tokens after the first prefill token. Existing external load and automatic clocks are uncontrolled. Request-level p95/p99 describe mean TPOT across requests, not individual-token tails.

| Engine | Timing protocol | Requests | Median TPOT (ms) | p95 (ms) | p99 (ms) |
|---|---|---:|---:|---:|---:|
| native | PASS | 90 | 11.189 | 11.221 | 11.229 |
| vllm | PASS | 90 | 11.640 | 11.685 | 11.697 |

Native wall timing includes host dispatch, embedding, decoder, head, greedy selection and final token transfer; prefill and KV restoration are excluded. vLLM timing uses engine first/last-token timestamps and includes engine scheduling/sampling. TTFT/end-to-end latency is saved separately. These are application-level decode scopes with different runtime boundaries; they are not isolated kernel speedups.

Native sequence numerical qualification remains open. A ten-step teacher-forced diagnostic exceeded the 3% raw logit NRMSE gate at one step (3.076%); see [raw checks](autoregressive-teacher-forced.json). Native first-step acceptance does not qualify the full trajectory. No quality-matched speedup is claimed.

- native: raw data [bench-native-autoregressive.json](bench-native-autoregressive.json); 40/90 requests reproduce its first trajectory.
- vllm: raw data [bench-vllm-autoregressive.json](bench-vllm-autoregressive.json); 90/90 requests reproduce its first trajectory.

First measured trajectories agree at 13/101 token positions. Timing PASS means protocol completion; full numerical equivalence is not implied. Native first-step numerical checks and its PyTorch trajectory agreement are retained in the raw report.
