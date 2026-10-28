# Autoregressive decode measurements

Batch/concurrency one, BF16, context 1024, 100 decode tokens after the first prefill token. Existing external load and automatic clocks are uncontrolled. Request-level p95/p99 describe mean TPOT across requests, not individual-token tails.

| Engine | Timing protocol | Requests | Median TPOT (ms) | p95 (ms) | p99 (ms) |
|---|---|---:|---:|---:|---:|
| native | PASS | 90 | 11.144 | 11.169 | 11.187 |
| vllm | PASS | 90 | 11.666 | 11.737 | 11.742 |

Native wall timing includes host dispatch, embedding, decoder, head, greedy selection and final token transfer; prefill and KV restoration are excluded. vLLM timing uses engine first/last-token timestamps and includes engine scheduling/sampling. TTFT/end-to-end latency is saved separately. These are application-level decode scopes with different runtime boundaries; they are not isolated kernel speedups.

Timing protocol completion alone does not establish numerical equivalence or a quality-matched speedup.

Native per-step numerical qualification against math attention: **FAIL**, 3/100 failed steps; worst logit NRMSE 3.085%, hidden NRMSE 2.556%. The tested binary, inputs and native KV trajectory match this timing run. See [raw qualification](teacher-native-compensated-current-math.json). No quality-matched speedup is claimed.

- native: raw data [bench-native-compensated-autoregressive.json](bench-native-compensated-autoregressive.json); 90/90 requests reproduce its first trajectory.
- Native/PyTorch greedy sequences match at all positions in 90/90 requests.
- vllm: raw data [bench-vllm-compensated-comparison.json](bench-vllm-compensated-comparison.json); 90/90 requests reproduce its first trajectory.

First measured trajectories agree at 13/101 token positions. Timing PASS means protocol completion; full numerical equivalence is not implied. Native first-step numerical checks and its PyTorch trajectory agreement are retained in the raw report.
