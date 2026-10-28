| Run | Engine / variant | Context | Status | Scope | Samples | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|
| bc68e8c2-a999-4c89-9ec2-8bbc54f0ec22 | native:segments-16-eager | 1024 | PASS | decode | 90 | 11.436272 | 14.637566 | 14.850542 |
| db71b816-7851-4ff4-b1d1-f78f351563ac | native:segments-4-eager | 1024 | PASS | decode | 90 | 11.344928 | 14.696215 | 14.817895 |
| 7aa50b29-39b6-434d-b606-3b71e0cd172a | native:segments-1-eager | 1024 | PASS | decode | 90 | 11.651056 | 14.776640 | 14.994274 |
| 0c02fbfc-d43f-40ba-8d51-eb5254410f9b | native:segments-16-graph | 1024 | PASS | decode | 90 | 11.357360 | 14.493175 | 14.627684 |
| 6345a199-fd2a-470e-b41d-8610f12f8cc6 | native:segments-4-graph | 1024 | PASS | decode | 90 | 11.305680 | 14.359416 | 14.658236 |
| 9bcd1fa4-76c9-4e92-9d38-8538989ed844 | native:segments-1-graph | 1024 | PASS | decode | 90 | 11.476288 | 14.537530 | 14.734016 |

Same-cache decode replays are not autoregressive generation. Read each run's scope, accuracy gates, power profile and contention metadata before comparing. Tail estimates from small sample sets have limited precision.

All variants use the real Llama-1B model, same BF16 weights and context-1024 frozen KV. Segment sizes 16/4/1 mean 1/4/16 persistent-kernel launches. Stream order guarantees that earlier segments finish before later segments consume their global activations/barriers; barriers are reset once per full decode. The host validates complete instruction coverage and rejects backward cross-segment dependencies. Every variant passes ninety numerical checks. The sixteen-kernel graph also passes native-filtered memcheck and synccheck. Graph replay currently fixes the decode position; no advancing-position autoregressive graph speedup is claimed.

The four-kernel graph observes 11.306 ms p50 versus 11.436 ms for the single ordinary launch. These small sequential-run differences under uncontrolled existing load and automatic clocks do not select a default or establish statistical significance. The existing temporal numerical failures remain open.
