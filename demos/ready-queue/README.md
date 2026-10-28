# Ready-only persistent worker prototype

This standalone FP32 vector-DAG experiment tests scheduling and publication. It does not execute Llama/OpenPI instructions and is not evidence of model decode acceleration. Both `static` and `ready` modes are selectable, with `both` as the comparison mode.

```bash
source tools/thor/env.sh
python -m tools.experiments.ready_queue --stress --output docs/thor/ready-queue-stress
python -m tools.experiments.ready_queue --output docs/thor/ready-queue-timing
python -m tools.experiments.ready_queue --sanitizer memcheck --output docs/thor/ready-queue-memcheck
```

The fixture creates random, chain or independent-task DAGs with up to two distinct earlier parents. Every node computes a 256-element vector using immutable input, parent vectors and a task-dependent bias. A task-dependent sleep perturbs progress. A host FP32 implementation checks every output element and per-task execution counters must equal one after every warmup and timed invocation. The runner imposes a process-group timeout, records source/binary hashes, toolchain and device inventory, and preserves failures. Stress runs are correctness-only; the timing mode saves five warmups and three batches of thirty kernel-event samples, excluding reset and validation.

## Admission and progress contract

The cooperative grid is checked against CUDA occupancy and hardware cooperative-launch support. The prototype conservatively limits workers to the physical SM count. Entry grid synchronization establishes admission before any dependencies are consumed. Worker IDs are logical block indices; no one-CTA-per-SM placement is assumed.

The initial queue contains all roots. A completed node decrements each child's pending-parent counter with device-scope acquire/release ordering. Only the last parent publishes the child. Consumers execute only published ready IDs. Empty workers poll for publication/completion; they never take an unready graph node. A consumer can wait briefly on a reserved queue slot, whose producer is an already resident worker and publishes before dequeuing again. No global lock or unscheduled producer is required.

Each node is enqueued once. Capacity equals node count; the linear queue does not wrap, so there is no slot reuse/ABA in this prototype. This is a bounded single-inference design, not an unbounded production MPMC ring. All output-writing threads fence and synchronize before the leader publishes completion. The pending-counter acquire/release chain collects parent writes, and queue-slot release/acquire publication carries them to the child. Each CTA synchronizes around its shared task ID.

The static control assigns increasing task IDs round-robin. All parents have smaller IDs, so adding queue-order edges preserves acyclicity; all CTAs are admitted before they wait. It polls parent completion rather than paying dynamic enqueue/dependency-update overhead. These distinct scheduling costs are part of the comparison.

## Limits and next integration step

Timing order is static then dynamic, under existing external load and automatic clocks; results are exploratory and cannot establish a controlled speedup. Delays are synthetic. Vector tasks have no TMA pages, warp-specialized loader/storer, instruction serialization or model operands.

Native integration still requires mapping each real instruction's dependency count and completion contribution, handling multi-producer reductions without early publication, defining a queue lifetime across decode steps, admitting all workers with actual compiled shared-memory/register usage, and preserving checked model accuracy and page lifetimes. Repeat the same watchdog, sanitizer and randomized-DAG coverage before model timing. This experiment tests the scheduling mechanism and memory-order contract rather than substituting a toy workload for model qualification.
