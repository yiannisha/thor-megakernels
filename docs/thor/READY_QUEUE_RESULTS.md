# Ready-only persistent queue prototype results

The standalone source is `demos/ready-queue/probe.cu`; [design and progress contract](../../demos/ready-queue/README.md). This is a model-free FP32 vector DAG, not Llama/OpenPI integration or a model-runtime speedup.

All 72 stress fixtures pass at 1/17/257/4096 nodes, 1/3/20 workers, two seeds, and random/chain/wide DAG shapes. Each fixture runs static and ready-queue modes with five warmups and three checked repeats. Every node executes exactly once and all 256 output elements per node match the CPU FP32 reference (observed maximum absolute difference zero). Task-dependent delays perturb progress. Invalid zero-task/zero-worker and over-SM-count grids are rejected before launch.

Each of memcheck, racecheck and synccheck passes random/chain/wide 257-node fixtures with twenty workers in both modes. All tool summaries report zero errors/hazards. The cooperative admission check ensures all workers are resident before consuming dependencies. Ready publication uses device acquire/release atomics, and output writers fence before completion. See [stress report](ready-queue-stress/report.json) and `ready-queue-{memcheck,racecheck,synccheck}/`.

Exploratory timing fixture: 1024 nodes, 256 FP32 elements per node, seed 1, random DAG, task-dependent delay quantum 500 ns. Kernel CUDA events exclude reset, CPU reference and output checking. Each row has ninety samples.

| Workers | Mode | p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|
| 12 | static_rr | 1.911 | 2.001 | 2.005 |
| 12 | ready_queue | 1.198 | 1.225 | 1.237 |
| 20 | static_rr | 0.799 | 0.808 | 0.811 |
| 20 | ready_queue | 0.774 | 0.800 | 0.801 |

These fixed-order, static-then-ready measurements show lower observed ready-queue latency on this delayed vector fixture, especially at twelve workers. Existing GPU load, automatic clocks and overlapping project CPU compilation are uncontrolled. No statistical or causal speedup is established. The result does not predict TMA/warp-specialized Llama performance. Production integration still needs real instruction readiness counts, reduction completion semantics, page lifetime validation, queue reset across decode steps and full-model accuracy qualification. The prototype keeps a bounded, one-enqueue-per-node linear queue rather than an unbounded reusable ring. [Timing report and source/binary hashes](ready-queue-timing/report.json).
