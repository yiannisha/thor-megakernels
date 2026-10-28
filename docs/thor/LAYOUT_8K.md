# 8-KiB / 256-wide layout experiment

Source snapshot `bb15f3d` adds a selectable 8-KiB page / 256-wide TMA weight tile while preserving 16-KiB / 512 compatibility. A 2048-wide matvec stage still transfers 64 KiB: it uses eight smaller pages instead of four large ones. Activation storage occupies one page. Minimum budgets are 9/17/25 pages for one/two/three weight stages. Consumer warps must divide across the eight stage pages (8 or 16 warps); the four-warp combination is explicitly rejected. The GPU matrix here uses sixteen consumer warps and cap 96.

Attention retains the existing interleaved K/V packing on 16-KiB pages. On 8-KiB pages, three K tiles and three V tiles use separate pages, with shared Q/output/L storage in a third page. Waits, completion publication and page-release order include both KV pages. Compile-time size checks cover matvec activations/tiles, attention staging and reduction buffers. The host executes the same release arithmetic as CUDA, with 12,600 seeded lifetime/reallocation sequences spanning both page granularities.

| Page / chunk | Stages | Pages | Static + dynamic SMEM bytes | Resident CTAs/SM | p50 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|
| 16 KiB / 512 | 1 | 5 | 10320 + 82943 | 1 | 10.964 | 14.497 |
| 16 KiB / 512 | 2 | 9 | 10352 + 148479 | 1 | 11.355 | 15.020 |
| 16 KiB / 512 | 3 | 13 | 10384 + 214015 | 1 | 11.564 | 14.824 |
| 8 KiB / 256 | 1 | 9 | 10352 + 74751 | 1 | 10.978 | 14.468 |
| 8 KiB / 256 | 2 | 17 | 10416 + 140287 | 1 | 11.432 | 14.890 |
| 8 KiB / 256 | 3 | 25 | 10480 + 205823 | 1 | 11.548 | 14.847 |

Both matrices use identical source hashes, BF16 numerical variants, context 1024, seed 1, batch 1 and power-profile observation. Each variant passes all eighteen opcode cases and ninety timed full-model replay checks (108 opcode cases and 540 timed checks total). Each 8-KiB layout additionally passes all eighteen cases under each of racecheck, memcheck and synccheck (162 sanitizer cases, all process return codes zero). The raw matrices retain commands, compiler output, resource usage and numerical results. H100 and B200 builds pass; execution on those devices remains NOT RUN.

The smaller granularity saves 8,160/8,128/8,096 shared-memory bytes at stages 1/2/3, but does not raise occupancy at cap 96. The observed timing differences between page granularities are small and mixed. Runs are sequential under uncontrolled existing load and automatic clocks; neither a causal speedup nor a new default is established. These are one-token frozen-cache replays, not autoregressive timing or temporal qualification.

Raw evidence: `layout-8k-256/`, `layout-16k-control/`, `layout-8k-{racecheck,memcheck,synccheck}/`, [phase commands/status](layout-qualification-phases.json), [cross-build output](layout-cross-builds.json). The [raw-verified table](benchmark-page-width.md), CSV and SVG show the full distributions. The existing 100-step numerical failures remain open.

Extended 8-KiB qualification passes all 27 layout/context/seed combinations (stages 1/2/3, contexts 128/1024/8192, seeds 1/2/3), with five timed correctness checks each. All 135 greedy tokens match PyTorch; worst logit NRMSE is 1.6819%, under the unchanged 3% logit gate. Hidden/K/V gates remain 2% and all pass. These runs use zero warmups solely for correctness; their timing summaries are not steady-state performance data. [Commands and statuses](layout-extended-qualification.json), raw reports in `layout-8k-contexts/`.
