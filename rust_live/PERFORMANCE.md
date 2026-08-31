# Local performance characterization

The approved 2026-08-31 Windows baseline is the median of seven independent
release processes pinned to logical CPU 4 on the Ryzen 9 7900. The pure policy
kernel measured 57.81 ns p50 and 104.69 ns p99 of 64-decision batch means; the
exact production `HotPathEngine::step` measured 73.44/134.38 ns. These are
batch-mean distributions, not individual-decision tail latency. The checked-in
environment fingerprint and baseline live under `benchmarks/baselines`; the
gate reruns the same seven-process study and refuses incompatible hosts.

## Earlier characterizations

The post-remediation unpinned Windows release run on 2026-08-23 measured a
55.57 ns baseline quote-loop median and 58.22 ns monitored (4.78% one-in-16
sampling overhead, within the 5% gate). Independent 64-decision batches
reported p50/p95/p99 of 56.25/93.75/98.44 ns per decision. The HJB solve was
0.599 ms at p50. The gain over the previous 84–89 ns range comes from O(1)
time bracketing (the uniform grid's `partition_point` binary search became a
division), an integer `ilog10` price decade replacing two float `log10` calls
per decision, and the two-stage typed feed decode upstream of the kernel. The
release profile now unwinds on panic (so a panicking task cannot bypass live
order cancellation); the switch measured no regression. Unpinned run —
informational, not an approved gate baseline.

Note when comparing across dates: `runtime.hot_path_cpu = 4` is now set in
`config/cashcat.toml` and the tokio runtime is bounded (≤6 workers, blocking
pool capped) with worker affinity excluding the hot core. The benchmark
itself still pins only when `MM_BENCH_CPU` is set.

## Superseded 2026-08-22 measurements

The prior unpinned run measured an 89.22 ns paired quote-loop median. Independent 64-decision batches reported
p50/p95/p99 of 87.50/132.81/146.88 ns per decision. One-in-16 latency sampling
raised the paired median to 93.58 ns, or 4.88%. The HJB solve distribution was
0.698/0.976/1.120 ms at p50/p95/p99. This is a large improvement over the
pre-change 143–156 ns quote range, chiefly from eliminating two inventory-grid
allocations per decision and selecting both depths in one lookup. Because this
run was not CPU-pinned, it remains informational rather than an approved gate
baseline.

## Historical pre-hardening measurements

Measured on 2026-08-21 with an AMD Ryzen 9 7900 using the locked Rust 1.92
release profile (`opt-level=3`, fat LTO, one codegen unit). These are computation
benchmarks, not exchange-latency measurements.

| Component | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| BBO/inventory/time → Cartea–Jaimungal quote | 150.45 ns | 148.11 ns | 147.77 ns |
| Complete 600-step asymmetric HJB solve | 0.583 ms | 0.621 ms | 0.597 ms |

A separate 2026-08-22 Linux/Docker verification on the same host measured
152.34 ns per quote decision and 0.322 ms per 600-step solve. Container and host
figures are reported separately rather than treated as a scientific parity gate.

After adding the rolling latency producer, a paired nine-run release benchmark
measured a 147.59 ns baseline median and 148.93 ns monitored median: 1.34 ns
(0.91%) incremental median cost with one-in-16 sampling and four-sample batching.
Run minima/maxima overlapped, so this is a local characterization rather than a
portable overhead guarantee. The same run measured 0.303 ms per 600-step HJB
solve. Sorting, percentiles, JSON, and file I/O remain on the observer thread.

Real mainnet connector observations from this development machine were much
larger than compute latency: `/info` p99 was about 324 ms, account WebSocket ping
p99 about 561 ms, passive submit acknowledgement about 695 ms, passive cancel
acknowledgement about 931-961 ms, and the final IOC submit acknowledgements about
959-976 ms. These are end-to-end development-host measurements, not production
benchmarks. They exceed the configured 150 ms production p95 gate and therefore
support refusing production trading from this machine.

The stateful-backend acceptance campaign measured WebSocket order
acknowledgements around 0.60-1.07 seconds and cancel acknowledgements around
0.94-1.02 seconds. Once invoked, reduce-only IOC market close reached
authoritative flat confirmation in 1.58 seconds for a long and 1.79 seconds for
a short. These timings validate behavior and instrumentation, not competitive
latency. A production-mode continuous smoke submitted zero actions because its
20-sample public/account warm-up could not complete in the bounded run.

The quote benchmark includes episodic time interpolation, fractional inventory
interpolation, fee/cushion/clamp assembly, prospective risk checks, and venue
rounding. The HJB solve is a cold-path operation performed after calibration.

Excluded from these measurements:

- network and exchange round-trip latency;
- JSON/WebSocket parsing;
- Parquet reads and writes;
- queue position and fill uncertainty;
- OS scheduling around the dedicated hot thread.

Run the approved same-machine gate from `rust_live`:

```powershell
cd rust_live
.\scripts\check-performance.ps1 `
  -Baseline .\benchmarks\baselines\windows-ryzen9-7900-cpu4-rust1.92-portable.json `
  -Cpu 4 -Runs 7
```

The gate compares the median of seven pinned process runs and rejects a CPU,
core, target, Rust, profile, target-feature, or schema mismatch. Policy-kernel
and complete-hot-step p50 may regress by at most 5%; their batch-mean p99 may
regress by at most 10%. Monitoring overhead may rise by at most 5 percentage
points relative to the approved baseline. These percentiles describe batch
means, not individual-decision tails.

For a diagnostic single run, invoke `cargo run --locked --release --bin
hot-path-bench`; set `MM_BENCH_CPU` to pin it and `MM_BENCH_OUTPUT` to retain
its schema-3 JSON. A single or unpinned run is informational and is not an
approved performance gate.
