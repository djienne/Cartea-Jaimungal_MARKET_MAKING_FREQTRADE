# Local performance characterization

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
benchmarks. They exceed the configured 150 ms production p99 gate and therefore
support refusing production trading from this machine.

The quote benchmark includes episodic time interpolation, fractional inventory
interpolation, fee/cushion/clamp assembly, prospective risk checks, and venue
rounding. The HJB solve is a cold-path operation performed after calibration.

Excluded from these measurements:

- network and exchange round-trip latency;
- JSON/WebSocket parsing;
- Parquet reads and writes;
- queue position and fill uncertainty;
- OS scheduling around the dedicated hot thread.

Run it with:

```powershell
cargo run --locked --manifest-path rust_live/Cargo.toml `
  --release --bin hot-path-bench
```
