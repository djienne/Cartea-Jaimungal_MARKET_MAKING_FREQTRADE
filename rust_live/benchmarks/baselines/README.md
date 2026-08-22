# Pinned-runner baselines

Store an approved `hot-path-bench` JSON report here only after recording it on
the designated production-class runner with `MM_BENCH_CPU` set. Baselines are
valid only for the recorded CPU, Rust toolchain, target features, and build
profile. Use `rust_live/scripts/check-performance.ps1` for the relative 5% p50,
10% p99, and 5% monitoring-overhead gates. Unpinned developer results are
informational and must not replace a pinned baseline.
