FROM rust:1.92-bookworm AS builder
WORKDIR /src
ARG MM_RUSTFLAGS=""
ARG MM_GIT_REVISION="unknown"
ENV RUSTFLAGS=${MM_RUSTFLAGS}
ENV MM_GIT_REVISION=${MM_GIT_REVISION}
COPY rust_live/Cargo.toml rust_live/Cargo.lock rust_live/rust-toolchain.toml ./rust_live/
COPY rust_live/build.rs ./rust_live/build.rs
COPY rust_live/crates ./rust_live/crates
COPY rust_live/src ./rust_live/src
COPY rust_live/tests ./rust_live/tests
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/src/rust_live/target \
    cargo build --locked --manifest-path rust_live/Cargo.toml --release --bin mm-live \
    && cp rust_live/target/release/mm-live /tmp/mm-live

FROM debian:bookworm-slim
ARG MM_BUILD_FLAVOR="portable-o3"
ARG MM_GIT_REVISION="unknown"
LABEL org.opencontainers.image.title="CASHCAT Cartea-Jaimungal Rust runtime" \
      org.opencontainers.image.description="O3/fat-LTO pure-Rust market-making runtime" \
      org.opencontainers.image.vendor="local research" \
      org.opencontainers.image.revision=${MM_GIT_REVISION} \
      mm.build.flavor=${MM_BUILD_FLAVOR}
# /opt/scripts/HL_data is not arbitrary and must track /opt/mm/config. Configs
# set `data_dir = "../../scripts/HL_data"`, resolved relative to the *config
# file*, so with configs at /opt/mm/config the tape must sit at
# /opt/scripts/HL_data. Move either path without the other and calibration
# fails with "no usable price data under ...".
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home mm-live \
    && mkdir -p /opt/mm/config /opt/mm/run /opt/mm/reports /opt/scripts/HL_data \
    && chown -R mm-live:mm-live /opt/mm /opt/scripts/HL_data
COPY --from=builder /tmp/mm-live /usr/local/bin/mm-live
# The whole config directory, not just cashcat.toml: dry-run-grid needs
# cashcat_dryrun_realistic.toml and grid_cashcat.toml too.
COPY rust_live/config/ /opt/mm/config/
USER 10001
WORKDIR /opt/mm
VOLUME ["/opt/scripts/HL_data", "/opt/mm/run", "/opt/mm/reports"]

# Liveness alone is not health here. A wedged `connect_async` leaves the process
# responsive and the leaderboard advancing while the market feed is gone -- on
# 2026-08-26 that state lasted 19.65 h. `grid-health` checks both, reading only
# the leaderboard: no config, no runtime, no network.
HEALTHCHECK --interval=60s --timeout=15s --retries=3 --start-period=300s \
    CMD ["/usr/local/bin/mm-live", "grid-health", \
         "--leaderboard", "/opt/mm/reports/grid_live/leaderboard.json"]

ENTRYPOINT ["/usr/local/bin/mm-live"]
CMD ["--config", "/opt/mm/config/cashcat.toml", "dry-run"]
