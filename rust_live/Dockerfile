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
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home mm-live \
    && mkdir -p /opt/mm/config /opt/mm/run /opt/mm/reports /opt/scripts/HL_data \
    && chown -R mm-live:mm-live /opt/mm /opt/scripts/HL_data
COPY --from=builder /tmp/mm-live /usr/local/bin/mm-live
COPY rust_live/config/cashcat.toml rust_live/config/cashcat.validation.json /opt/mm/config/
USER 10001
WORKDIR /opt/mm
VOLUME ["/opt/scripts/HL_data", "/opt/mm/run", "/opt/mm/reports"]
ENTRYPOINT ["/usr/local/bin/mm-live"]
CMD ["--config", "/opt/mm/config/cashcat.toml", "dry-run"]
