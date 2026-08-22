use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use hyperliquid_connector::acceptance::{self, AcceptancePhase};
use mm_live::config::AppConfig;
use mm_live::hyperliquid::meta::discover_instrument;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "mm-live-acceptance",
    about = "Explicitly bounded Hyperliquid real-account acceptance runner"
)]
struct Cli {
    #[arg(long, default_value = "rust_live/config/cashcat.toml")]
    config: PathBuf,
    #[arg(long, value_enum)]
    phase: Phase,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Phase {
    Verify,
    Leverage,
    TwoSided,
    CrossingAlo,
    UnknownOutcome,
    RestartOrderPrepare,
    RestartOrderRecover,
    Deadman,
    MakerFill,
    RestartPositionPrepare,
    RestartPositionRecover,
    Final,
}

impl From<Phase> for AcceptancePhase {
    fn from(value: Phase) -> Self {
        match value {
            Phase::Verify => Self::Verify,
            Phase::Leverage => Self::Leverage,
            Phase::TwoSided => Self::TwoSided,
            Phase::CrossingAlo => Self::CrossingAlo,
            Phase::UnknownOutcome => Self::UnknownOutcome,
            Phase::RestartOrderPrepare => Self::RestartOrderPrepare,
            Phase::RestartOrderRecover => Self::RestartOrderRecover,
            Phase::Deadman => Self::Deadman,
            Phase::MakerFill => Self::MakerFill,
            Phase::RestartPositionPrepare => Self::RestartPositionPrepare,
            Phase::RestartPositionRecover => Self::RestartPositionRecover,
            Phase::Final => Self::Final,
        }
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    mm_live::BuildInfo::current().ensure_optimized()?;
    rustls::crypto::ring::default_provider()
        .install_default()
        .map_err(|_| anyhow::anyhow!("cannot install rustls ring crypto provider"))?;
    let cli = Cli::parse();
    let config = AppConfig::load(&cli.config)?;
    if !config.live.enabled {
        bail!("live.enabled=false; refusing before credentials or order transport");
    }
    let instrument = discover_instrument(config.runtime.network, &config.instrument).await?;
    acceptance::run(&config, instrument, cli.phase.into()).await
}
