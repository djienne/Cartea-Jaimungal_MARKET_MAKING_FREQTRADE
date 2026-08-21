use crate::config::{InstrumentProfile, Network};
use crate::instrument::InstrumentSpec;
use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AssetMeta {
    name: String,
    sz_decimals: u32,
    max_leverage: f64,
}

#[derive(Debug, Deserialize)]
struct MetaResponse {
    universe: Vec<AssetMeta>,
}

pub async fn discover_instrument(
    network: Network,
    profile: &InstrumentProfile,
) -> Result<InstrumentSpec> {
    let response = reqwest::Client::new()
        .post(network.info_url())
        .json(&serde_json::json!({"type": "meta"}))
        .send()
        .await
        .context("cannot fetch Hyperliquid metadata")?
        .error_for_status()
        .context("Hyperliquid metadata request failed")?
        .json::<MetaResponse>()
        .await
        .context("cannot decode Hyperliquid metadata")?;
    let (asset_id, asset) = response
        .universe
        .iter()
        .enumerate()
        .find(|(_, asset)| asset.name == profile.symbol)
        .with_context(|| format!("{} is absent from Hyperliquid metadata", profile.symbol))?;
    if asset.sz_decimals != profile.expected_sz_decimals {
        bail!(
            "{} szDecimals changed: expected {}, venue reports {}",
            profile.symbol,
            profile.expected_sz_decimals,
            asset.sz_decimals
        );
    }
    let spec = InstrumentSpec {
        symbol: profile.symbol.clone(),
        dex: profile.dex.clone(),
        asset_id: u32::try_from(asset_id)?,
        sz_decimals: asset.sz_decimals,
        max_price_decimals: 6_u32.saturating_sub(asset.sz_decimals),
        max_significant_figures: profile.max_significant_figures,
        max_leverage: asset.max_leverage,
        minimum_notional: profile.minimum_notional,
    };
    spec.validate()?;
    Ok(spec)
}
