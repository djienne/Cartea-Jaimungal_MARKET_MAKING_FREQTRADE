use crate::config::{InstrumentProfile, Network};
use crate::instrument::InstrumentSpec;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AssetMeta {
    name: String,
    sz_decimals: u32,
    max_leverage: f64,
    #[serde(default)]
    margin_table_id: u32,
    #[serde(default)]
    only_isolated: bool,
    #[serde(default)]
    margin_mode: String,
    #[serde(default)]
    is_delisted: bool,
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
        margin_table_id: asset.margin_table_id,
        only_isolated: asset.only_isolated,
        margin_mode: asset.margin_mode.clone(),
        is_delisted: asset.is_delisted,
        metadata_fingerprint: {
            let encoded = serde_json::to_vec(&serde_json::json!({
                "symbol": profile.symbol,
                "dex": profile.dex,
                "asset_id": asset_id,
                "sz_decimals": asset.sz_decimals,
                "max_leverage": asset.max_leverage,
                "margin_table_id": asset.margin_table_id,
                "only_isolated": asset.only_isolated,
                "margin_mode": asset.margin_mode,
                "is_delisted": asset.is_delisted,
            }))?;
            hex::encode(Sha256::digest(encoded))
        },
    };
    spec.validate()?;
    Ok(spec)
}
