use super::auth::HyperliquidCredentials;
use crate::config::Network;
use crate::instrument::InstrumentSpec;
use crate::types::Side;
use anyhow::{bail, Context, Result};
use k256::ecdsa::SigningKey;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use tiny_keccak::{Hasher, Keccak};

pub fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak::v256();
    let mut output = [0_u8; 32];
    hasher.update(data);
    hasher.finalize(&mut output);
    output
}

pub fn action_hash(
    message_pack: &[u8],
    nonce: u64,
    vault: Option<&[u8; 20]>,
    expires_after: Option<u64>,
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(message_pack.len() + 38);
    bytes.extend_from_slice(message_pack);
    bytes.extend_from_slice(&nonce.to_be_bytes());
    if let Some(vault) = vault {
        bytes.push(1);
        bytes.extend_from_slice(vault);
    } else {
        bytes.push(0);
    }
    if let Some(expires_after) = expires_after {
        bytes.push(0);
        bytes.extend_from_slice(&expires_after.to_be_bytes());
    }
    keccak256(&bytes)
}

fn domain_separator() -> [u8; 32] {
    let type_hash = keccak256(
        b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
    );
    let name_hash = keccak256(b"Exchange");
    let version_hash = keccak256(b"1");
    let mut chain_id = [0_u8; 32];
    chain_id[24..].copy_from_slice(&1_337_u64.to_be_bytes());
    let mut encoded = Vec::with_capacity(160);
    encoded.extend_from_slice(&type_hash);
    encoded.extend_from_slice(&name_hash);
    encoded.extend_from_slice(&version_hash);
    encoded.extend_from_slice(&chain_id);
    encoded.extend_from_slice(&[0_u8; 32]);
    keccak256(&encoded)
}

fn agent_digest(connection_id: &[u8; 32], network: Network) -> [u8; 32] {
    let agent_type_hash = keccak256(b"Agent(string source,bytes32 connectionId)");
    let source_hash = keccak256(match network {
        Network::Mainnet => b"a",
        Network::Testnet => b"b",
    });
    let mut encoded = Vec::with_capacity(96);
    encoded.extend_from_slice(&agent_type_hash);
    encoded.extend_from_slice(&source_hash);
    encoded.extend_from_slice(connection_id);
    let struct_hash = keccak256(&encoded);
    let mut digest = Vec::with_capacity(66);
    digest.extend_from_slice(&[0x19, 0x01]);
    digest.extend_from_slice(&domain_separator());
    digest.extend_from_slice(&struct_hash);
    keccak256(&digest)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvmSignature {
    pub r: String,
    pub s: String,
    pub v: u8,
}

fn sign_connection_id(
    signing_key: &SigningKey,
    connection_id: &[u8; 32],
    network: Network,
) -> Result<EvmSignature> {
    let digest = agent_digest(connection_id, network);
    let (signature, recovery_id) = signing_key
        .sign_prehash_recoverable(&digest)
        .context("cannot sign Hyperliquid action digest")?;
    Ok(EvmSignature {
        r: format!("0x{}", hex::encode(signature.r().to_bytes())),
        s: format!("0x{}", hex::encode(signature.s().to_bytes())),
        v: 27 + recovery_id.to_byte(),
    })
}

pub struct SignedEnvelope {
    pub body: serde_json::Value,
    pub nonce: u64,
    pub expires_after: Option<u64>,
}

pub fn sign_envelope<A: Serialize>(
    action: &A,
    credentials: &HyperliquidCredentials,
    network: Network,
    nonce: u64,
    expires_after: Option<u64>,
) -> Result<SignedEnvelope> {
    let message_pack = rmp_serde::to_vec_named(action).context("cannot MessagePack action")?;
    let connection_id = action_hash(
        &message_pack,
        nonce,
        credentials.vault_bytes(),
        expires_after,
    );
    let signature = sign_connection_id(&credentials.signing_key(), &connection_id, network)?;
    let mut body = json!({
        "action": action,
        "nonce": nonce,
        "signature": signature,
    });
    if let Some(vault_address) = credentials.vault_address() {
        body["vaultAddress"] = json!(vault_address);
    }
    if let Some(expires_after) = expires_after {
        body["expiresAfter"] = json!(expires_after);
    }
    Ok(SignedEnvelope {
        body,
        nonce,
        expires_after,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    Alo,
    Ioc,
}

impl TimeInForce {
    const fn wire(self) -> &'static str {
        match self {
            Self::Alo => "Alo",
            Self::Ioc => "Ioc",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveOrderRequest {
    pub side: Side,
    pub px_units: i64,
    pub qty_units: i64,
    pub reduce_only: bool,
    pub time_in_force: TimeInForce,
    pub cloid: String,
}

#[derive(Serialize)]
pub(super) struct LimitWire<'a> {
    tif: &'a str,
}

#[derive(Serialize)]
pub(super) struct OrderTypeWire<'a> {
    limit: LimitWire<'a>,
}

#[derive(Serialize)]
pub(super) struct OrderWire<'a> {
    a: u32,
    b: bool,
    p: String,
    s: String,
    r: bool,
    t: OrderTypeWire<'a>,
    c: &'a str,
}

#[derive(Serialize)]
pub(super) struct OrderAction<'a> {
    #[serde(rename = "type")]
    type_: &'static str,
    orders: Vec<OrderWire<'a>>,
    grouping: &'static str,
}

pub(super) fn order_action<'a>(
    instrument: &InstrumentSpec,
    requests: &'a [LiveOrderRequest],
) -> Result<OrderAction<'a>> {
    if requests.is_empty() {
        bail!("an order action must contain at least one order");
    }
    let mut orders = Vec::with_capacity(requests.len());
    for request in requests {
        if request.px_units <= 0 || request.qty_units <= 0 {
            bail!("live order price and quantity must be positive");
        }
        let price_quantum = instrument.price_quantum(request.px_units);
        if request.px_units % price_quantum != 0 {
            bail!("live order price violates instrument significant-figure rules");
        }
        validate_cloid(&request.cloid)?;
        orders.push(OrderWire {
            a: instrument.asset_id,
            b: request.side == Side::Buy,
            p: format_fixed(request.px_units, instrument.max_price_decimals)?,
            s: format_fixed(request.qty_units, instrument.sz_decimals)?,
            r: request.reduce_only,
            t: OrderTypeWire {
                limit: LimitWire {
                    tif: request.time_in_force.wire(),
                },
            },
            c: &request.cloid,
        });
    }
    Ok(OrderAction {
        type_: "order",
        orders,
        grouping: "na",
    })
}

#[derive(Serialize)]
pub(super) struct CancelByCloidWire<'a> {
    asset: u32,
    cloid: &'a str,
}

#[derive(Serialize)]
pub(super) struct CancelByCloidAction<'a> {
    #[serde(rename = "type")]
    type_: &'static str,
    cancels: Vec<CancelByCloidWire<'a>>,
}

pub(super) fn cancel_by_cloid_action(
    asset_id: u32,
    cloids: &[String],
) -> Result<CancelByCloidAction<'_>> {
    if cloids.is_empty() {
        bail!("cancelByCloid must contain at least one CLOID");
    }
    let mut cancels = Vec::with_capacity(cloids.len());
    for cloid in cloids {
        validate_cloid(cloid)?;
        cancels.push(CancelByCloidWire {
            asset: asset_id,
            cloid,
        });
    }
    Ok(CancelByCloidAction {
        type_: "cancelByCloid",
        cancels,
    })
}

#[derive(Serialize)]
pub(super) struct CancelByOidWire {
    a: u32,
    o: u64,
}

#[derive(Serialize)]
pub(super) struct CancelByOidAction {
    #[serde(rename = "type")]
    type_: &'static str,
    cancels: Vec<CancelByOidWire>,
}

pub(super) fn cancel_by_oid_action(asset_id: u32, oids: &[u64]) -> Result<CancelByOidAction> {
    if oids.is_empty() {
        bail!("cancel must contain at least one OID");
    }
    Ok(CancelByOidAction {
        type_: "cancel",
        cancels: oids
            .iter()
            .map(|oid| CancelByOidWire {
                a: asset_id,
                o: *oid,
            })
            .collect(),
    })
}

pub fn make_cloid(session_started_at_ms: u64, quote_seq: u64, side: Side, sequence: u64) -> String {
    let mut digest = Sha256::new();
    digest.update(b"cashcat-cj-rust-v1");
    digest.update(session_started_at_ms.to_be_bytes());
    digest.update(quote_seq.to_be_bytes());
    digest.update([u8::from(side == Side::Buy)]);
    digest.update(sequence.to_be_bytes());
    let digest = digest.finalize();
    format!("0x{}", hex::encode(&digest[..16]))
}

fn validate_cloid(cloid: &str) -> Result<()> {
    let raw = cloid
        .strip_prefix("0x")
        .context("CLOID must start with 0x")?;
    if raw.len() != 32 || !raw.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("CLOID must contain exactly 16 hexadecimal bytes");
    }
    Ok(())
}

pub fn format_fixed(value: i64, decimals: u32) -> Result<String> {
    if value < 0 {
        bail!("wire decimal cannot be negative");
    }
    if decimals == 0 {
        return Ok(value.to_string());
    }
    let scale = 10_i64.pow(decimals);
    let integer = value / scale;
    let fractional = value % scale;
    if fractional == 0 {
        return Ok(integer.to_string());
    }
    let mut fractional = format!("{fractional:0width$}", width = decimals as usize);
    while fractional.ends_with('0') {
        fractional.pop();
    }
    Ok(format!("{integer}.{fractional}"))
}

pub fn parse_fixed(value: &str, decimals: u32) -> Result<i64> {
    let value = value.trim();
    let (negative, unsigned) = value
        .strip_prefix('-')
        .map_or((false, value), |unsigned| (true, unsigned));
    let (integer, fractional) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    let fractional = fractional.trim_end_matches('0');
    if integer.is_empty()
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || !fractional.bytes().all(|byte| byte.is_ascii_digit())
        || fractional.len() > decimals as usize
    {
        bail!("decimal {value:?} does not fit {decimals} places");
    }
    let scale = 10_i128.pow(decimals);
    let integer = integer
        .parse::<i128>()
        .context("decimal integer overflows")?;
    let fractional = if fractional.is_empty() {
        0
    } else {
        fractional
            .parse::<i128>()
            .context("invalid decimal fraction")?
            * 10_i128.pow(decimals - fractional.len() as u32)
    };
    let magnitude = integer
        .checked_mul(scale)
        .and_then(|value| value.checked_add(fractional))
        .context("fixed-point decimal overflows")?;
    let signed = if negative { -magnitude } else { magnitude };
    i64::try_from(signed).context("fixed-point decimal exceeds i64")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crypto_matches_independent_python_oracle_vectors() {
        assert_eq!(
            hex::encode(domain_separator()),
            "d79297fcdf2ffcd4ae223d01edaa2ba214ff8f401d7c9300d995d17c82aa4040"
        );
        let mut connection_id = [0_u8; 32];
        for (index, byte) in connection_id.iter_mut().enumerate() {
            *byte = index as u8;
        }
        assert_eq!(
            hex::encode(agent_digest(&connection_id, Network::Mainnet)),
            "b9e7c81cff512fa0969928e37d7c2475af657f1b314b7458c8dd7a023044cac0"
        );
        let mut key = [0_u8; 32];
        key[31] = 1;
        let signing_key = SigningKey::from_slice(&key).unwrap();
        let signature = sign_connection_id(&signing_key, &connection_id, Network::Mainnet).unwrap();
        assert_eq!(
            signature.r,
            "0x6fac96b099be7b8cdf3bc5ccec1b7966b2543f97941734f89f106b3f18e28d3b"
        );
        assert_eq!(
            signature.s,
            "0x1ab95dff17bf2a60a06d9471d504e174cabc3496da75c50c04fe21ddb30f13ef"
        );
        assert_eq!(signature.v, 28);
    }

    #[test]
    fn message_pack_and_vault_expiry_hashes_match_oracle() {
        let instrument = InstrumentSpec {
            symbol: "SYN".to_owned(),
            dex: String::new(),
            asset_id: 5,
            sz_decimals: 1,
            max_price_decimals: 5,
            max_significant_figures: 5,
            max_leverage: 3.0,
            minimum_notional: 10.0,
        };
        let requests = [LiveOrderRequest {
            side: Side::Buy,
            px_units: 12_345_000,
            qty_units: 5,
            reduce_only: false,
            time_in_force: TimeInForce::Ioc,
            cloid: "0x000102030405060708090a0b0c0d0e0f".to_owned(),
        }];
        let action = order_action(&instrument, &requests).unwrap();
        let packed = rmp_serde::to_vec_named(&action).unwrap();
        assert_eq!(
            format!("0x{}", hex::encode(&packed)),
            "0x83a474797065a56f72646572a66f72646572739187a16105a162c3a170a63132332e3435a173a3302e35a172c2a17481a56c696d697481a3746966a3496f63a163d92230783030303130323033303430353036303730383039306130623063306430653066a867726f7570696e67a26e61"
        );
        assert_eq!(
            hex::encode(action_hash(&packed, 1_234_567, None, None)),
            "dc76e4ee96d5ed33eae5fc25d0c3efd2ae108782b741627fd794be7b006eeba5"
        );
        assert_eq!(
            hex::encode(action_hash(
                &packed,
                1_234_567,
                Some(&[0x11_u8; 20]),
                Some(9_999_999)
            )),
            "e82fb2a10383cb80986147f11cb09de79436a1dd8c976f3ab29bf64678b8836f"
        );
    }

    #[test]
    fn fixed_point_wire_format_never_uses_float_or_exponent_notation() {
        assert_eq!(format_fixed(131_970, 6).unwrap(), "0.13197");
        assert_eq!(format_fixed(75, 0).unwrap(), "75");
        assert_eq!(format_fixed(12_300, 4).unwrap(), "1.23");
        assert_eq!(parse_fixed("0.131970", 6).unwrap(), 131_970);
        assert_eq!(parse_fixed("-75", 0).unwrap(), -75);
        assert_eq!(parse_fixed("88.0", 0).unwrap(), 88);
        assert!(parse_fixed("1.001", 2).is_err());
    }

    #[test]
    fn cloids_are_stable_unique_and_wire_valid() {
        let bid = make_cloid(1, 2, Side::Buy, 3);
        let ask = make_cloid(1, 2, Side::Sell, 3);
        assert_ne!(bid, ask);
        assert_eq!(bid.len(), 34);
        validate_cloid(&bid).unwrap();
    }
}
