use anyhow::{bail, Context, Result};
use k256::ecdsa::SigningKey;
use std::path::Path;
use std::sync::Arc;
use zeroize::Zeroizing;

/// The only secret-bearing runtime object. It intentionally implements neither
/// `Debug` nor serialization, so ordinary logging and reports cannot expose the key.
pub struct HyperliquidCredentials {
    account: String,
    account_bytes: [u8; 20],
    signing_key: Arc<SigningKey>,
    is_vault: bool,
}

impl HyperliquidCredentials {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            Zeroizing::new(std::fs::read_to_string(path).with_context(|| {
                format!("cannot read Hyperliquid credentials {}", path.display())
            })?);
        let mut exchange = None;
        let mut account = None;
        let mut signing_key = None;
        let mut is_vault = None;
        for (line_number, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (raw_name, raw_value) = line.split_once('=').with_context(|| {
                format!(
                    "invalid credentials line {}: expected key=value",
                    line_number + 1
                )
            })?;
            let name = raw_name.trim();
            let value = trim_env_value(raw_value);
            match name {
                "exchange" => set_once(&mut exchange, value.to_owned(), name)?,
                "wallet_address" => set_once(&mut account, value.to_owned(), name)?,
                "private_key" => {
                    if signing_key.is_some() {
                        bail!("duplicate private_key in credentials file");
                    }
                    signing_key = Some(parse_signing_key(value)?);
                }
                "is_vault" => {
                    let parsed = value.parse::<bool>().with_context(|| {
                        format!("is_vault must be true or false, got {value:?}")
                    })?;
                    set_once(&mut is_vault, parsed, name)?;
                }
                _ => bail!("unknown Hyperliquid credential key {name:?}"),
            }
        }
        if exchange.as_deref() != Some("hyperliquid") {
            bail!("credentials exchange must be exactly 'hyperliquid'");
        }
        let account = account.context("credentials missing wallet_address")?;
        let account_bytes = parse_address(&account).context("wallet_address is invalid")?;
        let signing_key = Arc::new(signing_key.context("credentials missing private_key")?);
        let is_vault = is_vault.context("credentials missing is_vault")?;
        Ok(Self {
            account: address_hex(&account_bytes),
            account_bytes,
            signing_key,
            is_vault,
        })
    }

    pub fn account(&self) -> &str {
        &self.account
    }

    pub const fn account_bytes(&self) -> &[u8; 20] {
        &self.account_bytes
    }

    pub const fn is_vault(&self) -> bool {
        self.is_vault
    }

    pub fn vault_address(&self) -> Option<&str> {
        self.is_vault.then_some(self.account.as_str())
    }

    pub fn vault_bytes(&self) -> Option<&[u8; 20]> {
        self.is_vault.then_some(&self.account_bytes)
    }

    pub fn signing_key(&self) -> Arc<SigningKey> {
        self.signing_key.clone()
    }

    pub fn agent_address(&self) -> String {
        address_hex(&address_from_signing_key(&self.signing_key))
    }
}

fn set_once<T>(target: &mut Option<T>, value: T, name: &str) -> Result<()> {
    if target.replace(value).is_some() {
        bail!("duplicate {name} in credentials file");
    }
    Ok(())
}

fn trim_env_value(value: &str) -> &str {
    let value = value.trim();
    if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\'')))
    {
        &value[1..value.len() - 1]
    } else {
        value
    }
}

fn parse_signing_key(value: &str) -> Result<SigningKey> {
    let value = value.strip_prefix("0x").unwrap_or(value);
    if value.len() != 64 {
        bail!("private_key must contain exactly 32 bytes");
    }
    let mut bytes = Zeroizing::new([0_u8; 32]);
    hex::decode_to_slice(value, bytes.as_mut()).context("private_key must be hexadecimal")?;
    SigningKey::from_slice(bytes.as_ref()).context("private_key is not a valid secp256k1 key")
}

pub fn parse_address(value: &str) -> Result<[u8; 20]> {
    let value = value.strip_prefix("0x").unwrap_or(value);
    if value.len() != 40 {
        bail!("address must contain exactly 20 bytes");
    }
    let mut address = [0_u8; 20];
    hex::decode_to_slice(value, &mut address).context("address must be hexadecimal")?;
    Ok(address)
}

pub fn address_hex(address: &[u8; 20]) -> String {
    format!("0x{}", hex::encode(address))
}

pub fn address_from_signing_key(signing_key: &SigningKey) -> [u8; 20] {
    let encoded = signing_key.verifying_key().to_encoded_point(false);
    let hash = super::signing::keccak256(&encoded.as_bytes()[1..]);
    let mut address = [0_u8; 20];
    address.copy_from_slice(&hash[12..]);
    address
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY_ONE: &str = "0x0000000000000000000000000000000000000000000000000000000000000001";

    fn credential_file(body: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("hyperliquid.env");
        std::fs::write(&path, body).unwrap();
        (directory, path)
    }

    #[test]
    fn exact_four_key_passivbot_shape_loads_without_exposing_the_key() {
        let (_directory, path) = credential_file(&format!(
            "exchange=hyperliquid\nwallet_address=0x1111111111111111111111111111111111111111\nprivate_key={KEY_ONE}\nis_vault=true\n"
        ));
        let credentials = HyperliquidCredentials::load(&path).unwrap();
        assert_eq!(
            credentials.account(),
            "0x1111111111111111111111111111111111111111"
        );
        assert!(credentials.is_vault());
        assert_eq!(
            credentials.agent_address(),
            "0x7e5f4552091a69125d5dfcb7b8c2659029395bdf"
        );
    }

    #[test]
    fn unknown_or_duplicate_keys_fail_closed() {
        let (_directory, path) = credential_file(&format!(
            "exchange=hyperliquid\nwallet_address=0x1111111111111111111111111111111111111111\nprivate_key={KEY_ONE}\nis_vault=false\nextra=no\n"
        ));
        assert!(HyperliquidCredentials::load(&path).is_err());
    }
}
