use std::process::Command;

#[test]
fn live_command_fails_before_credentials_or_order_transport() {
    let config = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/cashcat.toml");
    let output = Command::new(env!("CARGO_BIN_EXE_mm-live"))
        .args(["--config", config.to_str().unwrap(), "live"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("live.enabled=false"));
    assert!(!stderr.contains("private key"));
}

#[test]
#[cfg(feature = "live-acceptance")]
fn live_canary_fails_before_network_or_credentials_without_explicit_confirmation() {
    let config = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/cashcat.toml");
    let output = Command::new(env!("CARGO_BIN_EXE_mm-live"))
        .args([
            "--config",
            config.to_str().unwrap(),
            "live-canary",
            "--credentials",
            "definitely-does-not-exist.env",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("before network or credential access"));
    assert!(!stderr.contains("definitely-does-not-exist"));
}

#[test]
fn flatten_command_uses_the_default_off_live_gate() {
    let config = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/cashcat.toml");
    let output = Command::new(env!("CARGO_BIN_EXE_mm-live"))
        .args(["--config", config.to_str().unwrap(), "live-flatten"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("live.enabled=false"));
}

#[test]
#[cfg(feature = "live-acceptance")]
fn acceptance_command_uses_the_default_off_live_gate() {
    let config = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/cashcat.toml");
    let output = Command::new(env!("CARGO_BIN_EXE_mm-live"))
        .args([
            "--config",
            config.to_str().unwrap(),
            "live-acceptance",
            "--phase",
            "verify",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("live.enabled=false"));
}

#[test]
#[cfg(not(feature = "live-acceptance"))]
fn production_binary_omits_real_account_acceptance_commands() {
    let output = Command::new(env!("CARGO_BIN_EXE_mm-live"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("live-canary"));
    assert!(!stdout.contains("live-acceptance"));
    assert!(!stdout.contains("live-gate-smoke"));
}
