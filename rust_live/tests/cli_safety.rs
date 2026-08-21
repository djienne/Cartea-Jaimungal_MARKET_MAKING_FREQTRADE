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
    assert!(stderr.contains("live order submission is intentionally unavailable"));
    assert!(!stderr.contains("private key"));
}
