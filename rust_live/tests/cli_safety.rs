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

/// The grid runs N parameter sets and must never be a path to real money.
///
/// The live-money configuration is a single explicit config, never a grid, so
/// this asserts the grid does not become a back door: it must not require the
/// live gate, must not read credentials, and must refuse a spec whose variants
/// are not independently valid.
#[test]
fn dry_run_grid_never_touches_the_live_path() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // A config with live.enabled = TRUE. The grid must still not arm anything:
    // it is a dry-run command and never constructs the live backend.
    let live_config = dir.join("config/cashcat_live_test.toml");
    let grid = dir.join("config/grid_cashcat.toml");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_mm-live"))
        .args([
            "--config",
            live_config.to_str().unwrap(),
            "dry-run-grid",
            "--grid",
            grid.to_str().unwrap(),
            "--duration-seconds",
            "1",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Whatever happens (it may fail for lack of market data in CI), it must
    // never have reached credentials or an order transport.
    assert!(
        !stderr.contains("private key") && !stdout.contains("private key"),
        "grid must not read credentials: {stderr}"
    );
    assert!(
        !stderr.contains("agent address") && !stdout.contains("agent address"),
        "grid must not derive an agent address: {stderr}"
    );
}

#[test]
fn dry_run_grid_refuses_a_spec_whose_variant_is_invalid() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let config = dir.join("config/cashcat.toml");
    let temporary = tempfile::tempdir().unwrap();
    let spec = temporary.path().join("bad.toml");
    // A hold window wider than the widest permitted quote: valid TOML, invalid
    // configuration. It must fail at startup, not hours into a run.
    std::fs::write(
        &spec,
        "[[variant]]
name = \"bad\"
replace_threshold_bps = 10000.0
",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_mm-live"))
        .args([
            "--config",
            config.to_str().unwrap(),
            "dry-run-grid",
            "--grid",
            spec.to_str().unwrap(),
            "--duration-seconds",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("bad") || stderr.contains("replace_threshold_bps"),
        "expected the invalid variant to be named: {stderr}"
    );
}
