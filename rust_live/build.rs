use std::process::Command;

fn command_output(program: &str, arguments: &[&str], fallback: &str) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| fallback.to_owned())
}

fn main() {
    let opt_level = std::env::var("OPT_LEVEL").unwrap_or_else(|_| "unknown".to_owned());
    assert_eq!(
        opt_level, "3",
        "mm-live executable and tests require opt-level=3; use the workspace profiles"
    );

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_owned());
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_owned());
    let target_features =
        std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_else(|_| "unknown".to_owned());
    let rustc = command_output("rustc", &["--version"], "unknown");
    let revision = std::env::var("MM_GIT_REVISION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| command_output("git", &["rev-parse", "--short=12", "HEAD"], "unknown"));

    println!("cargo:rustc-env=MM_BUILD_PROFILE={profile}");
    println!("cargo:rustc-env=MM_OPT_LEVEL={opt_level}");
    println!("cargo:rustc-env=MM_BUILD_TARGET={target}");
    println!("cargo:rustc-env=MM_TARGET_FEATURES={target_features}");
    println!("cargo:rustc-env=MM_RUSTC_VERSION={rustc}");
    println!("cargo:rustc-env=MM_GIT_REVISION={revision}");
    if std::path::Path::new("../.git/HEAD").exists() {
        println!("cargo:rerun-if-changed=../.git/HEAD");
    }
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=MM_GIT_REVISION");
}
