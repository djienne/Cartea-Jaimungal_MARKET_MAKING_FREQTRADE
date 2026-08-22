#[test]
fn hot_model_crate_has_no_cold_runtime_dependencies() {
    let manifest = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .unwrap()
        .to_ascii_lowercase();
    for forbidden in ["tokio", "arrow", "parquet", "reqwest", "redb", "serde_json"] {
        assert!(
            !manifest.contains(forbidden),
            "cold dependency {forbidden:?} leaked into cj-core"
        );
    }
}
