use std::path::{Path, PathBuf};

#[test]
fn rust_model_contains_only_the_cartea_jaimungal_policy() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_rs_files(&workspace.join("src"), &mut files);
    collect_rs_files(&workspace.join("crates"), &mut files);
    let forbidden = [
        "avellaneda",
        "stoikov",
        "reservation_price",
        "base_half_spread_bps",
        "inventory_skew_bps_per_lot",
        "model_half_spread",
        "pub gamma:",
        "pub sigma:",
    ];
    for file in files {
        let text = std::fs::read_to_string(&file).unwrap().to_ascii_lowercase();
        for term in forbidden {
            assert!(
                !text.contains(term),
                "obsolete model term {term:?} found in {}",
                file.display()
            );
        }
    }
}

fn collect_rs_files(directory: &Path, output: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_rs_files(&path, output);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            output.push(path);
        }
    }
}
