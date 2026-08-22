use anyhow::{bail, Result};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuildInfo {
    pub profile: &'static str,
    pub opt_level: &'static str,
    pub target: &'static str,
    pub target_features: &'static str,
    pub rustc: &'static str,
    pub revision: &'static str,
}

impl BuildInfo {
    pub const fn current() -> Self {
        Self {
            profile: env!("MM_BUILD_PROFILE"),
            opt_level: env!("MM_OPT_LEVEL"),
            target: env!("MM_BUILD_TARGET"),
            target_features: env!("MM_TARGET_FEATURES"),
            rustc: env!("MM_RUSTC_VERSION"),
            revision: env!("MM_GIT_REVISION"),
        }
    }

    pub fn ensure_optimized(self) -> Result<()> {
        if self.opt_level != "3" {
            bail!(
                "refusing to run an unoptimized binary (profile={}, opt-level={})",
                self.profile,
                self.opt_level
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_build_contract_is_optimized() {
        let info = BuildInfo::current();
        assert_eq!(info.opt_level, "3");
        info.ensure_optimized().unwrap();
        assert!(!info.target.is_empty());
        assert!(!info.rustc.is_empty());
    }
}
