//! Senet engine: exact rules (RULES.md), perfect indexing, value-iteration solver,
//! perfect-play database access, evaluators, bots and match tooling.

pub mod atomic;
pub mod board;
pub mod bots;
pub mod db;
pub mod eval;
pub mod game;
pub mod index;
pub mod manifest;
pub mod movegen;
pub mod net;
pub mod rng;
pub mod sha256;
pub mod solver;

pub use board::{Pos, Rules};
pub use movegen::{Kind, Move, MoveList, gen_moves, legal_moves};

/// x86-64 CPU features beyond the baseline: each with whether the compiler was allowed to
/// use it in this build and whether this CPU has it.
fn cpu_features() -> [(&'static str, bool, bool); 9] {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    macro_rules! present {
        ($name:tt) => {
            std::arch::is_x86_feature_detected!($name)
        };
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    macro_rules! present {
        ($name:tt) => {
            false
        };
    }
    macro_rules! features {
        ($($name:tt),*) => {
            [$(($name, cfg!(target_feature = $name), present!($name))),*]
        };
    }
    features!("popcnt", "sse4.2", "avx", "avx2", "fma", "bmi1", "bmi2", "lzcnt", "avx512f")
}

/// How this build was compiled: the version, the target, whether it is optimized, and
/// which of these x86-64 CPU features beyond the baseline the compiler was allowed to use.
/// A CPU without one of them cannot run the build (see `missing_cpu_features`). With
/// `bmi2`, `index::position_of` deposits bits with `pdep`.
pub fn build_info() -> serde_json::Value {
    let features: Vec<&str> = cpu_features().into_iter().filter_map(|(name, used, _)| used.then_some(name)).collect();
    serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "target": format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        "optimized": !cfg!(debug_assertions),
        "cpu_features": features,
    })
}

/// The CPU features this build may use (see `build_info`) that this CPU lacks, for a
/// program to check first and stop with a message rather than an illegal instruction.
/// Best effort: in a build that uses AVX, any code, this check included, may need it.
pub fn missing_cpu_features() -> Vec<&'static str> {
    cpu_features().into_iter().filter_map(|(name, used, present)| (used && !present).then_some(name)).collect()
}

/// A file whose contents are not what they should be.
pub(crate) fn invalid_data(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into())
}

/// An argument the function does not accept.
pub(crate) fn invalid_input(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, msg.into())
}

#[cfg(test)]
mod tests {
    #[test]
    fn this_cpu_runs_this_build() {
        assert!(super::missing_cpu_features().is_empty(), "the tests run where they were built");
        let info = super::build_info();
        assert_eq!(info["optimized"], !cfg!(debug_assertions));
    }
}
