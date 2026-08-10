use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf};
use walkdir::WalkDir;

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let mut paths = WalkDir::new(root.join("src"))
        .into_iter()
        .map(|entry| entry.expect("walk source tree"))
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect::<Vec<_>>();
    paths.extend([root.join("build.rs"), root.join("Cargo.toml")]);
    paths.sort();

    let mut digest = Sha256::new();
    for path in paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(&root).expect("source path below root");
        let relative = relative.to_string_lossy();
        let bytes = fs::read(&path).unwrap_or_else(|error| {
            panic!("read benchmark identity source {}: {error}", path.display())
        });
        digest.update((relative.len() as u64).to_be_bytes());
        digest.update(relative.as_bytes());
        digest.update((bytes.len() as u64).to_be_bytes());
        digest.update(bytes);
    }
    println!(
        "cargo:rustc-env=SEMAFUSE_SOURCE_SHA256={}",
        hex::encode(digest.finalize())
    );
}
