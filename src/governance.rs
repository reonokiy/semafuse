//! Portable archive export and local data-governance operations.

use crate::{blob::BlobStore, model::CanonicalArchive, storage::SqliteStore};
use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::Write,
    path::{Component, Path, PathBuf},
};

/// Summary of a complete portable archive directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExportReport {
    /// Final export directory.
    pub output: PathBuf,
    /// Raw native records written to `canonical-archive.json`.
    pub raw_records: usize,
    /// Canonical events written to `canonical-archive.json`.
    pub events: usize,
    /// Conversion diagnostics written to `canonical-archive.json`.
    pub diagnostics: usize,
    /// Unique verified attachment objects copied below `attachments/`.
    pub attachment_objects: usize,
}

/// Exports the canonical archive plus every referenced attachment to a new directory.
///
/// The destination is reserved with an atomic create operation and must not
/// already exist. An `incomplete` marker remains visible until attachment
/// digests and sizes have all been verified; ordinary errors remove the partial
/// directory, while a process crash leaves the marker for explicit diagnosis.
pub async fn export_directory(
    store: &SqliteStore,
    blobs: &dyn BlobStore,
    output: &Path,
) -> Result<ExportReport> {
    let _lifecycle_guard = store.lock_lifecycle()?;
    let archive = store.canonical_archive().await?;
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create export parent {}", parent.display()))?;
    create_private_export_dir(output).with_context(|| {
        format!(
            "reserve new export directory {}; existing paths are never overwritten",
            output.display()
        )
    })?;
    let marker = output.join(".semafuse-export-incomplete");
    write_private_file(&marker, b"Export did not complete.\n")?;
    let result = write_export(&archive, blobs, output).await;
    let attachment_objects = match result {
        Ok(count) => count,
        Err(error) => {
            let _ = std::fs::remove_dir_all(output);
            return Err(error);
        }
    };
    std::fs::remove_file(&marker)
        .with_context(|| format!("finalize export directory {}", output.display()))?;

    Ok(ExportReport {
        output: output.to_owned(),
        raw_records: archive.raw_records.len(),
        events: archive.events.len(),
        diagnostics: archive.diagnostics.len(),
        attachment_objects,
    })
}

#[cfg(unix)]
fn create_private_export_dir(path: &Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;

    let mut builder = std::fs::DirBuilder::new();
    builder.mode(0o700);
    builder.create(path)?;
    Ok(())
}

#[cfg(not(unix))]
fn create_private_export_dir(path: &Path) -> Result<()> {
    std::fs::create_dir(path)?;
    Ok(())
}

async fn write_export(
    archive: &CanonicalArchive,
    blobs: &dyn BlobStore,
    staging: &Path,
) -> Result<usize> {
    write_private_file(
        &staging.join("canonical-archive.json"),
        &serde_json::to_vec_pretty(archive)?,
    )?;

    let mut attachments = BTreeMap::new();
    for attachment in archive
        .events
        .iter()
        .flat_map(|event| event.attachment_refs())
    {
        if let Some((digest, size)) = attachments.insert(
            attachment.object_key.clone(),
            (attachment.digest.clone(), attachment.size),
        ) {
            anyhow::ensure!(
                digest == attachment.digest && size == attachment.size,
                "attachment key {} has conflicting metadata",
                attachment.object_key
            );
        }
    }
    if attachments.is_empty() {
        return Ok(0);
    }

    let root = staging.join("attachments");
    create_private_dir(&root)?;
    for (object_key, (expected_digest, expected_size)) in &attachments {
        let relative = safe_object_path(object_key)?;
        let bytes = blobs
            .get(object_key)
            .await
            .with_context(|| format!("read attachment object {object_key}"))?;
        anyhow::ensure!(
            bytes.len() as u64 == *expected_size,
            "attachment {object_key} size mismatch: expected {expected_size}, got {}",
            bytes.len()
        );
        let actual_digest = hex::encode(Sha256::digest(&bytes));
        anyhow::ensure!(
            actual_digest == *expected_digest,
            "attachment {object_key} digest mismatch"
        );
        let destination = root.join(relative);
        if let Some(parent) = destination.parent() {
            create_private_dir(parent)?;
        }
        write_private_file(&destination, &bytes)?;
    }
    Ok(attachments.len())
}

fn safe_object_path(object_key: &str) -> Result<PathBuf> {
    anyhow::ensure!(
        !object_key.is_empty(),
        "attachment object key cannot be empty"
    );
    anyhow::ensure!(
        object_key
            .split(['/', '\\'])
            .all(|part| !part.is_empty() && part != "." && part != ".."),
        "unsafe attachment object key {object_key:?}"
    );
    let path = Path::new(object_key);
    anyhow::ensure!(
        path.components()
            .all(|part| matches!(part, Component::Normal(_))),
        "unsafe attachment object key {object_key:?}"
    );
    Ok(path.to_owned())
}

#[cfg(unix)]
fn create_private_dir(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::create_dir_all(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)?;
    Ok(())
}

fn write_private_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("create export file {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::safe_object_path;

    #[test]
    fn attachment_object_paths_cannot_escape_export_root() {
        assert!(safe_object_path("sha256/ab/cd/digest").is_ok());
        assert!(safe_object_path("../secrets").is_err());
        assert!(safe_object_path("/absolute").is_err());
        assert!(safe_object_path("sha256/./digest").is_err());
    }
}
