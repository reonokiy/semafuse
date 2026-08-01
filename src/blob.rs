//! Content-addressed binary attachment persistence.

use crate::{config::BlobConfig, model::AttachmentRef, storage::SqliteStore};
use anyhow::{Context, Result};
use async_trait::async_trait;
use opendal::Operator;
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Asynchronous storage used for canonical event attachments.
#[async_trait]
pub trait BlobStore: Send + Sync {
    /// Returns whether durable forget rules reject an event before any blob I/O.
    async fn rejects_event(&self, _event: &crate::model::CanonicalEvent) -> Result<bool> {
        Ok(false)
    }
    /// Stores bytes and returns their durable reference.
    async fn put(
        &self,
        bytes: Vec<u8>,
        media_type: &str,
        name: Option<String>,
    ) -> Result<AttachmentRef>;
    /// Reads an object by the backend key in its [`AttachmentRef`].
    async fn get(&self, key: &str) -> Result<Vec<u8>>;
    /// Deletes an object; backend failures are returned to the caller.
    async fn delete(&self, key: &str) -> Result<()>;
    /// Stable fingerprint of the configured backend.
    fn backend_id(&self) -> &str;
    /// Whether this instance owns the namespace containing `key`.
    fn manages_key(&self, key: &str) -> bool;
}

/// A [`BlobStore`] backed by an OpenDAL operator.
///
/// Objects are keyed by SHA-256 digest and an existing object is not rewritten.
#[derive(Clone)]
pub struct OpenDalBlobStore {
    operator: Operator,
    fs_root: Option<std::path::PathBuf>,
    backend_id: String,
    managed_prefix: String,
    staging_store: Option<SqliteStore>,
}

/// A non-persistent blob store used by dry-run imports to preserve conversion semantics.
pub struct DryRunBlobStore;

/// One attachment object that could not be removed during garbage collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BlobGcFailure {
    /// Backend object key.
    pub object_key: String,
    /// Backend or queue update error.
    pub error: String,
}

/// Result of one bounded durable attachment garbage-collection pass.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct BlobGcReport {
    /// Objects successfully removed from the backend and queue.
    pub deleted: usize,
    /// Objects still queued after this pass.
    pub pending: usize,
    /// Per-object failures retained for retry.
    pub failures: Vec<BlobGcFailure>,
}

impl OpenDalBlobStore {
    /// Builds a backend from an OpenDAL scheme and service options.
    pub fn from_config(config: &BlobConfig) -> Result<Self> {
        let mut resolved = config.clone();
        let fs_root = if config.scheme == "fs" {
            let root = config
                .config
                .get("root")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(crate::config::default_blob_dir);
            secure_blob_root(&root)?;
            let root = root
                .canonicalize()
                .with_context(|| format!("resolve blob directory {}", root.display()))?;
            anyhow::ensure!(
                root.is_dir(),
                "blob root {} is not a directory",
                root.display()
            );
            let root_config = root
                .to_str()
                .with_context(|| format!("blob directory {} is not Unicode", root.display()))?;
            resolved.config.insert("root".into(), root_config.into());
            Some(root)
        } else {
            None
        };
        Ok(Self {
            operator: Operator::via_iter(&resolved.scheme, resolved.config.clone())
                .with_context(|| format!("initialize OpenDAL {} blob store", resolved.scheme))?,
            fs_root,
            backend_id: backend_fingerprint(&resolved),
            managed_prefix: String::new(),
            staging_store: None,
        })
    }

    /// Builds a backend isolated to one database and enables orphan tracking.
    pub async fn from_config_for_store(config: &BlobConfig, store: &SqliteStore) -> Result<Self> {
        let mut blobs = Self::from_config(config)?;
        store.bind_blob_backend(&blobs.backend_id).await?;
        blobs.managed_prefix = format!("semafuse/{}/", store.storage_id().await?);
        blobs.staging_store = Some(store.clone());
        Ok(blobs)
    }

    /// Wraps a preconfigured OpenDAL operator.
    pub fn from_operator(operator: Operator) -> Self {
        Self {
            operator,
            fs_root: None,
            backend_id: "preconfigured-operator".into(),
            managed_prefix: String::new(),
            staging_store: None,
        }
    }
}

fn backend_fingerprint(config: &BlobConfig) -> String {
    // Bind durable references to the backend's location, not to replaceable
    // authentication or tuning options. Otherwise routine credential rotation
    // makes a database reject the same bucket, endpoint, or filesystem root.
    let mut entries = config
        .config
        .iter()
        .filter(|(key, _)| is_backend_location_key(key))
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(right.0));
    let mut hasher = Sha256::new();
    hasher.update(config.scheme.as_bytes());
    for (key, value) in entries {
        hasher.update([0]);
        hasher.update(key.as_bytes());
        hasher.update([0]);
        hasher.update(value.as_bytes());
    }
    format!("{}:{}", config.scheme, hex::encode(hasher.finalize()))
}

fn is_backend_location_key(key: &str) -> bool {
    if is_credential_config_key(key) {
        return false;
    }
    let normalized = key.trim().to_ascii_lowercase().replace(['-', '.'], "_");
    if ["proxy", "cache", "temporary", "presign"]
        .into_iter()
        .any(|name| normalized.contains(name))
    {
        return false;
    }
    [
        "root",
        "prefix",
        "bucket",
        "bucket_id",
        "container",
        "container_id",
        "endpoint",
        "endpoints",
        "region",
        "zone",
        "host",
        "port",
        "server",
        "url",
        "base_url",
        "account",
        "account_name",
        "namespace",
        "namespace_id",
        "database",
        "database_id",
        "db",
        "table",
        "collection",
        "collection_id",
        "repository",
        "repo",
        "owner",
        "project",
        "project_id",
        "organization",
        "organization_id",
        "drive",
        "drive_id",
        "folder",
        "path",
        "filesystem",
        "file_system",
        "share",
        "volume",
        "cluster",
        "cluster_id",
        "cluster_file",
        "name_node",
        "user",
        "username",
        "index",
        "workspace",
    ]
    .into_iter()
    .any(|name| normalized == name || normalized.ends_with(&format!("_{name}")))
}

fn is_credential_config_key(key: &str) -> bool {
    let normalized = key.trim().to_ascii_lowercase().replace(['-', '.'], "_");
    normalized.contains("secret")
        || normalized.contains("password")
        || normalized.contains("token")
        || normalized.contains("credential")
        || normalized.contains("private_key")
        || normalized.contains("access_key")
        || normalized.contains("account_key")
        || normalized.contains("application_key")
        || normalized.contains("api_key")
        || normalized.contains("client_key")
        || normalized.contains("authorization")
        || normalized.contains("signature")
        || normalized.contains("certificate")
        || normalized.contains("service_account")
        || normalized.starts_with("oauth")
        || matches!(
            normalized.as_str(),
            "key"
                | "client_id"
                | "application_id"
                | "tenant_id"
                | "role_arn"
                | "external_id"
                | "profile"
        )
}

#[cfg(unix)]
fn secure_blob_root(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;

    if path.try_exists()? {
        return Ok(());
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut builder = std::fs::DirBuilder::new();
    builder.mode(0o700);
    match builder.create(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => {
            Err(error).with_context(|| format!("create private blob directory {}", path.display()))
        }
    }
}

#[cfg(not(unix))]
fn secure_blob_root(path: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(path)
        .with_context(|| format!("create blob directory {}", path.display()))
}

/// Drains at most `limit` durable attachment deletions.
///
/// Backend failures remain in SQLite with an incremented attempt count, so a
/// later `gc` run can retry without resurrecting forgotten event data.
pub async fn drain_blob_gc(
    store: &SqliteStore,
    blobs: &dyn BlobStore,
    limit: usize,
) -> Result<BlobGcReport> {
    anyhow::ensure!(
        (1..=10_000).contains(&limit),
        "GC limit must be between 1 and 10000"
    );
    let _guard = store.lock_lifecycle()?;
    let entries = store.claim_pending_blob_gc(limit).await?;
    let mut report = BlobGcReport::default();
    for entry in entries {
        if entry.backend_id != blobs.backend_id() || !blobs.manages_key(&entry.object_key) {
            let message = format!(
                "refusing deletion outside the bound backend/namespace (queued {}, configured {})",
                entry.backend_id,
                blobs.backend_id()
            );
            store
                .fail_blob_gc(
                    &entry.backend_id,
                    &entry.object_key,
                    &entry.claim_token,
                    &message,
                )
                .await?;
            report.failures.push(BlobGcFailure {
                object_key: entry.object_key,
                error: message,
            });
            continue;
        }
        if store.attachment_is_referenced(&entry.object_key).await? {
            store
                .complete_blob_gc(&entry.backend_id, &entry.object_key, &entry.claim_token)
                .await?;
            continue;
        }
        match blobs.delete(&entry.object_key).await {
            Ok(()) => {
                store
                    .complete_blob_gc(&entry.backend_id, &entry.object_key, &entry.claim_token)
                    .await?;
                report.deleted += 1;
            }
            Err(error) => {
                let message = format!("{error:#}");
                store
                    .fail_blob_gc(
                        &entry.backend_id,
                        &entry.object_key,
                        &entry.claim_token,
                        &message,
                    )
                    .await?;
                report.failures.push(BlobGcFailure {
                    object_key: entry.object_key,
                    error: message,
                });
            }
        }
    }
    report.pending = store.pending_blob_gc_count().await?;
    Ok(report)
}

#[async_trait]
impl BlobStore for OpenDalBlobStore {
    async fn rejects_event(&self, event: &crate::model::CanonicalEvent) -> Result<bool> {
        match &self.staging_store {
            Some(store) => store.event_is_tombstoned(event).await,
            None => Ok(false),
        }
    }

    async fn put(
        &self,
        bytes: Vec<u8>,
        media_type: &str,
        name: Option<String>,
    ) -> Result<AttachmentRef> {
        if let Some(root) = &self.fs_root {
            secure_blob_root(root)?;
        }
        let digest = hex::encode(Sha256::digest(&bytes));
        let key = format!(
            "{}sha256/{}/{}/{}",
            self.managed_prefix,
            &digest[..2],
            &digest[2..4],
            digest
        );
        if let Some(store) = &self.staging_store {
            store
                .register_blob_candidate(&self.backend_id, &key)
                .await?;
        }
        if !self.operator.exists(&key).await? {
            self.operator.write(&key, bytes.clone()).await?;
        }
        Ok(AttachmentRef {
            object_key: key,
            digest,
            media_type: media_type.to_owned(),
            size: bytes.len() as u64,
            name,
        })
    }
    async fn get(&self, key: &str) -> Result<Vec<u8>> {
        Ok(self.operator.read(key).await?.to_vec())
    }
    async fn delete(&self, key: &str) -> Result<()> {
        self.operator.delete(key).await?;
        Ok(())
    }
    fn backend_id(&self) -> &str {
        &self.backend_id
    }
    fn manages_key(&self, key: &str) -> bool {
        key.starts_with(&self.managed_prefix)
    }
}

#[async_trait]
impl BlobStore for DryRunBlobStore {
    async fn put(
        &self,
        bytes: Vec<u8>,
        media_type: &str,
        name: Option<String>,
    ) -> Result<AttachmentRef> {
        let digest = hex::encode(Sha256::digest(&bytes));
        Ok(AttachmentRef {
            object_key: format!("dry-run/sha256/{digest}"),
            digest,
            media_type: media_type.to_owned(),
            size: bytes.len() as u64,
            name,
        })
    }

    async fn get(&self, key: &str) -> Result<Vec<u8>> {
        anyhow::bail!("dry-run blob {key} was not persisted")
    }

    async fn delete(&self, key: &str) -> Result<()> {
        anyhow::bail!("dry-run blob {key} was not persisted")
    }
    fn backend_id(&self) -> &str {
        "dry-run"
    }
    fn manages_key(&self, _key: &str) -> bool {
        false
    }
}

/// Extracts supported inline attachments into content-addressed storage.
///
/// Recognized sources include base64 bodies and base64 data URLs. Local path
/// references are rejected for both hook payloads and history imports, without
/// opening the referenced path. Decoding and backend writes are fail-closed:
/// callers do not persist an event with a missing attachment.
pub async fn extract_attachments(
    value: &serde_json::Value,
    store: &dyn BlobStore,
) -> Result<Vec<AttachmentRef>> {
    let mut candidates = Vec::new();
    collect_candidates(value, &mut candidates);
    let mut prepared = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let (bytes, media_type, name) = match candidate {
            BlobCandidate::Base64 {
                data,
                media_type,
                name,
            } => (
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data)
                    .context("decode base64 attachment")?,
                media_type,
                name,
            ),
            BlobCandidate::ExternalReference { reference } => anyhow::bail!(
                "external attachment reference {reference:?} is disabled; use inline base64 data instead"
            ),
        };
        prepared.push((bytes, media_type, name));
    }
    let mut out = Vec::with_capacity(prepared.len());
    for (bytes, media_type, name) in prepared {
        match store.put(bytes, &media_type, name).await {
            Ok(attachment) => out.push(attachment),
            Err(error) => {
                // Never delete earlier content-addressed puts here: a key may
                // already be referenced by another event. Database-bound
                // stores register writes before I/O, so delayed GC can reclaim
                // only candidates that remain unreferenced.
                return Err(error).context("store attachment");
            }
        }
    }
    Ok(out)
}

enum BlobCandidate {
    Base64 {
        data: String,
        media_type: String,
        name: Option<String>,
    },
    ExternalReference {
        reference: String,
    },
}

fn collect_candidates(value: &serde_json::Value, out: &mut Vec<BlobCandidate>) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                collect_candidates(value, out);
            }
        }
        serde_json::Value::Object(map) => {
            let media = map
                .get("media_type")
                .or_else(|| map.get("mime_type"))
                .and_then(serde_json::Value::as_str);
            let is_binary = map
                .get("type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|t| {
                    matches!(
                        t,
                        "image" | "input_image" | "localImage" | "imageView" | "file"
                    )
                })
                || media.is_some_and(|m| {
                    m.starts_with("image/") || m.starts_with("audio/") || m == "application/pdf"
                });
            if is_binary {
                let source = map.get("source").unwrap_or(value);
                let mut consumed = false;
                if let Some(data) = source.get("data").and_then(serde_json::Value::as_str) {
                    let media_type = source
                        .get("media_type")
                        .and_then(serde_json::Value::as_str)
                        .or(media)
                        .unwrap_or("application/octet-stream")
                        .to_owned();
                    let name = map
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned);
                    out.push(BlobCandidate::Base64 {
                        data: data.to_owned(),
                        media_type,
                        name,
                    });
                    consumed = true;
                } else if let Some(path) = source
                    .get("path")
                    .or_else(|| source.get("file_path"))
                    .and_then(serde_json::Value::as_str)
                {
                    out.push(BlobCandidate::ExternalReference {
                        reference: path.to_owned(),
                    });
                    consumed = true;
                } else if let Some(url) = source
                    .get("image_url")
                    .or_else(|| source.get("url"))
                    .or_else(|| map.get("image_url"))
                    .or_else(|| map.get("url"))
                    .and_then(serde_json::Value::as_str)
                {
                    if let Some((header, data)) = url
                        .strip_prefix("data:")
                        .and_then(|value| value.split_once(','))
                    {
                        out.push(BlobCandidate::Base64 {
                            data: data.to_owned(),
                            media_type: header
                                .split(';')
                                .next()
                                .unwrap_or("application/octet-stream")
                                .to_owned(),
                            name: map
                                .get("name")
                                .and_then(serde_json::Value::as_str)
                                .map(str::to_owned),
                        });
                    } else {
                        out.push(BlobCandidate::ExternalReference {
                            reference: url.to_owned(),
                        });
                    }
                    consumed = true;
                }
                if consumed {
                    return;
                }
            }
            for child in map.values() {
                collect_candidates(child, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{DryRunBlobStore, OpenDalBlobStore, backend_fingerprint, extract_attachments};
    use crate::{config::BlobConfig, storage::SqliteStore};
    use serde_json::json;
    use std::collections::HashMap;

    #[tokio::test]
    async fn backend_identity_tracks_location_but_not_rotating_credentials() {
        let config = |access_key: &str, secret: &str, bucket: &str, endpoint: &str| BlobConfig {
            scheme: "s3".into(),
            config: HashMap::from([
                ("root".into(), "/conversation-memory".into()),
                ("bucket".into(), bucket.into()),
                ("endpoint".into(), endpoint.into()),
                ("region".into(), "eu-central-1".into()),
                ("access_key_id".into(), access_key.into()),
                ("secret_access_key".into(), secret.into()),
                ("session_token".into(), format!("token-{access_key}")),
            ]),
        };
        let original = config("old-access", "old-secret", "memory", "https://s3.example");
        let mut rotated = config("new-access", "new-secret", "memory", "https://s3.example");
        rotated
            .config
            .insert("enable_virtual_host_style".into(), "true".into());
        let mut other_root = rotated.clone();
        other_root
            .config
            .insert("root".into(), "/other-memory".into());
        let other_bucket = config(
            "new-access",
            "new-secret",
            "other-memory",
            "https://s3.example",
        );
        let other_endpoint = config(
            "new-access",
            "new-secret",
            "memory",
            "https://other-s3.example",
        );

        let original_id = backend_fingerprint(&original);
        let rotated_id = backend_fingerprint(&rotated);
        assert_eq!(original_id, rotated_id);
        assert_ne!(original_id, backend_fingerprint(&other_root));
        assert_ne!(original_id, backend_fingerprint(&other_bucket));
        assert_ne!(original_id, backend_fingerprint(&other_endpoint));

        let store = SqliteStore::open_memory().await.unwrap();
        store.bind_blob_backend(&original_id).await.unwrap();
        store.bind_blob_backend(&rotated_id).await.unwrap();
        assert!(
            store
                .bind_blob_backend(&backend_fingerprint(&other_bucket))
                .await
                .is_err()
        );
    }

    #[test]
    fn filesystem_backend_identity_canonicalizes_relative_roots() {
        let cwd = std::env::current_dir().unwrap();
        let temp = tempfile::tempdir_in(&cwd).unwrap();
        let relative_root = temp.path().strip_prefix(&cwd).unwrap().join("blobs");
        let absolute_root = cwd.join(&relative_root);
        let config = |root: &std::path::Path| BlobConfig {
            scheme: "fs".into(),
            config: HashMap::from([("root".into(), root.display().to_string())]),
        };

        let relative = OpenDalBlobStore::from_config(&config(&relative_root)).unwrap();
        let absolute = OpenDalBlobStore::from_config(&config(&absolute_root)).unwrap();
        let canonical_root = absolute_root.canonicalize().unwrap();

        assert_eq!(relative.backend_id, absolute.backend_id);
        assert_eq!(relative.fs_root.as_deref(), Some(canonical_root.as_path()));
        assert_eq!(absolute.fs_root.as_deref(), Some(canonical_root.as_path()));
    }

    #[tokio::test]
    async fn local_path_attachments_are_rejected_without_filesystem_access() {
        for value in [
            json!({
                "type": "localImage",
                "path": "relative/secret.png",
                "media_type": "image/png"
            }),
            json!({
                "type": "localImage",
                "path": "/absolute/secret.png",
                "media_type": "image/png"
            }),
            json!({
                "type": "file",
                "url": "file:///absolute/secret.png",
                "media_type": "image/png"
            }),
        ] {
            let error = extract_attachments(&value, &DryRunBlobStore)
                .await
                .unwrap_err();
            assert!(
                error.to_string().contains("external attachment reference")
                    && error.to_string().contains("is disabled")
            );
        }
    }
}
