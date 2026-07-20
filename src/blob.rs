//! Content-addressed binary attachment persistence.

use crate::{config::BlobConfig, model::AttachmentRef};
use anyhow::{Context, Result};
use async_trait::async_trait;
use opendal::Operator;
use sha2::{Digest, Sha256};

/// Asynchronous storage used for canonical event attachments.
#[async_trait]
pub trait BlobStore: Send + Sync {
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
}

/// A [`BlobStore`] backed by an OpenDAL operator.
///
/// Objects are keyed by SHA-256 digest and an existing object is not rewritten.
#[derive(Clone)]
pub struct OpenDalBlobStore {
    operator: Operator,
}

/// A non-persistent blob store used by dry-run imports to preserve conversion semantics.
pub struct DryRunBlobStore;

impl OpenDalBlobStore {
    /// Builds a backend from an OpenDAL scheme and service options.
    pub fn from_config(config: &BlobConfig) -> Result<Self> {
        Ok(Self {
            operator: Operator::via_iter(&config.scheme, config.config.clone())
                .with_context(|| format!("initialize OpenDAL {} blob store", config.scheme))?,
        })
    }

    /// Wraps a preconfigured OpenDAL operator.
    pub fn from_operator(operator: Operator) -> Self {
        Self { operator }
    }
}

#[async_trait]
impl BlobStore for OpenDalBlobStore {
    async fn put(
        &self,
        bytes: Vec<u8>,
        media_type: &str,
        name: Option<String>,
    ) -> Result<AttachmentRef> {
        let digest = hex::encode(Sha256::digest(&bytes));
        let key = format!("sha256/{}/{}/{}", &digest[..2], &digest[2..4], digest);
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
}

/// Extracts supported inline or local-file attachments into content-addressed storage.
///
/// Recognized sources include base64 bodies, base64 data URLs, and native local
/// image/file paths. Decoding, file reads, and backend writes are fail-closed:
/// callers do not persist a canonical event with a silently missing attachment.
pub async fn extract_attachments(
    value: &serde_json::Value,
    store: &dyn BlobStore,
) -> Result<Vec<AttachmentRef>> {
    let mut candidates = Vec::new();
    collect_candidates(value, &mut candidates);
    let mut out = Vec::new();
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
            BlobCandidate::Path {
                path,
                media_type,
                name,
            } => (
                std::fs::read(&path)
                    .with_context(|| format!("read attachment {}", path.display()))?,
                media_type,
                name.or_else(|| path.file_name().map(|s| s.to_string_lossy().into_owned())),
            ),
        };
        out.push(store.put(bytes, &media_type, name).await?);
    }
    Ok(out)
}

enum BlobCandidate {
    Base64 {
        data: String,
        media_type: String,
        name: Option<String>,
    },
    Path {
        path: std::path::PathBuf,
        media_type: String,
        name: Option<String>,
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
                    out.push(BlobCandidate::Path {
                        path: path.into(),
                        media_type: media.unwrap_or("application/octet-stream").to_owned(),
                        name: map
                            .get("name")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_owned),
                    });
                    consumed = true;
                } else if let Some(url) = source
                    .get("image_url")
                    .or_else(|| source.get("url"))
                    .or_else(|| map.get("image_url"))
                    .or_else(|| map.get("url"))
                    .and_then(serde_json::Value::as_str)
                    && let Some((header, data)) = url
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
