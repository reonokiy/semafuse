mod codex;
mod http;
mod openai;
mod processor;
mod reranker;
mod structured;

use crate::daemon_ipc::ModelServiceClient;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::{path::PathBuf, time::Duration};

#[cfg(feature = "benchmarks")]
pub(crate) use openai::OpenAiEndpoint;
pub(crate) use openai::{EmbeddingBackend, GenerationOptions};
pub(crate) use processor::PromptProcessorBackend;
pub(crate) use reranker::RerankBackend;
pub(crate) use structured::JsonContract;

pub(crate) const MAX_PROCESSOR_COMPRESSED_CONTENT_CHARS: usize = 512;
pub(crate) const MAX_PROCESSOR_REASON_CHARS: usize = 256;
const DEFAULT_SEMAFUSED_CONNECT_TIMEOUT_MS: u64 = 1_000;
const DEFAULT_SEMAFUSED_REQUEST_TIMEOUT_MS: u64 = 120_000;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SemafusedEndpointConfig {
    socket_path: PathBuf,
    model_fingerprint: String,
    #[serde(default = "default_semafused_connect_timeout_ms")]
    connect_timeout_ms: u64,
    #[serde(default = "default_semafused_request_timeout_ms")]
    request_timeout_ms: u64,
}

impl SemafusedEndpointConfig {
    fn endpoint(&self) -> ModelServiceClient {
        ModelServiceClient::new(
            self.socket_path.clone(),
            self.model_fingerprint.clone(),
            Duration::from_millis(self.connect_timeout_ms),
            Duration::from_millis(self.request_timeout_ms),
        )
    }

    pub(crate) fn validate(&self, label: &str) -> Result<()> {
        self.endpoint()
            .validate()
            .with_context(|| format!("invalid {label} semafused endpoint"))
    }

    pub(crate) fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }

    pub(crate) async fn embed(&self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        self.endpoint()
            .embed(input)
            .await
            .context("semafused embedding request failed")
    }

    pub(crate) async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>> {
        self.endpoint()
            .rerank(query, documents)
            .await
            .context("semafused reranker request failed")
    }
}

fn default_semafused_connect_timeout_ms() -> u64 {
    DEFAULT_SEMAFUSED_CONNECT_TIMEOUT_MS
}

fn default_semafused_request_timeout_ms() -> u64 {
    DEFAULT_SEMAFUSED_REQUEST_TIMEOUT_MS
}
