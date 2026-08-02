use super::SemafusedEndpointConfig;
use super::http::{default_retries, default_timeout_secs, post_json, validate_transport};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RerankProtocol {
    Tei,
    Cohere,
}

fn default_rerank_protocol() -> RerankProtocol {
    RerankProtocol::Tei
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RerankEndpoint {
    pub(crate) base_url: String,
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) api_key_env: Option<String>,
    #[serde(default = "default_rerank_protocol")]
    protocol: RerankProtocol,
    #[serde(default = "default_timeout_secs")]
    timeout_secs: u64,
    #[serde(default = "default_retries")]
    max_retries: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum RerankBackend {
    OpenaiCompatible { endpoint: RerankEndpoint },
    Semafused { endpoint: SemafusedEndpointConfig },
}

impl RerankEndpoint {
    pub(crate) fn validate(&self, label: &str) -> Result<()> {
        validate_transport(
            label,
            &self.base_url,
            self.api_key_env.as_deref(),
            self.timeout_secs,
            self.max_retries,
        )?;
        if matches!(self.protocol, RerankProtocol::Cohere) {
            ensure!(
                self.model
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty()),
                "{label}.model is required for the cohere protocol"
            );
        }
        Ok(())
    }

    pub(crate) async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let (path, body) = match self.protocol {
            RerankProtocol::Tei => (
                "rerank",
                json!({
                    "query": query,
                    "texts": documents,
                    "truncate": true,
                    "raw_scores": false,
                    "return_text": false
                }),
            ),
            RerankProtocol::Cohere => (
                "rerank",
                json!({
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": documents.len(),
                    "return_documents": false
                }),
            ),
        };
        let value = post_json(
            &self.base_url,
            path,
            self.api_key_env.as_deref(),
            self.timeout_secs,
            self.max_retries,
            &body,
        )
        .await
        .context("reranker request failed")?;
        let results = value
            .as_array()
            .or_else(|| value.get("results").and_then(Value::as_array))
            .context("reranker response must be an array or contain results")?;
        ensure!(
            results.len() == documents.len(),
            "reranker returned {} results for {} documents",
            results.len(),
            documents.len()
        );
        let mut scores = vec![None; documents.len()];
        for result in results {
            let index = result
                .get("index")
                .and_then(Value::as_u64)
                .context("reranker result is missing a non-negative index")?
                as usize;
            ensure!(
                index < documents.len(),
                "reranker index {index} is out of range"
            );
            ensure!(scores[index].is_none(), "duplicate reranker index {index}");
            let score = result
                .get("score")
                .or_else(|| result.get("relevance_score"))
                .and_then(Value::as_f64)
                .context("reranker result is missing a numeric score")?;
            ensure!(score.is_finite(), "reranker returned a non-finite score");
            scores[index] = Some(score);
        }
        scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| score.with_context(|| format!("missing reranker index {index}")))
            .collect()
    }
}

impl RerankBackend {
    pub(crate) fn validate(&self, label: &str) -> Result<()> {
        match self {
            Self::OpenaiCompatible { endpoint } => endpoint.validate(label),
            Self::Semafused { endpoint } => endpoint.validate(label),
        }
    }

    pub(crate) fn backend_name(&self) -> &'static str {
        match self {
            Self::OpenaiCompatible { .. } => "openai_compatible",
            Self::Semafused { .. } => "semafused",
        }
    }

    pub(crate) fn model_identity(&self) -> Option<&str> {
        match self {
            Self::OpenaiCompatible { endpoint } => endpoint.model.as_deref(),
            Self::Semafused { endpoint } => Some(endpoint.model_fingerprint()),
        }
    }

    pub(crate) async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>> {
        match self {
            Self::OpenaiCompatible { endpoint } => endpoint.rerank(query, documents).await,
            Self::Semafused { endpoint } => endpoint.rerank(query, documents).await,
        }
    }
}
