use super::SemafusedEndpointConfig;
use super::http::{default_retries, default_timeout_secs, post_json, validate_transport};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Map, Value, json};

const OPENROUTER_PROTECTED_FIELDS: [&str; 4] = ["query", "documents", "model", "top_n"];

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RerankProtocol {
    Tei,
    Cohere,
    #[serde(rename = "openrouter")]
    OpenRouter,
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
    /// OpenRouter-specific fields, including the `provider` routing policy.
    #[serde(default)]
    extra_body: Map<String, Value>,
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
        if matches!(
            self.protocol,
            RerankProtocol::Cohere | RerankProtocol::OpenRouter
        ) {
            ensure!(
                self.model
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty()),
                "{label}.model is required for the {} protocol",
                match self.protocol {
                    RerankProtocol::Cohere => "cohere",
                    RerankProtocol::OpenRouter => "openrouter",
                    RerankProtocol::Tei => unreachable!(),
                }
            );
        }
        ensure!(
            matches!(self.protocol, RerankProtocol::OpenRouter) || self.extra_body.is_empty(),
            "{label}.extra_body is only supported for the openrouter protocol"
        );
        if matches!(self.protocol, RerankProtocol::OpenRouter) {
            validate_openrouter_extra_body(&self.extra_body, label)?;
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
            RerankProtocol::OpenRouter => {
                let model = self
                    .model
                    .as_deref()
                    .filter(|value| !value.trim().is_empty())
                    .context("reranker.model is required for the openrouter protocol")?;
                validate_openrouter_extra_body(&self.extra_body, "reranker")?;
                let mut body = self.extra_body.clone();
                body.extend(Map::from_iter([
                    ("model".into(), json!(model)),
                    ("query".into(), json!(query)),
                    ("documents".into(), json!(documents)),
                    ("top_n".into(), json!(documents.len())),
                ]));
                ("rerank", Value::Object(body))
            }
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
        let results = match self.protocol {
            RerankProtocol::Tei | RerankProtocol::Cohere => value
                .as_array()
                .or_else(|| value.get("results").and_then(Value::as_array))
                .context("reranker response must be an array or contain results")?,
            RerankProtocol::OpenRouter => value
                .get("results")
                .and_then(Value::as_array)
                .context("OpenRouter reranker response is missing results array")?,
        };
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
            let score = match self.protocol {
                RerankProtocol::OpenRouter => result.get("relevance_score"),
                RerankProtocol::Tei | RerankProtocol::Cohere => result
                    .get("score")
                    .or_else(|| result.get("relevance_score")),
            }
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

fn validate_openrouter_extra_body(extra_body: &Map<String, Value>, label: &str) -> Result<()> {
    for field in OPENROUTER_PROTECTED_FIELDS {
        ensure!(
            !extra_body.contains_key(field),
            "{label}.extra_body cannot override {field}"
        );
    }
    Ok(())
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
            Self::OpenaiCompatible { endpoint } => match endpoint.protocol {
                RerankProtocol::Tei => "tei",
                RerankProtocol::Cohere => "cohere",
                RerankProtocol::OpenRouter => "openrouter",
            },
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    #[tokio::test]
    async fn openrouter_protocol_sends_routing_fields_and_orders_response_scores() {
        let listener = match TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => listener,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(error) => panic!("bind mock OpenRouter server: {error}"),
        };
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4_096];
            let header_end = loop {
                let count = stream.read(&mut buffer).await.unwrap();
                assert_ne!(count, 0, "client closed before sending HTTP headers");
                request.extend_from_slice(&buffer[..count]);
                if let Some(position) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                    break position + 4;
                }
            };
            let headers = String::from_utf8_lossy(&request[..header_end]).into_owned();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            while request.len() < header_end + content_length {
                let count = stream.read(&mut buffer).await.unwrap();
                assert_ne!(count, 0, "client closed before sending HTTP body");
                request.extend_from_slice(&buffer[..count]);
            }
            let request_line = headers.lines().next().unwrap().to_owned();
            let body: Value =
                serde_json::from_slice(&request[header_end..header_end + content_length]).unwrap();

            let response = json!({
                "model": "cohere/rerank-v3.5",
                "provider": "Cohere",
                "results": [
                    {"index": 1, "relevance_score": 0.25},
                    {"index": 0, "relevance_score": 0.875}
                ]
            });
            let encoded = serde_json::to_vec(&response).unwrap();
            let response_headers = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                encoded.len()
            );
            stream.write_all(response_headers.as_bytes()).await.unwrap();
            stream.write_all(&encoded).await.unwrap();
            (request_line, body)
        });

        let endpoint: RerankEndpoint = serde_json::from_value(json!({
            "base_url": format!("http://{address}/api/v1"),
            "model": "cohere/rerank-v3.5",
            "protocol": "openrouter",
            "timeout_secs": 2,
            "max_retries": 0,
            "extra_body": {
                "provider": {
                    "order": ["cohere"],
                    "allow_fallbacks": false
                }
            }
        }))
        .unwrap();
        endpoint.validate("reranker").unwrap();
        let documents = vec!["alpha".to_owned(), "beta".to_owned()];
        let scores = endpoint
            .rerank("which document?", &documents)
            .await
            .unwrap();
        let (request_line, body) = server.await.unwrap();

        assert_eq!(request_line, "POST /api/v1/rerank HTTP/1.1");
        assert_eq!(body["model"], "cohere/rerank-v3.5");
        assert_eq!(body["query"], "which document?");
        assert_eq!(body["documents"], json!(["alpha", "beta"]));
        assert_eq!(body["top_n"], 2);
        assert_eq!(body["provider"]["order"], json!(["cohere"]));
        assert_eq!(body["provider"]["allow_fallbacks"], false);
        assert_eq!(scores, vec![0.875, 0.25]);
    }

    #[test]
    fn openrouter_extra_body_cannot_override_protocol_fields() {
        for field in OPENROUTER_PROTECTED_FIELDS {
            let mut extra_body = Map::new();
            extra_body.insert(field.to_owned(), Value::Null);
            let endpoint = RerankEndpoint {
                base_url: "https://openrouter.ai/api/v1".into(),
                model: Some("cohere/rerank-v3.5".into()),
                api_key_env: Some("OPENROUTER_API_KEY".into()),
                protocol: RerankProtocol::OpenRouter,
                timeout_secs: 1,
                max_retries: 0,
                extra_body,
            };
            let error = endpoint.validate("reranker").unwrap_err().to_string();
            assert!(
                error.contains(&format!("cannot override {field}")),
                "{error}"
            );
        }
    }

    #[test]
    fn openrouter_requires_model_and_has_explicit_identity() {
        let missing_model: RerankEndpoint = serde_json::from_value(json!({
            "base_url": "https://openrouter.ai/api/v1",
            "protocol": "openrouter",
            "api_key_env": "OPENROUTER_API_KEY"
        }))
        .unwrap();
        assert!(missing_model.validate("reranker").is_err());

        let backend: RerankBackend = serde_json::from_value(json!({
            "backend": "openai_compatible",
            "endpoint": {
                "base_url": "https://openrouter.ai/api/v1",
                "model": "cohere/rerank-v3.5",
                "protocol": "openrouter",
                "api_key_env": "OPENROUTER_API_KEY"
            }
        }))
        .unwrap();
        backend.validate("reranker").unwrap();
        assert_eq!(backend.backend_name(), "openrouter");
        assert_eq!(backend.model_identity(), Some("cohere/rerank-v3.5"));
    }

    #[test]
    fn extra_body_is_rejected_for_non_openrouter_protocols() {
        let endpoint: RerankEndpoint = serde_json::from_value(json!({
            "base_url": "http://127.0.0.1:8002",
            "protocol": "tei",
            "extra_body": {"provider": {"sort": "price"}}
        }))
        .unwrap();
        let error = endpoint.validate("reranker").unwrap_err().to_string();
        assert!(error.contains("only supported for the openrouter protocol"));
    }
}
