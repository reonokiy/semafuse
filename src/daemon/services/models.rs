use std::sync::Arc;

use crate::daemon_ipc::protocol::{
    ErrorCode, IndexedEmbedding, IndexedScore, MAX_EMBEDDING_DIMENSIONS, MAX_EMBEDDING_VALUES,
    MAX_FINGERPRINT_BYTES, MAX_ITEMS, MAX_TEXT_BYTES, MAX_TOTAL_TEXT_BYTES, ModelRequest,
    ModelResponse, ModelServiceStatus, ProtocolError, ServiceResponse, ServiceState,
};
use async_trait::async_trait;
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore},
    task,
};

use crate::daemon::{
    models::{EmbeddingEngine, ModelEngineError, ModelEngines, RerankerEngine},
    router::{Service, ServiceRequest, protocol_error},
};

pub struct ModelsService {
    embedding: Option<Arc<dyn EmbeddingEngine>>,
    reranker: Option<Arc<dyn RerankerEngine>>,
    inference_permits: Arc<Semaphore>,
    queue_permits: Arc<Semaphore>,
}

impl ModelsService {
    pub fn new(engines: ModelEngines, max_concurrent_inferences: usize) -> Result<Self, String> {
        if engines.is_empty() {
            return Err("models service requires at least one engine".to_owned());
        }
        if max_concurrent_inferences == 0 {
            return Err("models service concurrency must be greater than zero".to_owned());
        }
        let max_queued_inferences = max_concurrent_inferences
            .checked_mul(2)
            .filter(|capacity| *capacity <= Semaphore::MAX_PERMITS)
            .ok_or_else(|| "models service concurrency is too large".to_owned())?;

        Ok(Self {
            embedding: engines.embedding,
            reranker: engines.reranker,
            inference_permits: Arc::new(Semaphore::new(max_concurrent_inferences)),
            queue_permits: Arc::new(Semaphore::new(max_queued_inferences)),
        })
    }

    /// Admit at most two waiting requests per active inference slot. Once
    /// admitted, a request waits fairly for an inference permit instead of
    /// failing just because another request won a short scheduling race.
    async fn acquire_inference_permit(&self) -> Result<OwnedSemaphorePermit, ProtocolError> {
        let queue_permit = self
            .queue_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| {
                protocol_error(ErrorCode::Overloaded, "model inference queue is full", true)
            })?;
        let inference_permit = self
            .inference_permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| {
                protocol_error(
                    ErrorCode::Internal,
                    "model inference scheduler is unavailable",
                    false,
                )
            })?;
        drop(queue_permit);
        Ok(inference_permit)
    }

    async fn embed(
        &self,
        expected_model_fingerprint: String,
        input: Vec<String>,
    ) -> Result<ServiceResponse, ProtocolError> {
        validate_fingerprint(&expected_model_fingerprint)?;
        validate_texts(&input, "input")?;

        let engine = self.embedding.clone().ok_or_else(|| {
            protocol_error(
                ErrorCode::NotReady,
                "embedding engine is not configured",
                false,
            )
        })?;
        require_fingerprint(
            engine.identity().fingerprint.as_str(),
            &expected_model_fingerprint,
        )?;
        let fingerprint = engine.identity().fingerprint.clone();
        let declared_dimensions = engine.identity().dimensions;

        let permit = self.acquire_inference_permit().await?;
        let expected_count = input.len();
        let vectors = task::spawn_blocking(move || {
            let _permit = permit;
            engine.embed(&input)
        })
        .await
        .map_err(|error| {
            protocol_error(
                ErrorCode::InferenceFailed,
                format!("embedding worker failed: {error}"),
                false,
            )
        })?
        .map_err(map_engine_error)?;

        if vectors.len() != expected_count {
            return Err(protocol_error(
                ErrorCode::InferenceFailed,
                format!(
                    "embedding engine returned {} vectors for {expected_count} inputs",
                    vectors.len()
                ),
                false,
            ));
        }

        let mut total_values = 0_usize;
        let mut batch_dimensions = None;
        let mut data = Vec::with_capacity(vectors.len());
        for (index, embedding) in vectors.into_iter().enumerate() {
            let dimensions = embedding.len();
            if dimensions == 0 || dimensions > MAX_EMBEDDING_DIMENSIONS {
                return Err(protocol_error(
                    ErrorCode::InferenceFailed,
                    format!("embedding {index} has invalid dimension {dimensions}"),
                    false,
                ));
            }
            if let Some(declared) = declared_dimensions
                && dimensions != declared
            {
                return Err(protocol_error(
                    ErrorCode::InferenceFailed,
                    format!(
                        "embedding {index} has dimension {dimensions}; model identity declares {declared}"
                    ),
                    false,
                ));
            }
            match batch_dimensions {
                Some(expected) if expected != dimensions => {
                    return Err(protocol_error(
                        ErrorCode::InferenceFailed,
                        format!(
                            "embedding {index} has dimension {dimensions}; the first embedding has dimension {expected}"
                        ),
                        false,
                    ));
                }
                None => batch_dimensions = Some(dimensions),
                _ => {}
            }
            if embedding.iter().any(|value| !value.is_finite()) {
                return Err(protocol_error(
                    ErrorCode::InferenceFailed,
                    format!("embedding {index} contains a non-finite value"),
                    false,
                ));
            }
            total_values = total_values.saturating_add(dimensions);
            if total_values > MAX_EMBEDDING_VALUES {
                return Err(protocol_error(
                    ErrorCode::InferenceFailed,
                    "embedding response exceeds the protocol value limit",
                    false,
                ));
            }
            data.push(IndexedEmbedding {
                index: index as u32,
                embedding,
            });
        }

        Ok(ServiceResponse::Models(ModelResponse::Embed {
            model_fingerprint: fingerprint,
            data,
        }))
    }

    async fn rerank(
        &self,
        expected_model_fingerprint: String,
        query: String,
        documents: Vec<String>,
    ) -> Result<ServiceResponse, ProtocolError> {
        validate_fingerprint(&expected_model_fingerprint)?;
        validate_text(&query, "query")?;
        validate_texts(&documents, "documents")?;
        if query.len().saturating_add(total_text_bytes(&documents)) > MAX_TOTAL_TEXT_BYTES {
            return Err(protocol_error(
                ErrorCode::InvalidRequest,
                "query and documents exceed the total text byte limit",
                false,
            ));
        }

        let engine = self.reranker.clone().ok_or_else(|| {
            protocol_error(
                ErrorCode::NotReady,
                "reranker engine is not configured",
                false,
            )
        })?;
        require_fingerprint(
            engine.identity().fingerprint.as_str(),
            &expected_model_fingerprint,
        )?;
        let fingerprint = engine.identity().fingerprint.clone();

        let permit = self.acquire_inference_permit().await?;
        let expected_count = documents.len();
        // RerankerEngine deliberately returns raw classifier logits. Keeping
        // calibration here gives every backend the same public [0, 1] score.
        let logits = task::spawn_blocking(move || {
            let _permit = permit;
            engine.rerank(&query, &documents)
        })
        .await
        .map_err(|error| {
            protocol_error(
                ErrorCode::InferenceFailed,
                format!("reranker worker failed: {error}"),
                false,
            )
        })?
        .map_err(map_engine_error)?;

        if logits.len() != expected_count {
            return Err(protocol_error(
                ErrorCode::InferenceFailed,
                format!(
                    "reranker engine returned {} logits for {expected_count} documents",
                    logits.len()
                ),
                false,
            ));
        }
        let mut results = Vec::with_capacity(logits.len());
        for (index, logit) in logits.into_iter().enumerate() {
            if !logit.is_finite() {
                return Err(protocol_error(
                    ErrorCode::InferenceFailed,
                    format!("reranker logit {index} is not finite"),
                    false,
                ));
            }
            let score = stable_sigmoid(logit);
            if !(0.0..=1.0).contains(&score) {
                return Err(protocol_error(
                    ErrorCode::InferenceFailed,
                    format!("reranker score {index} is outside [0, 1]"),
                    false,
                ));
            }
            results.push(IndexedScore {
                index: index as u32,
                score,
            });
        }

        Ok(ServiceResponse::Models(ModelResponse::Rerank {
            model_fingerprint: fingerprint,
            results,
        }))
    }
}

#[async_trait]
impl Service for ModelsService {
    fn name(&self) -> &'static str {
        "models"
    }

    fn state(&self) -> ServiceState {
        ServiceState::Ready
    }

    fn status(&self) -> serde_json::Value {
        serde_json::to_value(ModelServiceStatus {
            state: self.state(),
            service_version: env!("CARGO_PKG_VERSION").to_owned(),
            embedding: self
                .embedding
                .as_ref()
                .map(|engine| engine.identity().clone()),
            reranker: self
                .reranker
                .as_ref()
                .map(|engine| engine.identity().clone()),
        })
        .expect("ModelServiceStatus serialization cannot fail")
    }

    async fn call(&self, request: ServiceRequest) -> Result<ServiceResponse, ProtocolError> {
        match request {
            ServiceRequest::Models(ModelRequest::Embed {
                expected_model_fingerprint,
                input,
            }) => self.embed(expected_model_fingerprint, input).await,
            ServiceRequest::Models(ModelRequest::Rerank {
                expected_model_fingerprint,
                query,
                documents,
            }) => {
                self.rerank(expected_model_fingerprint, query, documents)
                    .await
            }
        }
    }
}

fn validate_fingerprint(fingerprint: &str) -> Result<(), ProtocolError> {
    if fingerprint.is_empty()
        || fingerprint.len() > MAX_FINGERPRINT_BYTES
        || !fingerprint.bytes().all(|byte| byte.is_ascii_graphic())
    {
        return Err(protocol_error(
            ErrorCode::InvalidRequest,
            format!(
                "expected_model_fingerprint must contain 1..={MAX_FINGERPRINT_BYTES} ASCII graphic bytes"
            ),
            false,
        ));
    }
    Ok(())
}

fn require_fingerprint(actual: &str, expected: &str) -> Result<(), ProtocolError> {
    if actual != expected {
        return Err(protocol_error(
            ErrorCode::ModelMismatch,
            format!("configured model fingerprint is {actual:?}, not {expected:?}"),
            false,
        ));
    }
    Ok(())
}

fn validate_texts(texts: &[String], field: &str) -> Result<(), ProtocolError> {
    if texts.is_empty() || texts.len() > MAX_ITEMS {
        return Err(protocol_error(
            ErrorCode::InvalidRequest,
            format!("{field} must contain 1..={MAX_ITEMS} items"),
            false,
        ));
    }
    for (index, text) in texts.iter().enumerate() {
        validate_text(text, &format!("{field}[{index}]"))?;
    }
    if total_text_bytes(texts) > MAX_TOTAL_TEXT_BYTES {
        return Err(protocol_error(
            ErrorCode::InvalidRequest,
            format!("{field} exceeds the total text byte limit"),
            false,
        ));
    }
    Ok(())
}

fn validate_text(text: &str, field: &str) -> Result<(), ProtocolError> {
    if text.is_empty() || text.len() > MAX_TEXT_BYTES {
        return Err(protocol_error(
            ErrorCode::InvalidRequest,
            format!("{field} must contain 1..={MAX_TEXT_BYTES} bytes"),
            false,
        ));
    }
    Ok(())
}

fn total_text_bytes(texts: &[String]) -> usize {
    texts
        .iter()
        .fold(0_usize, |total, text| total.saturating_add(text.len()))
}

fn stable_sigmoid(logit: f64) -> f64 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exponential = logit.exp();
        exponential / (1.0 + exponential)
    }
}

fn map_engine_error(error: ModelEngineError) -> ProtocolError {
    match error {
        ModelEngineError::InvalidInput(message) => {
            protocol_error(ErrorCode::InvalidRequest, message, false)
        }
        ModelEngineError::Inference(message) => {
            protocol_error(ErrorCode::InferenceFailed, message, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use crate::daemon_ipc::protocol::{ModelIdentity, RequestEnvelope, ResponseEnvelope};
    use tokio::sync::Barrier;

    use crate::daemon::router::ServiceRouter;

    use super::*;

    struct FakeEmbedding {
        identity: ModelIdentity,
        outputs: Option<Vec<Vec<f32>>>,
        delay: Duration,
    }

    impl EmbeddingEngine for FakeEmbedding {
        fn identity(&self) -> &ModelIdentity {
            &self.identity
        }

        fn embed(&self, input: &[String]) -> Result<Vec<Vec<f32>>, ModelEngineError> {
            thread::sleep(self.delay);
            Ok(self.outputs.clone().unwrap_or_else(|| {
                input
                    .iter()
                    .map(|text| vec![text.len() as f32, 1.0])
                    .collect()
            }))
        }
    }

    struct FakeReranker {
        identity: ModelIdentity,
        logits: Vec<f64>,
    }

    impl RerankerEngine for FakeReranker {
        fn identity(&self) -> &ModelIdentity {
            &self.identity
        }

        fn rerank(
            &self,
            _query: &str,
            _documents: &[String],
        ) -> Result<Vec<f64>, ModelEngineError> {
            Ok(self.logits.clone())
        }
    }

    fn embedding(
        dimensions: Option<usize>,
        outputs: Option<Vec<Vec<f32>>>,
        delay: Duration,
    ) -> Arc<dyn EmbeddingEngine> {
        Arc::new(FakeEmbedding {
            identity: ModelIdentity {
                fingerprint: "sha256:test".to_owned(),
                model: "fake-embedding".to_owned(),
                dimensions,
            },
            outputs,
            delay,
        })
    }

    fn reranker(logits: Vec<f64>) -> Arc<dyn RerankerEngine> {
        Arc::new(FakeReranker {
            identity: ModelIdentity {
                fingerprint: "sha256:reranker".to_owned(),
                model: "fake-reranker".to_owned(),
                dimensions: None,
            },
            logits,
        })
    }

    fn router_with(engines: ModelEngines, concurrency: usize) -> ServiceRouter {
        let service = ModelsService::new(engines, concurrency).unwrap();
        let mut router = ServiceRouter::new("test");
        router.register(Arc::new(service)).unwrap();
        router
    }

    fn router() -> ServiceRouter {
        router_with(
            ModelEngines {
                embedding: Some(embedding(Some(2), None, Duration::ZERO)),
                reranker: None,
            },
            1,
        )
    }

    fn embed_request(input: Vec<String>) -> RequestEnvelope {
        RequestEnvelope::Models(ModelRequest::Embed {
            expected_model_fingerprint: "sha256:test".to_owned(),
            input,
        })
    }

    #[tokio::test]
    async fn injected_embedding_engine_is_routed_and_indexed() {
        let response = router()
            .route(embed_request(vec!["a".to_owned(), "hello".to_owned()]))
            .await;
        let ResponseEnvelope::Ok {
            response:
                ServiceResponse::Models(ModelResponse::Embed {
                    model_fingerprint,
                    data,
                }),
        } = response
        else {
            panic!("unexpected response");
        };
        assert_eq!(model_fingerprint, "sha256:test");
        assert_eq!(data[0].index, 0);
        assert_eq!(data[0].embedding, vec![1.0, 1.0]);
        assert_eq!(data[1].index, 1);
        assert_eq!(data[1].embedding, vec![5.0, 1.0]);
    }

    #[tokio::test]
    async fn fingerprint_mismatch_is_explicit() {
        let response = router()
            .route(RequestEnvelope::Models(ModelRequest::Embed {
                expected_model_fingerprint: "sha256:wrong".to_owned(),
                input: vec!["hello".to_owned()],
            }))
            .await;
        let ResponseEnvelope::Error { error } = response else {
            panic!("expected error");
        };
        assert_eq!(error.code, ErrorCode::ModelMismatch);
    }

    #[test]
    fn fingerprints_are_nonempty_bounded_ascii_graphic_strings() {
        assert!(validate_fingerprint("sha256:test").is_ok());
        for invalid in ["", "sha256:has space", "sha256:\nnewline", "指纹"] {
            let error = validate_fingerprint(invalid).unwrap_err();
            assert_eq!(error.code, ErrorCode::InvalidRequest);
        }
        assert!(validate_fingerprint(&"a".repeat(MAX_FINGERPRINT_BYTES + 1)).is_err());
    }

    #[tokio::test]
    async fn empty_query_inputs_and_documents_are_rejected() {
        let response = router().route(embed_request(vec![String::new()])).await;
        let ResponseEnvelope::Error { error } = response else {
            panic!("expected empty embedding input to fail");
        };
        assert_eq!(error.code, ErrorCode::InvalidRequest);

        for (query, documents) in [
            (String::new(), vec!["document".to_owned()]),
            ("query".to_owned(), vec![String::new()]),
        ] {
            let response = router()
                .route(RequestEnvelope::Models(ModelRequest::Rerank {
                    expected_model_fingerprint: "sha256:reranker".to_owned(),
                    query,
                    documents,
                }))
                .await;
            let ResponseEnvelope::Error { error } = response else {
                panic!("expected empty reranker text to fail");
            };
            assert_eq!(error.code, ErrorCode::InvalidRequest);
        }
    }

    #[tokio::test]
    async fn embedding_dimensions_must_match_the_model_identity() {
        let router = router_with(
            ModelEngines {
                embedding: Some(embedding(
                    Some(3),
                    Some(vec![vec![1.0, 2.0]]),
                    Duration::ZERO,
                )),
                reranker: None,
            },
            1,
        );
        let response = router.route(embed_request(vec!["hello".to_owned()])).await;
        let ResponseEnvelope::Error { error } = response else {
            panic!("expected declared dimension mismatch to fail");
        };
        assert_eq!(error.code, ErrorCode::InferenceFailed);
        assert!(error.message.contains("identity declares 3"));
    }

    #[tokio::test]
    async fn embedding_dimensions_must_be_consistent_within_a_batch() {
        let router = router_with(
            ModelEngines {
                embedding: Some(embedding(
                    None,
                    Some(vec![vec![1.0], vec![1.0, 2.0]]),
                    Duration::ZERO,
                )),
                reranker: None,
            },
            1,
        );
        let response = router
            .route(embed_request(vec!["one".to_owned(), "two".to_owned()]))
            .await;
        let ResponseEnvelope::Error { error } = response else {
            panic!("expected inconsistent batch dimensions to fail");
        };
        assert_eq!(error.code, ErrorCode::InferenceFailed);
        assert!(error.message.contains("first embedding"));
    }

    #[tokio::test]
    async fn reranker_raw_logits_are_converted_with_a_stable_sigmoid() {
        let router = router_with(
            ModelEngines {
                embedding: None,
                reranker: Some(reranker(vec![-1_000.0, 0.0, 1_000.0])),
            },
            1,
        );
        let response = router
            .route(RequestEnvelope::Models(ModelRequest::Rerank {
                expected_model_fingerprint: "sha256:reranker".to_owned(),
                query: "query".to_owned(),
                documents: vec!["low".to_owned(), "mid".to_owned(), "high".to_owned()],
            }))
            .await;
        let ResponseEnvelope::Ok {
            response: ServiceResponse::Models(ModelResponse::Rerank { results, .. }),
        } = response
        else {
            panic!("unexpected response");
        };
        assert_eq!(results[0].score, 0.0);
        assert_eq!(results[1].score, 0.5);
        assert_eq!(results[2].score, 1.0);
        assert!(results.iter().all(|item| (0.0..=1.0).contains(&item.score)));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_requests_wait_for_a_slot_instead_of_failing_randomly() {
        let router = Arc::new(router_with(
            ModelEngines {
                embedding: Some(embedding(Some(2), None, Duration::from_millis(75))),
                reranker: None,
            },
            1,
        ));
        let barrier = Arc::new(Barrier::new(3));
        let mut tasks = Vec::new();
        for text in ["first", "second"] {
            let router = router.clone();
            let barrier = barrier.clone();
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                router.route(embed_request(vec![text.to_owned()])).await
            }));
        }
        barrier.wait().await;

        for task in tasks {
            assert!(matches!(task.await.unwrap(), ResponseEnvelope::Ok { .. }));
        }
    }
}
