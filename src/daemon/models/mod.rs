//! Injectable, synchronous model engines.
//!
//! Engines are synchronous because local inference is CPU/GPU blocking work.
//! The service layer owns admission control and runs calls on Tokio's blocking
//! pool, keeping the socket reactor independent from a particular ML runtime.

use std::sync::Arc;

use crate::daemon_ipc::protocol::ModelIdentity;
use thiserror::Error;

use crate::daemon::config::ModelsConfig;

#[derive(Debug, Error)]
pub enum ModelEngineError {
    #[error("model input is invalid: {0}")]
    InvalidInput(String),
    #[error("model inference failed: {0}")]
    Inference(String),
}

pub trait EmbeddingEngine: Send + Sync + 'static {
    fn identity(&self) -> &ModelIdentity;

    /// Return one embedding for each input, in input order.
    fn embed(&self, input: &[String]) -> Result<Vec<Vec<f32>>, ModelEngineError>;
}

pub trait RerankerEngine: Send + Sync + 'static {
    fn identity(&self) -> &ModelIdentity;

    /// Return one raw classifier logit for each document, in document order.
    /// The service boundary applies the stable sigmoid used by Semafuse.
    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>, ModelEngineError>;
}

#[derive(Clone, Default)]
pub struct ModelEngines {
    pub embedding: Option<Arc<dyn EmbeddingEngine>>,
    pub reranker: Option<Arc<dyn RerankerEngine>>,
}

impl ModelEngines {
    pub fn is_empty(&self) -> bool {
        self.embedding.is_none() && self.reranker.is_none()
    }
}

/// Load configured engines without allowing the long-running process to fetch
/// artifacts. Model directories must have been provisioned by a one-shot
/// command before `semafused` starts.
#[cfg(feature = "models-candle")]
pub fn load_engines(config: &ModelsConfig) -> Result<ModelEngines, String> {
    candle::load_engines(config)
}

#[cfg(not(feature = "models-candle"))]
pub fn load_engines(_config: &ModelsConfig) -> Result<ModelEngines, String> {
    Err(
        "models are configured, but this semafused binary was built without the `models-candle` feature"
            .to_owned(),
    )
}

#[cfg(feature = "models-candle")]
mod candle;
