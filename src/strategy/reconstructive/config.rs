use crate::inference::{
    EmbeddingBackend, GenerationOptions, PromptProcessorBackend, RerankBackend,
};
use anyhow::{Result, ensure};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReconstructiveConfig {
    pub(super) embedding: EmbeddingBackend,
    pub(super) reranker: RerankBackend,
    pub(super) controller: ControllerConfig,
    #[serde(default)]
    pub(super) retrieval: RetrievalConfig,
    #[serde(default)]
    pub(super) context: ContextConfig,
    #[serde(default = "default_embedding_batch_size")]
    pub(super) embedding_batch_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum ControllerConfig {
    /// One retrieval call with the same anchor and context budget as the
    /// multi-round variants. It is not compute-matched to repeated reranking.
    OneShot { hits: usize },
    /// Repeats the same query while excluding prior anchors. This isolates the
    /// contribution of seen-item masking from query refinement.
    Masked {
        hits_per_round: usize,
        max_rounds: usize,
    },
    /// Training-free evidence-gap controller inspired by MemR3 and
    /// MemMachine's chain-of-query agent.
    EvidenceGap {
        hits_per_round: usize,
        max_rounds: usize,
        #[serde(default = "default_confidence_threshold")]
        confidence_threshold: f64,
        backend: PromptProcessorBackend,
        #[serde(default)]
        generation: ControllerGenerationConfig,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct RetrievalConfig {
    pub(super) dense_limit: usize,
    pub(super) sparse_limit: usize,
    pub(super) fused_limit: usize,
    pub(super) require_dense_index: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            dense_limit: 100,
            sparse_limit: 100,
            fused_limit: 100,
            require_dense_index: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct ContextConfig {
    pub(super) neighbor_before: usize,
    pub(super) neighbor_after: usize,
    pub(super) max_events: usize,
    pub(super) max_context_chars: usize,
    pub(super) max_controller_chars: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            neighbor_before: 1,
            neighbor_after: 2,
            max_events: 100,
            max_context_chars: 32_768,
            max_controller_chars: 24_576,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct ControllerGenerationConfig {
    temperature: f64,
    top_p: f64,
    max_tokens: usize,
    enable_thinking: bool,
    pub(super) validation_retries: usize,
}

impl Default for ControllerGenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 768,
            enable_thinking: false,
            validation_retries: 2,
        }
    }
}

impl ReconstructiveConfig {
    pub(super) fn validate(&self) -> Result<()> {
        self.embedding.validate("embedding")?;
        self.reranker.validate("reranker")?;
        ensure!(
            (1..=1_024).contains(&self.embedding_batch_size),
            "embedding_batch_size must be between 1 and 1024"
        );
        for (name, value) in [
            ("retrieval.dense_limit", self.retrieval.dense_limit),
            ("retrieval.sparse_limit", self.retrieval.sparse_limit),
            ("retrieval.fused_limit", self.retrieval.fused_limit),
        ] {
            ensure!((1..=2_000).contains(&value), "{name} must be in 1..=2000");
        }
        ensure!(
            self.retrieval.fused_limit
                <= self
                    .retrieval
                    .dense_limit
                    .saturating_add(self.retrieval.sparse_limit),
            "retrieval.fused_limit cannot exceed the combined dense and sparse limits"
        );
        ensure!(
            self.context.max_events > 0 && self.context.max_events <= 1_000,
            "context.max_events must be in 1..=1000"
        );
        ensure!(
            (1_024..=1_048_576).contains(&self.context.max_context_chars),
            "context.max_context_chars must be in 1024..=1048576"
        );
        ensure!(
            (1_024..=1_048_576).contains(&self.context.max_controller_chars),
            "context.max_controller_chars must be in 1024..=1048576"
        );
        match &self.controller {
            ControllerConfig::OneShot { hits } => self.validate_rounds(*hits, 1),
            ControllerConfig::Masked {
                hits_per_round,
                max_rounds,
            } => self.validate_rounds(*hits_per_round, *max_rounds),
            ControllerConfig::EvidenceGap {
                hits_per_round,
                max_rounds,
                confidence_threshold,
                backend,
                generation,
            } => {
                self.validate_rounds(*hits_per_round, *max_rounds)?;
                ensure!(
                    confidence_threshold.is_finite() && (0.0..=1.0).contains(confidence_threshold),
                    "controller.confidence_threshold must be in [0, 1]"
                );
                ensure!(
                    generation.temperature.is_finite() && generation.temperature >= 0.0,
                    "controller.generation.temperature must be finite and non-negative"
                );
                ensure!(
                    generation.top_p.is_finite()
                        && generation.top_p > 0.0
                        && generation.top_p <= 1.0,
                    "controller.generation.top_p must be in (0, 1]"
                );
                ensure!(
                    (1..=8_192).contains(&generation.max_tokens),
                    "controller.generation.max_tokens must be in 1..=8192"
                );
                ensure!(
                    generation.validation_retries <= 5,
                    "controller.generation.validation_retries must be in 0..=5"
                );
                backend.validate("controller.backend")?;
                backend.validate_generation("controller.backend", &generation.options())
            }
        }
    }

    fn validate_rounds(&self, hits_per_round: usize, max_rounds: usize) -> Result<()> {
        ensure!(
            (1..=50).contains(&hits_per_round),
            "controller hits per round must be in 1..=50"
        );
        ensure!(
            (1..=10).contains(&max_rounds),
            "controller max_rounds must be in 1..=10"
        );
        ensure!(
            hits_per_round.saturating_mul(max_rounds) <= self.context.max_events,
            "controller anchor budget exceeds context.max_events"
        );
        Ok(())
    }

    pub(super) fn max_rounds(&self) -> usize {
        match &self.controller {
            ControllerConfig::OneShot { .. } => 1,
            ControllerConfig::Masked { max_rounds, .. }
            | ControllerConfig::EvidenceGap { max_rounds, .. } => *max_rounds,
        }
    }

    pub(super) fn hits_per_round(&self) -> usize {
        match &self.controller {
            ControllerConfig::OneShot { hits } => *hits,
            ControllerConfig::Masked { hits_per_round, .. }
            | ControllerConfig::EvidenceGap { hits_per_round, .. } => *hits_per_round,
        }
    }
}

impl ControllerGenerationConfig {
    pub(super) fn options(&self) -> GenerationOptions {
        GenerationOptions {
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            enable_thinking: self.enable_thinking,
        }
    }

    pub(super) fn enable_thinking(&self) -> bool {
        self.enable_thinking
    }
}

fn default_embedding_batch_size() -> usize {
    32
}

fn default_confidence_threshold() -> f64 {
    0.8
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn controller_variants_are_strict_and_budgeted() {
        let config: ReconstructiveConfig = serde_json::from_value(json!({
            "embedding": {
                "backend": "openai_compatible",
                "endpoint": {
                    "base_url": "https://example.test/v1",
                    "model": "embed",
                    "api_key_env": "TEST_KEY"
                }
            },
            "reranker": {
                "backend": "openai_compatible",
                "endpoint": {
                    "base_url": "https://example.test/v1",
                    "model": "rerank",
                    "api_key_env": "TEST_KEY"
                }
            },
            "controller": {
                "kind": "masked",
                "hits_per_round": 5,
                "max_rounds": 5
            }
        }))
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.hits_per_round(), 5);
        assert_eq!(config.max_rounds(), 5);
    }
}
