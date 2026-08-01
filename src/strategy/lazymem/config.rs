use super::{
    types::{
        PAPER_CONTEXT_RADIUS, PAPER_RETRIEVAL_TOP_N, PAPER_WINDOW_LENGTH, PAPER_WINDOW_STRIDE,
    },
    window::{HistoryWindowConfig, SpanMergeRule},
};
use crate::inference::{GenerationOptions, OpenAiEndpoint, PromptProcessorBackend, RerankEndpoint};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum ProtocolProfile {
    PaperV2,
    PublicRepro,
    Custom,
}

fn default_profile() -> ProtocolProfile {
    ProtocolProfile::PaperV2
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LazyMemConfig {
    #[serde(default = "default_profile")]
    pub(super) protocol: ProtocolProfile,
    pub(super) embedding: OpenAiEndpoint,
    pub(super) reranker: RerankEndpoint,
    pub(super) processor: PromptProcessorBackend,
    #[serde(default)]
    pub(super) retrieval: RetrievalConfig,
    #[serde(default)]
    pub(super) window: WindowConfig,
    #[serde(default)]
    generation: ProcessorGenerationConfig,
    #[serde(default = "default_embedding_batch_size")]
    pub(super) embedding_batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct RetrievalConfig {
    pub(super) dense_limit: usize,
    pub(super) sparse_limit: usize,
    pub(super) fused_limit: usize,
    pub(super) rerank_top_n: usize,
    pub(super) require_dense_index: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            dense_limit: 100,
            sparse_limit: 100,
            fused_limit: 100,
            rerank_top_n: PAPER_RETRIEVAL_TOP_N,
            require_dense_index: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct WindowConfig {
    radius: usize,
    merge_rule: SpanMergeRule,
    max_messages: usize,
    stride: usize,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            radius: PAPER_CONTEXT_RADIUS,
            merge_rule: SpanMergeRule::PaperOverlap,
            max_messages: PAPER_WINDOW_LENGTH,
            stride: PAPER_WINDOW_STRIDE,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ProcessorGenerationConfig {
    temperature: f64,
    top_p: f64,
    max_tokens: usize,
    max_concurrency: Option<usize>,
    enable_thinking: bool,
}

impl Default for ProcessorGenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            top_p: 0.95,
            max_tokens: 3_072,
            max_concurrency: None,
            enable_thinking: true,
        }
    }
}

fn default_embedding_batch_size() -> usize {
    32
}

impl LazyMemConfig {
    pub(super) fn validate(&self) -> Result<()> {
        self.embedding.validate("embedding")?;
        self.reranker.validate("reranker")?;
        self.processor.validate("processor")?;
        ensure!(
            (1..=1_000).contains(&self.retrieval.dense_limit),
            "retrieval.dense_limit must be between 1 and 1000"
        );
        ensure!(
            (1..=1_000).contains(&self.retrieval.sparse_limit),
            "retrieval.sparse_limit must be between 1 and 1000"
        );
        ensure!(
            (1..=2_000).contains(&self.retrieval.fused_limit),
            "retrieval.fused_limit must be between 1 and 2000"
        );
        ensure!(
            (1..=self.retrieval.fused_limit).contains(&self.retrieval.rerank_top_n),
            "retrieval.rerank_top_n must be positive and no greater than fused_limit"
        );
        ensure!(
            self.window.max_messages > 0 && self.window.stride > 0,
            "window message limit and stride must be positive"
        );
        ensure!(
            self.window.stride <= self.window.max_messages,
            "window.stride must not exceed max_messages"
        );
        ensure!(
            self.generation.temperature.is_finite() && self.generation.temperature >= 0.0,
            "generation.temperature must be finite and non-negative"
        );
        ensure!(
            self.generation.top_p.is_finite()
                && self.generation.top_p > 0.0
                && self.generation.top_p <= 1.0,
            "generation.top_p must be in (0, 1]"
        );
        ensure!(
            (1..=65_536).contains(&self.generation.max_tokens),
            "generation.max_tokens must be between 1 and 65536"
        );
        ensure!(
            (1..=self.processor.max_concurrency_limit())
                .contains(&self.effective_max_concurrency()),
            "generation.max_concurrency must be between 1 and {} for the {} processor backend",
            self.processor.max_concurrency_limit(),
            self.processor.identity().backend
        );
        ensure!(
            (1..=1_024).contains(&self.embedding_batch_size),
            "embedding_batch_size must be between 1 and 1024"
        );

        if matches!(
            self.protocol,
            ProtocolProfile::PaperV2 | ProtocolProfile::PublicRepro
        ) {
            ensure!(
                matches!(
                    &self.processor,
                    PromptProcessorBackend::OpenaiCompatible { .. }
                ),
                "paper-v2/public-repro require the openai_compatible processor backend; codex_cli is a prompt-only custom-profile ablation"
            );
            ensure!(
                self.retrieval.dense_limit == 100
                    && self.retrieval.sparse_limit == 100
                    && self.retrieval.fused_limit == 100
                    && self.retrieval.rerank_top_n == PAPER_RETRIEVAL_TOP_N
                    && self.retrieval.require_dense_index,
                "paper-v2 requires dense=100, sparse=100, fused=100, rerank_top_n=50, and require_dense_index=true"
            );
            let expected_merge_rule = match self.protocol {
                ProtocolProfile::PaperV2 => SpanMergeRule::PaperOverlap,
                ProtocolProfile::PublicRepro => SpanMergeRule::RepositoryContiguous,
                ProtocolProfile::Custom => unreachable!(),
            };
            ensure!(
                self.window.radius == PAPER_CONTEXT_RADIUS
                    && self.window.merge_rule == expected_merge_rule
                    && self.window.max_messages == PAPER_WINDOW_LENGTH
                    && self.window.stride == PAPER_WINDOW_STRIDE,
                "paper-v2/public-repro require radius=2, their profile-specific merge rule, max_messages=8, and stride=7"
            );
            ensure!(
                self.generation.temperature == 0.1
                    && self.generation.top_p == 0.95
                    && self.generation.max_tokens == 3_072
                    && self.generation.enable_thinking,
                "paper-v2 requires the published inference reconstruction: temperature=0.1, top_p=0.95, max_tokens=3072, enable_thinking=true"
            );
        }
        Ok(())
    }

    pub(super) fn window_config(&self) -> HistoryWindowConfig {
        HistoryWindowConfig {
            radius: self.window.radius,
            merge_rule: self.window.merge_rule,
            max_window_messages: self.window.max_messages,
            stride: self.window.stride,
        }
    }

    pub(super) fn generation_options(&self) -> GenerationOptions {
        GenerationOptions {
            temperature: self.generation.temperature,
            top_p: self.generation.top_p,
            max_tokens: self.generation.max_tokens,
            enable_thinking: self.generation.enable_thinking,
        }
    }

    pub(super) fn effective_max_concurrency(&self) -> usize {
        self.generation
            .max_concurrency
            .unwrap_or_else(|| self.processor.default_max_concurrency())
    }

    pub(super) fn generation_metadata(&self) -> Value {
        json!({
            "requested_sampling": {
                "temperature": self.generation.temperature,
                "top_p": self.generation.top_p,
                "max_tokens": self.generation.max_tokens,
                "enable_thinking": self.generation.enable_thinking,
            },
            "effective_max_concurrency": self.effective_max_concurrency(),
            "sampling_controls_applied": matches!(&self.processor, PromptProcessorBackend::OpenaiCompatible { .. }),
        })
    }
}
