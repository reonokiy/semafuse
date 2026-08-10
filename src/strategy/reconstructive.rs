//! Ground-truth-preserving, closed-loop memory retrieval.
//!
//! This strategy keeps source messages verbatim and layers a bounded,
//! training-free evidence-gap controller over Semafuse's dense + BM25 + RRF +
//! cross-encoder retrieval stack. Derived controller state guides retrieval but
//! never replaces the source evidence shown to the downstream reader.

mod config;
mod controller;
mod render;

use self::{
    config::{ControllerConfig, ReconstructiveConfig},
    controller::{
        CONTROLLER_PROMPT_VERSION, CONTROLLER_SYSTEM_PROMPT, ControllerCallTrace,
        ControllerFallback, ControllerFallbackKind, ControllerModelDecisionTrace,
        ControllerResolution, ControllerResolutionTrace, ControllerRoundTrace,
        RETRY_CORRECTION_RESERVE_CHARS, controller_prompt, is_repeated_query,
        parse_and_validate_decision,
    },
    render::{AnchorHit, render_verbatim_context},
};
use super::{
    MemoryStrategy, RecallItem, RecallRequest, RecallResult, StrategyContext, StrategyEvent,
    lazymem::retrieval::{
        RerankInput, RerankScore, RerankedCandidate, RrfInput, apply_reranker,
        okapi_bm25_ranked_ids, paper_retrieval_text, reciprocal_rank_fusion,
    },
    registry::StrategyRegistration,
};
use crate::{
    inference::JsonContract,
    model::{
        CanonicalEvent, EventData, MemoryMutation, SearchQuery, StrategyEventVector, VectorQuery,
    },
};
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    time::{Duration, Instant},
};

pub(in crate::strategy) const REGISTRATION: StrategyRegistration =
    StrategyRegistration::new("reconstructive", build);

const NAMESPACE: &str = "reconstructive";
const BENCHMARK_TRACE_SCHEMA_VERSION: &str = "semafuse.reconstructive-recall/v1";

fn build(settings: Option<&Value>) -> Result<Box<dyn MemoryStrategy>> {
    let settings = settings
        .context("reconstructive requires explicit embedding, reranker, and controller settings")?;
    let config: ReconstructiveConfig =
        serde_json::from_value(settings.clone()).context("deserialize reconstructive settings")?;
    config.validate()?;
    Ok(Box::new(ReconstructiveStrategy { config }))
}

struct ReconstructiveStrategy {
    config: ReconstructiveConfig,
}

#[async_trait]
impl MemoryStrategy for ReconstructiveStrategy {
    fn name(&self) -> &'static str {
        NAMESPACE
    }

    async fn on_event(
        &self,
        _context: &dyn StrategyContext,
        event: &StrategyEvent,
    ) -> Result<Vec<MemoryMutation>> {
        let source = event
            .events
            .iter()
            .filter_map(|event| {
                let content = event.search_text();
                (is_message_event(event) && !content.trim().is_empty())
                    .then_some((event, paper_retrieval_text(event)))
            })
            .collect::<Vec<_>>();
        let mut mutations = Vec::with_capacity(source.len());
        for chunk in source.chunks(self.config.embedding_batch_size) {
            let texts = chunk
                .iter()
                .map(|(_, text)| text.clone())
                .collect::<Vec<_>>();
            let vectors = self.config.embedding.embed(&texts).await?;
            ensure!(
                vectors.len() == chunk.len(),
                "embedding backend returned {} vectors for {} events",
                vectors.len(),
                chunk.len()
            );
            for ((event, _), values) in chunk.iter().zip(vectors) {
                mutations.push(MemoryMutation::UpsertVector(StrategyEventVector {
                    namespace: NAMESPACE.into(),
                    event_id: event.id.clone(),
                    model: self.config.embedding.model_identity().to_owned(),
                    values,
                }));
            }
        }
        Ok(mutations)
    }

    async fn recall(
        &self,
        context: &dyn StrategyContext,
        request: &RecallRequest,
    ) -> Result<RecallResult> {
        self.recall_internal(context, request).await
    }
}

impl ReconstructiveStrategy {
    async fn recall_internal(
        &self,
        context: &dyn StrategyContext,
        request: &RecallRequest,
    ) -> Result<RecallResult> {
        if request.query.limit == 0 {
            return Ok(RecallResult {
                items: Vec::new(),
                metadata: Map::new(),
            });
        }
        ensure!(
            !request.query.text.trim().is_empty(),
            "reconstructive recall requires a non-empty query"
        );
        let started = Instant::now();
        let mut effective_query = request.query.clone();
        effective_query.until_ms = Some(
            effective_query
                .until_ms
                .map_or(request.query_time_ms, |until| {
                    until.min(request.query_time_ms)
                }),
        );
        let (checkpoint, high_water) = context.strategy_progress(NAMESPACE).await?;
        ensure!(
            checkpoint == high_water,
            "reconstructive index is stale: checkpoint {checkpoint}, event high-water {high_water}; dispatch or rebuild before recall"
        );
        let corpus = context.scoped_message_events(&effective_query).await?;
        if corpus.is_empty() {
            return Ok(empty_result(started.elapsed()));
        }
        self.validate_index(context, &effective_query, &corpus)
            .await?;

        let mut query = request.query.text.clone();
        let mut attempted_queries = Vec::new();
        let mut anchors = Vec::<AnchorHit>::new();
        let mut excluded = HashSet::<String>::new();
        let mut round_traces = Vec::<ControllerRoundTrace>::new();
        let mut retrieval_elapsed = Duration::ZERO;
        let mut processing_elapsed = Duration::ZERO;

        for round in 1..=self.config.max_rounds() {
            attempted_queries.push(query.clone());
            let retrieval_started = Instant::now();
            let output = self
                .retrieve_round(
                    context,
                    &effective_query,
                    &corpus,
                    &query,
                    &excluded,
                    self.config.hits_per_round(),
                )
                .await?;
            retrieval_elapsed += retrieval_started.elapsed();
            if output.candidates.is_empty() {
                round_traces.push(ControllerRoundTrace {
                    round,
                    query: query.clone(),
                    anchor_ids: Vec::new(),
                    selected_context_ids: round_traces
                        .last()
                        .map(|trace| trace.selected_context_ids.clone())
                        .unwrap_or_default(),
                    new_context_ids: Vec::new(),
                    controller_evidence_ids: Vec::new(),
                    dense_candidate_count: output.dense_candidate_count,
                    sparse_candidate_count: output.sparse_candidate_count,
                    fused_candidate_count: output.fused_candidate_count,
                    reranked_candidate_count: output.reranked_candidate_count,
                    query_embedding_calls: output.query_embedding_calls,
                    reranker_calls: output.reranker_calls,
                    resolution: None,
                    calls: Vec::new(),
                });
                break;
            }

            let anchor_ids = output
                .candidates
                .iter()
                .map(|candidate| candidate.message_id.clone())
                .collect::<Vec<_>>();
            for (rank, candidate) in output.candidates.into_iter().enumerate() {
                anchors.push(AnchorHit {
                    message_id: candidate.message_id,
                    round,
                    rank: rank + 1,
                    rerank_score: candidate.rerank_score,
                });
            }
            let previously_selected = round_traces
                .last()
                .map(|trace| {
                    trace
                        .selected_context_ids
                        .iter()
                        .cloned()
                        .collect::<HashSet<_>>()
                })
                .unwrap_or_default();
            let rendered = render_verbatim_context(&corpus, &anchors, &self.config.context)?;
            excluded.extend(rendered.selected_event_ids.iter().cloned());
            let new_context_ids = rendered
                .selected_event_ids
                .iter()
                .filter(|id| !previously_selected.contains(id.as_str()))
                .cloned()
                .collect();
            let mut trace = ControllerRoundTrace {
                round,
                query: query.clone(),
                anchor_ids: anchor_ids.clone(),
                selected_context_ids: rendered.selected_event_ids.clone(),
                new_context_ids,
                controller_evidence_ids: Vec::new(),
                dense_candidate_count: output.dense_candidate_count,
                sparse_candidate_count: output.sparse_candidate_count,
                fused_candidate_count: output.fused_candidate_count,
                reranked_candidate_count: output.reranked_candidate_count,
                query_embedding_calls: output.query_embedding_calls,
                reranker_calls: output.reranker_calls,
                resolution: None,
                calls: Vec::new(),
            };

            match &self.config.controller {
                ControllerConfig::OneShot { .. } => {
                    round_traces.push(trace);
                    break;
                }
                ControllerConfig::Masked { .. } => round_traces.push(trace),
                ControllerConfig::EvidenceGap {
                    confidence_threshold,
                    backend,
                    generation,
                    ..
                } => {
                    if round == self.config.max_rounds() {
                        // No later retrieval can consume another refinement.
                        // The controller never answers or filters final context,
                        // so a final-round call would only add cost and failure
                        // surface without changing the returned memory.
                        round_traces.push(trace);
                        break;
                    }
                    let event_map = corpus
                        .iter()
                        .map(|event| (event.id.as_str(), event))
                        .collect::<HashMap<_, _>>();
                    let mut evidence_ids = anchor_ids.clone();
                    evidence_ids
                        .extend(anchors.iter().rev().map(|anchor| anchor.message_id.clone()));
                    evidence_ids.extend(rendered.selected_event_ids.iter().cloned());
                    let mut unique_evidence_ids = HashSet::new();
                    evidence_ids.retain(|id| unique_evidence_ids.insert(id.clone()));
                    let evidence = evidence_ids
                        .iter()
                        .map(|id| {
                            event_map
                                .get(id.as_str())
                                .copied()
                                .with_context(|| format!("selected evidence {id} is missing"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let prompt = controller_prompt(
                        &request.query.text,
                        request.query_time_ms,
                        &attempted_queries,
                        &evidence,
                        self.config.context.max_controller_chars,
                    )?;
                    trace.controller_evidence_ids = evidence_ids
                        .iter()
                        .take(prompt.evidence_count)
                        .cloned()
                        .collect();
                    let processing_started = Instant::now();
                    let (resolution, calls) = Self::decide(
                        backend,
                        generation,
                        &prompt.text,
                        prompt.evidence_count,
                        &attempted_queries,
                        self.config.context.max_controller_chars,
                    )
                    .await?;
                    processing_elapsed += processing_started.elapsed();
                    let stop = resolution.should_stop(*confidence_threshold);
                    let applied_query =
                        resolution.applied_query(&request.query.text, *confidence_threshold);
                    let model_decision =
                        resolution
                            .decision
                            .as_ref()
                            .map(|decision| ControllerModelDecisionTrace {
                                is_sufficient: decision.is_sufficient,
                                evidence_indices: decision.evidence_indices.clone(),
                                gaps: decision.gaps.clone(),
                                proposed_query: (!decision.new_query.trim().is_empty())
                                    .then(|| decision.new_query.trim().to_owned()),
                                confidence_score: decision.confidence_score,
                            });
                    trace.resolution = Some(ControllerResolutionTrace {
                        model_decision,
                        applied_query: applied_query.clone(),
                        fallback: resolution.fallback,
                    });
                    trace.calls = calls;
                    round_traces.push(trace);
                    if stop {
                        break;
                    }
                    query = applied_query.context(
                        "non-stopping controller resolution omitted its applied retrieval query",
                    )?;
                }
            }
        }

        let render_started = Instant::now();
        let rendered = render_verbatim_context(&corpus, &anchors, &self.config.context)?;
        processing_elapsed += render_started.elapsed();
        ensure!(
            !rendered.selected_event_ids.is_empty(),
            "reconstructive retrieval produced anchors but no renderable source evidence"
        );
        let timing = timing_metadata(retrieval_elapsed, processing_elapsed, started.elapsed());
        let processing = processing_metadata(&round_traces);
        let retrieval_usage = retrieval_usage_metadata(&round_traces);
        let benchmark_trace = json!({
            "schema_version": BENCHMARK_TRACE_SCHEMA_VERSION,
            "retrieved_event_ids": anchors.iter().map(|anchor| &anchor.message_id).collect::<Vec<_>>(),
            "covered_event_ids": &rendered.covered_event_ids,
            "source_event_ids": &rendered.selected_event_ids,
            "context_chars": rendered.content.chars().count(),
            "timing_ms": &timing,
        });
        let metadata = Map::from_iter([
            ("query_time_ms".into(), json!(request.query_time_ms)),
            ("effective_until_ms".into(), json!(effective_query.until_ms)),
            (
                "controller_kind".into(),
                json!(controller_kind(&self.config.controller)),
            ),
            (
                "controller_prompt_version".into(),
                json!(CONTROLLER_PROMPT_VERSION),
            ),
            (
                "controller_rounds".into(),
                serde_json::to_value(&round_traces)?,
            ),
            (
                "embedding_model".into(),
                json!(self.config.embedding.model_identity()),
            ),
            (
                "reranker_model".into(),
                json!(self.config.reranker.model_identity()),
            ),
            ("processing".into(), processing),
            ("retrieval_usage".into(), retrieval_usage),
            ("timing".into(), timing.clone()),
            ("anchor_count".into(), json!(anchors.len())),
            (
                "covered_event_count".into(),
                json!(rendered.covered_event_ids.len()),
            ),
            (
                "selected_event_count".into(),
                json!(rendered.selected_event_ids.len()),
            ),
            (
                "context_chars".into(),
                json!(rendered.content.chars().count()),
            ),
            (
                "context_omitted_event_ids".into(),
                json!(&rendered.omitted_event_ids),
            ),
            ("strategy_benchmark_trace".into(), benchmark_trace),
        ]);

        let mut digest = Sha256::new();
        digest.update(request.query.text.as_bytes());
        digest.update(request.query_time_ms.to_le_bytes());
        for id in &rendered.selected_event_ids {
            digest.update(id.as_bytes());
        }
        let id = format!("recall_{}", &hex::encode(digest.finalize())[..24]);
        let score = anchors
            .iter()
            .map(|anchor| anchor.rerank_score)
            .max_by(f64::total_cmp)
            .unwrap_or(0.0);
        Ok(RecallResult {
            items: vec![RecallItem::Memory {
                namespace: NAMESPACE.into(),
                id,
                content: rendered.content,
                metadata: metadata.clone(),
                source_event_ids: rendered.selected_event_ids,
                score,
            }],
            metadata,
        })
    }

    async fn validate_index(
        &self,
        context: &dyn StrategyContext,
        query: &SearchQuery,
        corpus: &[CanonicalEvent],
    ) -> Result<()> {
        if !self.config.retrieval.require_dense_index {
            return Ok(());
        }
        let indexed = context
            .indexed_event_ids(NAMESPACE, self.config.embedding.model_identity(), query)
            .await?;
        let corpus_ids = corpus
            .iter()
            .map(|event| event.id.as_str())
            .collect::<BTreeSet<_>>();
        let indexed_ids = indexed.iter().map(String::as_str).collect::<BTreeSet<_>>();
        ensure!(
            corpus_ids == indexed_ids,
            "reconstructive dense index does not exactly cover the scoped corpus (missing {}, unexpected {})",
            corpus_ids.difference(&indexed_ids).count(),
            indexed_ids.difference(&corpus_ids).count(),
        );
        Ok(())
    }

    async fn retrieve_round(
        &self,
        context: &dyn StrategyContext,
        scope: &SearchQuery,
        corpus: &[CanonicalEvent],
        query_text: &str,
        excluded: &HashSet<String>,
        hits: usize,
    ) -> Result<RetrievalRoundOutput> {
        let sparse_pool_limit = self
            .config
            .retrieval
            .sparse_limit
            .saturating_add(excluded.len())
            .min(corpus.len());
        let sparse_ids: Vec<String> = okapi_bm25_ranked_ids(query_text, corpus, sparse_pool_limit)
            .into_iter()
            .filter(|id| !excluded.contains(id))
            .take(self.config.retrieval.sparse_limit)
            .collect();
        let query_vectors = self
            .config
            .embedding
            .embed(&[query_text.to_owned()])
            .await?;
        let query_vector = query_vectors
            .into_iter()
            .next()
            .context("embedding endpoint omitted the query vector")?;
        let mut dense_scope = scope.clone();
        dense_scope.limit = self
            .config
            .retrieval
            .dense_limit
            .saturating_add(excluded.len())
            .min(corpus.len());
        let dense_ids = context
            .query_vectors(&VectorQuery {
                namespace: NAMESPACE.into(),
                model: self.config.embedding.model_identity().to_owned(),
                values: query_vector,
                scope: dense_scope,
            })
            .await?
            .into_iter()
            .map(|hit| hit.event.id)
            .filter(|id| !excluded.contains(id))
            .take(self.config.retrieval.dense_limit)
            .collect::<Vec<_>>();
        let dense_candidate_count = dense_ids.len();
        let sparse_candidate_count = sparse_ids.len();
        let rrf = reciprocal_rank_fusion(&RrfInput {
            dense_ranked_ids: dense_ids,
            sparse_ranked_ids: sparse_ids,
            tie_break_ranked_ids: corpus
                .iter()
                .filter(|event| !excluded.contains(&event.id))
                .map(|event| event.id.clone())
                .collect(),
        })?;
        let fused = rrf
            .candidates
            .iter()
            .filter(|candidate| !excluded.contains(&candidate.message_id))
            .take(self.config.retrieval.fused_limit)
            .cloned()
            .collect::<Vec<_>>();
        let fused_candidate_count = fused.len();
        if fused.is_empty() {
            return Ok(RetrievalRoundOutput {
                candidates: Vec::new(),
                dense_candidate_count,
                sparse_candidate_count,
                fused_candidate_count,
                reranked_candidate_count: 0,
                query_embedding_calls: 1,
                reranker_calls: 0,
            });
        }
        let event_map = corpus
            .iter()
            .map(|event| (event.id.as_str(), event))
            .collect::<HashMap<_, _>>();
        let documents = fused
            .iter()
            .map(|candidate| {
                event_map
                    .get(candidate.message_id.as_str())
                    .with_context(|| format!("fused event {} is missing", candidate.message_id))
                    .map(|event| paper_retrieval_text(event))
            })
            .collect::<Result<Vec<_>>>()?;
        let scores = self.config.reranker.rerank(query_text, &documents).await?;
        let fully_reranked = apply_reranker(&RerankInput {
            scores: fused
                .iter()
                .zip(scores)
                .map(|(candidate, score)| RerankScore {
                    message_id: candidate.message_id.clone(),
                    score,
                })
                .collect(),
            fused_candidates: fused,
            top_n: self.config.retrieval.fused_limit.min(documents.len()),
        })?;
        let reranked_candidate_count = fully_reranked.candidates.len();
        let candidates = fully_reranked.candidates.into_iter().take(hits).collect();
        Ok(RetrievalRoundOutput {
            candidates,
            dense_candidate_count,
            sparse_candidate_count,
            fused_candidate_count,
            reranked_candidate_count,
            query_embedding_calls: 1,
            reranker_calls: 1,
        })
    }

    async fn decide(
        backend: &crate::inference::PromptProcessorBackend,
        generation: &config::ControllerGenerationConfig,
        prompt: &str,
        event_count: usize,
        attempted_queries: &[String],
        max_prompt_chars: usize,
    ) -> Result<(ControllerResolution, Vec<ControllerCallTrace>)> {
        let mut calls = Vec::new();
        let mut last_error: Option<anyhow::Error> = None;
        for attempt in 1..=(generation.validation_retries + 1) {
            let retry_prompt = last_error
                .as_ref()
                .map(|error| {
                    let reason = error.to_string().chars().take(256).collect::<String>();
                    let correction = format!(
                        "\n\n# Retry correction\nThe previous response was rejected: {reason}. Return a corrected object satisfying every schema and decision invariant."
                    );
                    ensure!(
                        correction.chars().count() <= RETRY_CORRECTION_RESERVE_CHARS,
                        "controller retry correction exceeded its reserved budget"
                    );
                    ensure!(
                        prompt
                            .chars()
                            .count()
                            .saturating_add(correction.chars().count())
                            <= max_prompt_chars,
                        "controller retry prompt exceeded the configured character budget"
                    );
                    Ok(format!("{prompt}{correction}"))
                })
                .transpose()?;
            let completion = backend
                .complete_structured(
                    CONTROLLER_SYSTEM_PROMPT,
                    retry_prompt.as_deref().unwrap_or(prompt),
                    generation.options(),
                    JsonContract::EvidenceGap,
                )
                .await
                .with_context(|| format!("evidence-gap controller attempt {attempt} failed"))?;
            let invalid_finish = completion.requires_stop_finish_reason
                && completion.finish_reason.as_deref() != Some("stop");
            let invalid_reasoning = !generation.enable_thinking()
                && (completion.reasoning_present
                    || completion.reasoning_tokens.unwrap_or_default() > 0);
            calls.push(ControllerCallTrace {
                attempt,
                model: completion.model.clone(),
                provider: completion.provider.clone(),
                finish_reason: completion.finish_reason.clone(),
                prompt_tokens: completion.prompt_tokens,
                completion_tokens: completion.completion_tokens,
                reasoning_tokens: completion.reasoning_tokens,
                reasoning_present: completion.reasoning_present,
                cost: completion.cost,
            });
            if invalid_finish {
                last_error = Some(anyhow::anyhow!(
                    "controller finish_reason was {:?}",
                    completion.finish_reason
                ));
                continue;
            }
            if invalid_reasoning {
                last_error = Some(anyhow::anyhow!(
                    "controller returned reasoning despite enable_thinking=false"
                ));
                continue;
            }
            match parse_and_validate_decision(&completion.content, event_count) {
                Ok(decision) => {
                    let fallback = is_repeated_query(&decision, attempted_queries).then(|| {
                        ControllerFallback {
                            kind: ControllerFallbackKind::RepeatedQuery,
                            reason: "controller repeated a prior query; continued with the original query under the seen mask"
                                .into(),
                        }
                    });
                    return Ok((
                        ControllerResolution {
                            decision: Some(decision),
                            fallback,
                        },
                        calls,
                    ));
                }
                Err(error) => last_error = Some(error),
            }
        }
        let error =
            last_error.unwrap_or_else(|| anyhow::anyhow!("controller returned no decision"));
        Ok((
            ControllerResolution {
                decision: None,
                fallback: Some(ControllerFallback {
                    kind: ControllerFallbackKind::ValidationExhausted,
                    reason: format!(
                        "controller validation exhausted; continued with the original query under the seen mask: {error}"
                    ),
                }),
            },
            calls,
        ))
    }
}

struct RetrievalRoundOutput {
    candidates: Vec<RerankedCandidate>,
    dense_candidate_count: usize,
    sparse_candidate_count: usize,
    fused_candidate_count: usize,
    reranked_candidate_count: usize,
    query_embedding_calls: usize,
    reranker_calls: usize,
}

fn controller_kind(config: &ControllerConfig) -> &'static str {
    match config {
        ControllerConfig::OneShot { .. } => "one_shot",
        ControllerConfig::Masked { .. } => "masked",
        ControllerConfig::EvidenceGap { .. } => "evidence_gap",
    }
}

fn processing_metadata(rounds: &[ControllerRoundTrace]) -> Value {
    let calls = rounds
        .iter()
        .flat_map(|round| round.calls.iter())
        .collect::<Vec<_>>();
    let prompt_tokens = calls
        .iter()
        .filter_map(|call| call.prompt_tokens)
        .sum::<u64>();
    let completion_tokens = calls
        .iter()
        .filter_map(|call| call.completion_tokens)
        .sum::<u64>();
    let reasoning_tokens = calls
        .iter()
        .filter_map(|call| call.reasoning_tokens)
        .sum::<u64>();
    let costs = calls
        .iter()
        .filter_map(|call| call.cost)
        .collect::<Vec<_>>();
    let cost = (!calls.is_empty() && costs.len() == calls.len()).then(|| costs.iter().sum::<f64>());
    let fallbacks = rounds
        .iter()
        .filter_map(|round| round.resolution.as_ref()?.fallback.as_ref())
        .collect::<Vec<_>>();
    json!({
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cost": cost,
            "calls": calls.len(),
            "calls_missing_prompt_tokens": calls.iter().filter(|call| call.prompt_tokens.is_none()).count(),
            "calls_missing_completion_tokens": calls.iter().filter(|call| call.completion_tokens.is_none()).count(),
            "calls_missing_reasoning_tokens": calls.iter().filter(|call| call.reasoning_tokens.is_none()).count(),
            "calls_missing_cost": calls.iter().filter(|call| call.cost.is_none()).count(),
        },
        "processor_calls": rounds.iter().filter(|round| !round.calls.is_empty()).map(|round| json!({
            "round": round.round,
            "calls": &round.calls,
        })).collect::<Vec<_>>(),
        "controller_fallbacks": {
            "total": fallbacks.len(),
            "repeated_query": fallbacks.iter().filter(|fallback| fallback.kind == ControllerFallbackKind::RepeatedQuery).count(),
            "validation_exhausted": fallbacks.iter().filter(|fallback| fallback.kind == ControllerFallbackKind::ValidationExhausted).count(),
        },
    })
}

fn retrieval_usage_metadata(rounds: &[ControllerRoundTrace]) -> Value {
    json!({
        "scope": "recall_only; history-index embedding calls and provider-side costs are not included",
        "rounds": rounds.len(),
        "query_embedding_calls": rounds.iter().map(|round| round.query_embedding_calls).sum::<usize>(),
        "reranker_calls": rounds.iter().map(|round| round.reranker_calls).sum::<usize>(),
        "dense_candidates": rounds.iter().map(|round| round.dense_candidate_count).sum::<usize>(),
        "sparse_candidates": rounds.iter().map(|round| round.sparse_candidate_count).sum::<usize>(),
        "fused_candidates": rounds.iter().map(|round| round.fused_candidate_count).sum::<usize>(),
        "reranked_candidates": rounds.iter().map(|round| round.reranked_candidate_count).sum::<usize>(),
    })
}

fn timing_metadata(retrieval: Duration, processing: Duration, total: Duration) -> Value {
    json!({
        "retrieval_ms": retrieval.as_secs_f64() * 1_000.0,
        "window_construction_ms": 0.0,
        "processing_ms": processing.as_secs_f64() * 1_000.0,
        "total_ms": total.as_secs_f64() * 1_000.0,
    })
}

fn empty_result(total: Duration) -> RecallResult {
    let timing = timing_metadata(total, Duration::ZERO, total);
    RecallResult {
        items: Vec::new(),
        metadata: Map::from_iter([
            (
                "processing".into(),
                json!({
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "reasoning_tokens": 0,
                        "cost": null,
                        "calls": 0,
                        "calls_missing_prompt_tokens": 0,
                        "calls_missing_completion_tokens": 0,
                        "calls_missing_reasoning_tokens": 0,
                        "calls_missing_cost": 0,
                    },
                    "processor_calls": [],
                    "controller_fallbacks": {
                        "total": 0,
                        "repeated_query": 0,
                        "validation_exhausted": 0,
                    },
                }),
            ),
            (
                "retrieval_usage".into(),
                json!({
                    "scope": "recall_only; history-index embedding calls and provider-side costs are not included",
                    "rounds": 0,
                    "query_embedding_calls": 0,
                    "reranker_calls": 0,
                    "dense_candidates": 0,
                    "sparse_candidates": 0,
                    "fused_candidates": 0,
                    "reranked_candidates": 0,
                }),
            ),
            ("timing".into(), timing.clone()),
            ("anchor_count".into(), json!(0)),
            ("covered_event_count".into(), json!(0)),
            ("selected_event_count".into(), json!(0)),
            ("context_chars".into(), json!(0)),
            (
                "strategy_benchmark_trace".into(),
                json!({
                    "schema_version": BENCHMARK_TRACE_SCHEMA_VERSION,
                    "retrieved_event_ids": [],
                    "covered_event_ids": [],
                    "source_event_ids": [],
                    "timing_ms": timing,
                }),
            ),
        ]),
    }
}

fn is_message_event(event: &CanonicalEvent) -> bool {
    matches!(
        &event.data,
        EventData::Message { .. } | EventData::ToolCall { .. } | EventData::ToolResult { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::{AgentKind, EventSource, MessageRole},
        storage::{EventStore, SqliteStore},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
    };

    async fn read_json_request(stream: &mut TcpStream) -> (String, Value) {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 8_192];
        let header_end = loop {
            let count = stream.read(&mut buffer).await.unwrap();
            assert!(count > 0, "mock client closed before sending HTTP headers");
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
            .unwrap_or(0);
        while request.len() < header_end + content_length {
            let count = stream.read(&mut buffer).await.unwrap();
            assert!(count > 0, "mock client closed before sending the HTTP body");
            request.extend_from_slice(&buffer[..count]);
        }
        (
            headers.lines().next().unwrap().to_owned(),
            serde_json::from_slice(&request[header_end..header_end + content_length]).unwrap(),
        )
    }

    async fn write_json_response(stream: &mut TcpStream, response: &Value) {
        let encoded = serde_json::to_vec(response).unwrap();
        let header = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            encoded.len()
        );
        stream.write_all(header.as_bytes()).await.unwrap();
        stream.write_all(&encoded).await.unwrap();
    }

    async fn controller_backend_with_responses(
        responses: Vec<Value>,
    ) -> (
        crate::inference::PromptProcessorBackend,
        Arc<AtomicUsize>,
        tokio::task::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let server_calls = Arc::clone(&calls);
        let server = tokio::spawn(async move {
            for decision in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let (request_line, body) = read_json_request(&mut stream).await;
                assert!(request_line.contains("/chat/completions"));
                assert_eq!(body["response_format"]["json_schema"]["strict"], true);
                server_calls.fetch_add(1, Ordering::SeqCst);
                let response = json!({
                    "model": "controller",
                    "provider": "mock",
                    "choices": [{
                        "finish_reason": "stop",
                        "message": {"content": serde_json::to_string(&decision).unwrap()}
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                        "cost": 0.001
                    }
                });
                write_json_response(&mut stream, &response).await;
            }
        });
        let backend = serde_json::from_value(json!({
            "backend": "openai_compatible",
            "endpoint": {
                "base_url": format!("http://{address}/v1"),
                "model": "controller",
                "thinking_control": "no_control"
            }
        }))
        .unwrap();
        (backend, calls, server)
    }

    fn controller_generation(validation_retries: usize) -> config::ControllerGenerationConfig {
        serde_json::from_value(json!({
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 256,
            "enable_thinking": false,
            "validation_retries": validation_retries
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn repeated_controller_query_preserves_proposal_and_applies_original_query() {
        let (backend, calls, server) = controller_backend_with_responses(vec![json!({
            "is_sufficient": false,
            "evidence_indices": [0],
            "gaps": ["the missing date"],
            "new_query": "Original query?",
            "confidence_score": 0.9
        })])
        .await;
        let (resolution, traces) = ReconstructiveStrategy::decide(
            &backend,
            &controller_generation(1),
            "test prompt",
            1,
            &["original   query".into()],
            4_096,
        )
        .await
        .unwrap();
        server.await.unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(traces.len(), 1);
        assert_eq!(
            resolution.decision.as_ref().unwrap().new_query,
            "Original query?"
        );
        assert_eq!(
            resolution.fallback.as_ref().unwrap().kind,
            ControllerFallbackKind::RepeatedQuery
        );
        assert_eq!(
            resolution.applied_query("original query", 0.8).as_deref(),
            Some("original query")
        );
    }

    #[tokio::test]
    async fn semantic_validation_exhaustion_continues_with_masked_original_query() {
        let invalid = json!({
            "is_sufficient": false,
            "evidence_indices": [9],
            "gaps": ["the missing date"],
            "new_query": "new query",
            "confidence_score": 0.9
        });
        let (backend, calls, server) =
            controller_backend_with_responses(vec![invalid.clone(), invalid]).await;
        let (resolution, traces) = ReconstructiveStrategy::decide(
            &backend,
            &controller_generation(1),
            "test prompt",
            1,
            &["original query".into()],
            4_096,
        )
        .await
        .unwrap();
        server.await.unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(traces.len(), 2);
        assert!(resolution.decision.is_none());
        assert_eq!(
            resolution.fallback.as_ref().unwrap().kind,
            ControllerFallbackKind::ValidationExhausted
        );
        assert_eq!(
            resolution.applied_query("original query", 0.8).as_deref(),
            Some("original query")
        );
    }

    #[test]
    fn fallback_metadata_separates_query_policy_from_validation_failures() {
        let round = |round, kind| ControllerRoundTrace {
            round,
            query: "query".into(),
            anchor_ids: Vec::new(),
            selected_context_ids: Vec::new(),
            new_context_ids: Vec::new(),
            controller_evidence_ids: Vec::new(),
            dense_candidate_count: 0,
            sparse_candidate_count: 0,
            fused_candidate_count: 0,
            reranked_candidate_count: 0,
            query_embedding_calls: 0,
            reranker_calls: 0,
            resolution: Some(ControllerResolutionTrace {
                model_decision: None,
                applied_query: Some("query".into()),
                fallback: Some(ControllerFallback {
                    kind,
                    reason: "test".into(),
                }),
            }),
            calls: Vec::new(),
        };
        let metadata = processing_metadata(&[
            round(1, ControllerFallbackKind::RepeatedQuery),
            round(2, ControllerFallbackKind::ValidationExhausted),
        ]);
        assert_eq!(metadata["controller_fallbacks"]["total"], 2);
        assert_eq!(metadata["controller_fallbacks"]["repeated_query"], 1);
        assert_eq!(metadata["controller_fallbacks"]["validation_exhausted"], 1);
    }

    #[tokio::test]
    async fn evidence_gap_pipeline_rewrites_stops_and_preserves_source_truth() {
        let listener = match TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => listener,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(error) => panic!("bind mock inference server: {error}"),
        };
        let address = listener.local_addr().unwrap();
        let controller_calls = Arc::new(AtomicUsize::new(0));
        let server_controller_calls = Arc::clone(&controller_calls);
        let server = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let controller_calls = Arc::clone(&server_controller_calls);
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut buffer = [0_u8; 8_192];
                    let header_end = loop {
                        let count = stream.read(&mut buffer).await.unwrap();
                        if count == 0 {
                            return;
                        }
                        request.extend_from_slice(&buffer[..count]);
                        if let Some(position) =
                            request.windows(4).position(|part| part == b"\r\n\r\n")
                        {
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
                        .unwrap_or(0);
                    while request.len() < header_end + content_length {
                        let count = stream.read(&mut buffer).await.unwrap();
                        if count == 0 {
                            return;
                        }
                        request.extend_from_slice(&buffer[..count]);
                    }
                    let request_line = headers.lines().next().unwrap();
                    let body: Value =
                        serde_json::from_slice(&request[header_end..header_end + content_length])
                            .unwrap();
                    let response = if request_line.contains("/embeddings") {
                        let input = body["input"].as_array().unwrap();
                        json!({
                            "data": input.iter().enumerate().map(|(index, value)| {
                                let text = value.as_str().unwrap().to_ascii_lowercase();
                                let embedding = if text.contains("migration") {
                                    vec![1.0, 0.0, 0.0]
                                } else if text.contains("database") || text.contains("sqlite") {
                                    vec![0.0, 1.0, 0.0]
                                } else {
                                    vec![0.0, 0.0, 1.0]
                                };
                                json!({"index": index, "embedding": embedding})
                            }).collect::<Vec<_>>()
                        })
                    } else if request_line.contains("/rerank") {
                        let query = body["query"].as_str().unwrap().to_ascii_lowercase();
                        let texts = body["texts"].as_array().unwrap();
                        Value::Array(
                            texts
                                .iter()
                                .enumerate()
                                .rev()
                                .map(|(index, value)| {
                                    let text = value.as_str().unwrap().to_ascii_lowercase();
                                    let score = if (query.contains("migration")
                                        && text.contains("migration"))
                                        || ((query.contains("database")
                                            || query.contains("sqlite"))
                                            && text.contains("sqlite"))
                                    {
                                        0.95
                                    } else {
                                        0.05
                                    };
                                    json!({"index": index, "score": score})
                                })
                                .collect(),
                        )
                    } else if request_line.contains("/chat/completions") {
                        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
                        let prompt = body["messages"][1]["content"].as_str().unwrap();
                        assert!(prompt.contains("# Query time"));
                        assert!(prompt.contains("session=session"));
                        let call = controller_calls.fetch_add(1, Ordering::SeqCst);
                        let decision = if call == 0 {
                            json!({
                                "is_sufficient": false,
                                "evidence_indices": [0],
                                "gaps": ["the database used after migration"],
                                "new_query": "database SQLite",
                                "confidence_score": 0.95
                            })
                        } else {
                            json!({
                                "is_sufficient": true,
                                "evidence_indices": [0, 1],
                                "gaps": [],
                                "new_query": "",
                                "confidence_score": 0.99
                            })
                        };
                        json!({
                            "model": "controller",
                            "provider": "mock",
                            "choices": [{
                                "finish_reason": "stop",
                                "message": {"content": serde_json::to_string(&decision).unwrap()}
                            }],
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 3,
                                "completion_tokens_details": {"reasoning_tokens": 0},
                                "cost": 0.001
                            }
                        })
                    } else {
                        panic!("unexpected request: {request_line}");
                    };
                    let encoded = serde_json::to_vec(&response).unwrap();
                    let header = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        encoded.len()
                    );
                    stream.write_all(header.as_bytes()).await.unwrap();
                    stream.write_all(&encoded).await.unwrap();
                });
            }
        });

        let endpoint = format!("http://{address}");
        let strategy = build(Some(&json!({
            "embedding": {
                "backend": "openai_compatible",
                "endpoint": {"base_url": format!("{endpoint}/v1"), "model": "embed"}
            },
            "reranker": {
                "backend": "openai_compatible",
                "endpoint": {"base_url": &endpoint, "model": "reranker"}
            },
            "controller": {
                "kind": "evidence_gap",
                "hits_per_round": 1,
                "max_rounds": 3,
                "confidence_threshold": 0.8,
                "backend": {
                    "backend": "openai_compatible",
                    "endpoint": {
                        "base_url": format!("{endpoint}/v1"),
                        "model": "controller",
                        "thinking_control": "no_control"
                    }
                },
                "generation": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 256,
                    "enable_thinking": false,
                    "validation_retries": 1
                }
            },
            "retrieval": {
                "dense_limit": 3,
                "sparse_limit": 3,
                "fused_limit": 3,
                "require_dense_index": true
            },
            "context": {
                "neighbor_before": 0,
                "neighbor_after": 0,
                "max_events": 3,
                "max_context_chars": 4096,
                "max_controller_chars": 4096
            },
            "embedding_batch_size": 2
        })))
        .unwrap();
        let store = SqliteStore::open_memory().await.unwrap();
        let events = [
            ("migration happened on Monday", 1),
            ("the database remained SQLite", 2),
            ("future unrelated evidence", 20),
        ]
        .into_iter()
        .map(|(content, timestamp)| {
            CanonicalEvent::message_from_text(
                EventSource::new(
                    AgentKind::Codex,
                    "session",
                    None,
                    None,
                    "message",
                    None,
                    Some("/repo".into()),
                ),
                MessageRole::Assistant,
                content,
                Some(timestamp),
            )
        })
        .collect::<Vec<_>>();
        let stats = store.ingest(&events).await.unwrap();
        let sequence = stats.domain_event_seq.unwrap();
        let mutations = strategy
            .on_event(
                &store,
                &StrategyEvent {
                    sequence,
                    events: events.clone(),
                },
            )
            .await
            .unwrap();
        store
            .apply_strategy_event(NAMESPACE, sequence, &mutations)
            .await
            .unwrap();

        let result = strategy
            .recall(
                &store,
                &RecallRequest {
                    query: SearchQuery {
                        text: "when was the migration and which database was used?".into(),
                        repository: Some("/repo".into()),
                        limit: 50,
                        ..SearchQuery::default()
                    },
                    query_time_ms: 10,
                },
            )
            .await
            .unwrap();
        assert_eq!(controller_calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            result.metadata["retrieval_usage"]["query_embedding_calls"],
            2
        );
        assert_eq!(result.metadata["retrieval_usage"]["reranker_calls"], 2);
        assert_eq!(result.metadata["processing"]["usage"]["calls"], 2);
        assert_eq!(result.metadata["processing"]["usage"]["cost"], 0.002);
        assert_eq!(
            result.metadata["controller_rounds"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        let RecallItem::Memory {
            content,
            source_event_ids,
            ..
        } = &result.items[0]
        else {
            panic!("reconstructive strategy must return one memory")
        };
        assert!(content.contains("migration happened on Monday"));
        assert!(content.contains("the database remained SQLite"));
        assert!(!content.contains("future unrelated evidence"));
        assert_eq!(source_event_ids.len(), 2);
        server.abort();
    }
}
