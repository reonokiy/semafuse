use super::{
    MemoryStrategy, RecallItem, RecallRequest, RecallResult, StrategyContext, StrategyEvent,
    registry::StrategyRegistration,
};
use crate::model::{MemoryMutation, QueryMode};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

pub(super) const REGISTRATION: StrategyRegistration = StrategyRegistration::new("history", build);

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct HistoryConfig {
    retrieval: QueryMode,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            retrieval: QueryMode::Hybrid,
        }
    }
}

pub(super) struct HistoryStrategy {
    config: HistoryConfig,
}

fn build(settings: Option<&Value>) -> Result<Box<dyn MemoryStrategy>> {
    let config = settings
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .context("deserialize history strategy settings")?
        .unwrap_or_default();
    Ok(Box::new(HistoryStrategy { config }))
}

#[async_trait]
impl MemoryStrategy for HistoryStrategy {
    fn name(&self) -> &'static str {
        "history"
    }

    async fn on_event(
        &self,
        _context: &dyn StrategyContext,
        _event: &StrategyEvent,
    ) -> Result<Vec<MemoryMutation>> {
        Ok(Vec::new())
    }

    async fn recall(
        &self,
        context: &dyn StrategyContext,
        request: &RecallRequest,
    ) -> Result<RecallResult> {
        Ok(RecallResult {
            items: context
                .query(self.config.retrieval, &request.query)
                .await?
                .into_iter()
                .map(RecallItem::from)
                .collect(),
        })
    }
}
