use super::{MemoryStrategy, RecallRequest, RecallResult, StrategyContext, StrategyEvent};
use crate::model::{MemoryMutation, QueryMode};
use anyhow::Result;
use async_trait::async_trait;

pub(super) struct HistoryStrategy;

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
            hits: context.query(QueryMode::Hybrid, &request.query).await?,
        })
    }
}
