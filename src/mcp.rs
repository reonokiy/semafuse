//! Model Context Protocol server implemented with the official Rust SDK.

use crate::{
    config,
    model::{QueryMode, SearchQuery},
    storage::{EventStore, SqliteStore},
    strategy,
};
use anyhow::Result;
use rmcp::{
    ErrorData as McpError, ServerHandler, ServiceExt,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    schemars::JsonSchema,
    tool, tool_handler, tool_router,
    transport::stdio,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone)]
struct SemafuseServer {
    store: SqliteStore,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchArgs {
    /// Full-text query to search for.
    query: String,
    /// Exact normalized repository filter.
    repository: Option<String>,
    /// Retrieval mode: lexical, recent, or hybrid.
    mode: Option<String>,
    /// Inclusive lower timestamp bound in Unix milliseconds.
    since_ms: Option<i64>,
    /// Inclusive upper timestamp bound in Unix milliseconds.
    until_ms: Option<i64>,
    /// Maximum number of results, clamped to 1–100.
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RecallArgs {
    /// Full-text query to recall context for.
    query: String,
    /// Exact normalized repository filter.
    repository: Option<String>,
    /// Maximum number of results, clamped to 1–100.
    limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SessionArgs {
    /// Native session identifier.
    session_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RecentArgs {
    /// Exact normalized repository filter.
    repository: Option<String>,
    /// Inclusive lower timestamp bound in Unix milliseconds.
    since_ms: Option<i64>,
    /// Inclusive upper timestamp bound in Unix milliseconds.
    until_ms: Option<i64>,
    /// Maximum number of results, clamped to 1–100.
    limit: Option<usize>,
}

#[tool_router]
impl SemafuseServer {
    #[tool(
        name = "semafuse_search",
        description = "Search raw coding-agent history with provenance"
    )]
    async fn search(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let mode = args
            .mode
            .as_deref()
            .unwrap_or("hybrid")
            .parse::<QueryMode>()
            .map_err(|error| McpError::invalid_params(error.to_string(), None))?;
        let query = SearchQuery {
            text: args.query,
            repository: nonempty(args.repository),
            agent: None,
            since_ms: args.since_ms,
            until_ms: args.until_ms,
            limit: limit(args.limit),
        };
        Ok(tool_result(self.store.query(mode, &query).await))
    }

    #[tool(
        name = "semafuse_recall",
        description = "Recall context through the configured memory strategy"
    )]
    async fn recall(
        &self,
        Parameters(args): Parameters<RecallArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let query = SearchQuery {
            text: args.query,
            repository: nonempty(args.repository),
            agent: None,
            since_ms: None,
            until_ms: None,
            limit: limit(args.limit),
        };
        let result = match config::load() {
            Ok(config) => strategy::recall(&self.store, &config, query).await,
            Err(error) => Err(error),
        };
        Ok(tool_result(result))
    }

    #[tool(
        name = "semafuse_session",
        description = "Read normalized events for a prior session"
    )]
    async fn session(
        &self,
        Parameters(args): Parameters<SessionArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        Ok(tool_result(
            self.store.session_events(&args.session_id).await,
        ))
    }

    #[tool(
        name = "semafuse_recent",
        description = "List recent normalized coding-agent events"
    )]
    async fn recent(
        &self,
        Parameters(args): Parameters<RecentArgs>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let query = SearchQuery {
            text: String::new(),
            repository: nonempty(args.repository),
            agent: None,
            since_ms: args.since_ms,
            until_ms: args.until_ms,
            limit: limit(args.limit),
        };
        Ok(tool_result(self.store.recent(&query).await))
    }
}

#[tool_handler(name = "semafuse")]
impl ServerHandler for SemafuseServer {}

/// Serves Semafuse's MCP tools over standard input and standard output.
///
/// The official SDK owns JSON-RPC framing, lifecycle negotiation, tool routing,
/// and protocol errors. Application failures are returned as caller-visible MCP
/// tool errors, while malformed arguments remain JSON-RPC invalid-params errors.
pub async fn serve(store: &SqliteStore) -> Result<()> {
    let service = SemafuseServer {
        store: store.clone(),
    }
    .serve(stdio())
    .await?;
    service.waiting().await?;
    Ok(())
}

fn tool_result<T: Serialize>(result: Result<T>) -> CallToolResult {
    match result.and_then(|results| {
        let results = serde_json::to_value(results)?;
        let text = serde_json::to_string_pretty(&results)?;
        let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
        result.structured_content = Some(json!({ "results": results }));
        Ok(result)
    }) {
        Ok(result) => result,
        Err(error) => CallToolResult::error(vec![ContentBlock::text(format!(
            "Semafuse error: {error:#}"
        ))]),
    }
}

fn nonempty(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.is_empty())
}

fn limit(value: Option<usize>) -> usize {
    value.unwrap_or(10).clamp(1, 100)
}
