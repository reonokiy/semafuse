//! Line-delimited JSON-RPC server implementing Semafuse's MCP interface.

use crate::{
    config,
    model::{QueryMode, SearchQuery},
    storage::{EventStore, SqliteStore},
    strategy,
};
use anyhow::Result;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Serves MCP over standard input and standard output until input reaches EOF.
///
/// The server exposes `semafuse_search` for raw lexical/recent/hybrid search,
/// `semafuse_recall` for configured strategy recall, `semafuse_session` for a
/// session timeline, and `semafuse_recent` for recent events. Malformed lines
/// and notifications without IDs are ignored; tool failures are returned as MCP
/// error content while unknown methods receive JSON-RPC method-not-found.
pub async fn serve(store: &SqliteStore) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let Some(id) = request.get("id").cloned() else {
            continue;
        };
        let method = request
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let result = match method {
            "initialize" => json!({"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"semafuse","version":env!("CARGO_PKG_VERSION")}}),
            "ping" => json!({}),
            "tools/list" => json!({"tools":[
                {"name":"semafuse_search","description":"Search raw coding-agent history with provenance","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"repository":{"type":"string"},"mode":{"type":"string","enum":["lexical","recent","hybrid"]},"limit":{"type":"integer"}},"required":["query"]}},
                {"name":"semafuse_recall","description":"Recall context through the configured memory strategy","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"repository":{"type":"string"},"limit":{"type":"integer"}},"required":["query"]}},
                {"name":"semafuse_session","description":"Read normalized events for a prior session","inputSchema":{"type":"object","properties":{"session_id":{"type":"string"}},"required":["session_id"]}},
                {"name":"semafuse_recent","description":"List recent normalized coding-agent events","inputSchema":{"type":"object","properties":{"repository":{"type":"string"},"limit":{"type":"integer"}}}}
            ]}),
            "tools/call" => tool_call(store, request.get("params").unwrap_or(&Value::Null)).await.unwrap_or_else(|error| json!({"content":[{"type":"text","text":format!("Semafuse error: {error:#}")}],"isError":true})),
            _ => {
                let response = json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":format!("method not found: {method}")}});
                stdout.write_all(format!("{}\n", response).as_bytes()).await?;
                stdout.flush().await?;
                continue;
            }
        };
        let response = json!({"jsonrpc":"2.0","id":id,"result":result});
        stdout
            .write_all(format!("{}\n", response).as_bytes())
            .await?;
        stdout.flush().await?;
    }
    Ok(())
}

async fn tool_call(store: &SqliteStore, params: &Value) -> Result<Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let args = params.get("arguments").unwrap_or(&Value::Null);
    let value = match name {
        "semafuse_session" => {
            let id = args
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            serde_json::to_value(store.session_events(id).await?)?
        }
        "semafuse_recent" => {
            let query = SearchQuery {
                text: String::new(),
                repository: nonempty(args.get("repository")),
                agent: None,
                limit: limit(args),
            };
            serde_json::to_value(store.recent(&query).await?)?
        }
        "semafuse_search" | "semafuse_recall" => {
            let query = SearchQuery {
                text: args
                    .get("query")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                repository: nonempty(args.get("repository")),
                agent: None,
                limit: limit(args),
            };
            let mode = args.get("mode").and_then(Value::as_str).unwrap_or("hybrid");
            let hits = if name == "semafuse_recall" {
                strategy::recall(store, &config::load()?, query).await?
            } else {
                store.query(mode.parse::<QueryMode>()?, &query).await?
            };
            serde_json::to_value(hits)?
        }
        _ => anyhow::bail!("unknown MCP tool: {name}"),
    };
    Ok(
        json!({"content":[{"type":"text","text":serde_json::to_string_pretty(&value)?}],"structuredContent":{"results":value}}),
    )
}

fn nonempty(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}
fn limit(args: &Value) -> usize {
    args.get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(10)
        .clamp(1, 100) as usize
}
