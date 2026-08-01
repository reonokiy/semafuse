use config::{File, FileFormat};
use semafuse::{
    AgentKind, CanonicalEvent, EventSource, MessageRole, RawRecord, RecallItem, SearchQuery,
    adapters::{ClaudeAdapter, HookProjector},
    config::Config,
    storage::{EventStore, IngestBatch, SqliteStore},
    strategy,
};
use serde_json::json;

fn message(content: &str, timestamp_ms: i64) -> CanonicalEvent {
    CanonicalEvent::message_from_text(
        EventSource::new(
            AgentKind::Codex,
            "session-a",
            None,
            None,
            "assistant_message",
            Some("/workspace/demo".into()),
            Some("/workspace/demo".into()),
        ),
        MessageRole::Assistant,
        content,
        Some(timestamp_ms),
    )
}

async fn ingest(store: &SqliteStore, mut events: Vec<CanonicalEvent>) -> Vec<CanonicalEvent> {
    let raws = events
        .iter_mut()
        .enumerate()
        .map(|(index, event)| {
            let raw = RawRecord::new(
                event.source.agent,
                format!("strategy-test:{}.jsonl", event.source.session_id),
                index as u64,
                serde_json::to_value(&*event).unwrap(),
            );
            event.attach_raw_record(raw.id.clone());
            raw
        })
        .collect::<Vec<_>>();
    store
        .ingest_batch(IngestBatch {
            raw_records: &raws,
            raw_contexts: &[],
            events: &events,
            diagnostics: &[],
        })
        .await
        .unwrap();
    events
}

fn history_config() -> Config {
    let mut config = Config::default();
    config.strategies.enabled = vec!["history".into()];
    config.strategies.recall = "history".into();
    config
        .strategies
        .settings
        .insert("history".into(), json!({"retrieval": "lexical"}));
    config
}

fn query() -> SearchQuery {
    SearchQuery {
        text: "migration".into(),
        repository: Some("/workspace/demo".into()),
        agent: None,
        since_ms: None,
        until_ms: None,
        limit: 5,
    }
}

#[test]
fn strategy_modules_own_and_validate_their_settings() {
    let parsed: Config = config::Config::builder()
        .add_source(File::from_str(
            r#"
                [strategies]
                enabled = ["history"]
                recall = "history"

                [strategies.settings.history]
                retrieval = "lexical"
            "#,
            FileFormat::Toml,
        ))
        .build()
        .unwrap()
        .try_deserialize()
        .unwrap();
    strategy::validate(&parsed).unwrap();

    let mut typo = parsed.clone();
    typo.strategies
        .settings
        .insert("history".into(), json!({"retrievall":"lexical"}));
    assert!(
        strategy::validate(&typo)
            .unwrap_err()
            .to_string()
            .contains("invalid settings for memory strategy history")
    );

    let mut removed = parsed.clone();
    removed.strategies.enabled = vec!["lazy".into()];
    removed.strategies.recall = "lazy".into();
    removed.strategies.settings.clear();
    assert!(
        strategy::validate(&removed)
            .unwrap_err()
            .to_string()
            .contains("unknown memory strategy: lazy")
    );

    let mut unknown = parsed;
    unknown
        .strategies
        .settings
        .insert("not-registered".into(), json!({}));
    assert!(
        strategy::validate(&unknown)
            .unwrap_err()
            .to_string()
            .contains("unknown memory strategy: not-registered")
    );

    let selector_typo = config::Config::builder()
        .add_source(File::from_str(
            r#"
                [strategies]
                enabled = ["history"]
                recalll = "history"
            "#,
            FileFormat::Toml,
        ))
        .build()
        .unwrap()
        .try_deserialize::<Config>();
    assert!(selector_typo.is_err());
}

#[tokio::test]
async fn history_remains_an_independent_recall_module() {
    let store = SqliteStore::open_memory().await.unwrap();
    let events = ingest(
        &store,
        vec![
            message("Decision: keep SQLite as the canonical store.", 10),
            message("Implemented the migration for memory indexes.", 20),
            message("Constraint: every memory must cite provenance.", 30),
        ],
    )
    .await;
    let config = history_config();

    assert_eq!(
        strategy::dispatch_pending(&store, &config, None)
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        strategy::dispatch_pending(&store, &config, None)
            .await
            .unwrap(),
        0
    );

    let recalled = strategy::recall(&store, &config, query()).await.unwrap();
    assert_eq!(recalled.len(), 1);
    let RecallItem::Event { event, .. } = &recalled[0] else {
        panic!("history strategy must emit an event")
    };
    assert_eq!(event.id, events[1].id);
    assert_eq!(serde_json::to_value(&recalled[0]).unwrap()["kind"], "event");
}

#[test]
fn hook_formatting_accepts_strategy_derived_memories() {
    let item = RecallItem::Memory {
        namespace: "lazymem".into(),
        id: "qry_example".into(),
        content: "Untrusted evidence\n- [sem:evt_one] remembered constraint".into(),
        metadata: Default::default(),
        source_event_ids: vec!["evt_one".into()],
        score: 1.0,
    };
    let response = ClaudeAdapter
        .recall_response("UserPromptSubmit", &[item])
        .unwrap();
    let context = response["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert!(context.contains("sem-memory:lazymem/qry_example"));
    assert!(context.contains("sources=evt_one"));
    assert!(context.contains("remembered constraint"));
}
