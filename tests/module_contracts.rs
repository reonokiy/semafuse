use async_trait::async_trait;
use opendal::{Operator, services::Memory};
use semafuse::{
    AgentKind, AttachmentRef, CanonicalEvent, ContentPart, DomainEventSequence, EventSource,
    MemoryMutation, MemoryQuery, MemoryRecord, MessageRole, RawRecord, SearchQuery,
    adapters::history_importer,
    blob::{BlobStore, OpenDalBlobStore, drain_blob_gc},
    canonical_archive_schema,
    config::{BlobConfig, Config},
    hooks, retrieval,
    storage::{EventStore, ForgetFilter, IngestBatch, MemoryRepository, MemoryStore, SqliteStore},
};
use serde_json::Map;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::Notify;

fn event(content: &str, timestamp_ms: i64) -> CanonicalEvent {
    CanonicalEvent::message(
        EventSource::new(
            AgentKind::Codex,
            "contract-session",
            Some("turn-1".into()),
            None,
            "assistant_message",
            Some("/workspace/demo".into()),
            Some("/workspace/demo".into()),
        ),
        MessageRole::Assistant,
        vec![ContentPart::Text {
            text: content.into(),
        }],
        Some(timestamp_ms),
    )
}

async fn persist_event_with_raw(store: &SqliteStore, event: &mut CanonicalEvent) {
    let raw = RawRecord::new(
        event.source.agent,
        format!("contract:{}.json", event.id),
        0,
        serde_json::to_value(&*event).unwrap(),
    );
    event.attach_raw_record(raw.id.clone());
    store
        .ingest_batch(IngestBatch {
            raw_records: std::slice::from_ref(&raw),
            raw_contexts: &[],
            events: std::slice::from_ref(event),
            diagnostics: &[],
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn sqlite_persists_raw_source_truth_and_event_provenance() {
    let store = SqliteStore::open_memory().await.unwrap();
    let raw = RawRecord::new(
        AgentKind::Codex,
        "fixture.jsonl",
        7,
        serde_json::json!({"type":"future_event","nested":{"kept":true}}),
    );
    let mut projected = event("known projection", 10);
    projected.attach_raw_record(raw.id.clone());
    let stats = store
        .ingest_batch(IngestBatch {
            raw_records: std::slice::from_ref(&raw),
            raw_contexts: &[],
            events: std::slice::from_ref(&projected),
            diagnostics: &[],
        })
        .await
        .unwrap();
    assert_eq!(stats.raw_inserted, 1);
    assert_eq!(
        store.raw_record(&raw.id).await.unwrap().unwrap().payload,
        raw.payload
    );
    assert_eq!(
        store.raw_records_for_event(&projected.id).await.unwrap()[0].id,
        raw.id
    );
}

#[test]
fn checked_in_archive_schema_matches_the_rust_contract() {
    let checked_in: serde_json::Value =
        serde_json::from_str(include_str!("../schemas/canonical-archive-v1.schema.json")).unwrap();
    assert_eq!(checked_in, canonical_archive_schema());
    assert_eq!(
        checked_in["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
}

#[tokio::test]
async fn four_external_formats_convert_to_valid_formal_archives() {
    for (agent, fixture) in [
        (AgentKind::Amp, "tests/fixtures/amp-thread-complete.json"),
        (
            AgentKind::Codex,
            "tests/fixtures/codex-rollout-complete.jsonl",
        ),
        (
            AgentKind::ClaudeCode,
            "tests/fixtures/claude-session-complete.jsonl",
        ),
        (
            AgentKind::OpenCode,
            "tests/fixtures/opencode-session-complete.json",
        ),
    ] {
        let report = history_importer(agent)
            .import_path(std::path::Path::new(fixture), None)
            .await
            .unwrap();
        assert!(report.errors.is_empty(), "{agent}: {:?}", report.errors);
        let archive = report.canonical_archive();
        assert!(!archive.raw_records.is_empty());
        assert!(!archive.events.is_empty());
        assert_eq!(report.raw_contexts.len(), archive.raw_records.len());
        assert!(
            report
                .raw_contexts
                .iter()
                .all(|context| context.session_id.is_some()),
            "{agent} left raw records without session ownership"
        );
        archive.validate().unwrap();
    }
}

#[tokio::test]
async fn sqlite_emits_only_for_new_events_and_isolates_strategy_memory() {
    let store = SqliteStore::open_memory().await.unwrap();
    let item = event("a durable derived decision", 10);
    let first = store.ingest(std::slice::from_ref(&item)).await.unwrap();
    assert_eq!(first.inserted_events.len(), 1);
    assert_eq!(
        store
            .pending_domain_events(DomainEventSequence::ORIGIN)
            .await
            .unwrap()
            .len(),
        1
    );
    let second = store.ingest(std::slice::from_ref(&item)).await.unwrap();
    assert!(second.inserted_events.is_empty());
    assert!(second.domain_event_seq.is_none());

    let record = MemoryRecord {
        namespace: "test-strategy".into(),
        id: "decision-1".into(),
        content: "use the durable spool".into(),
        metadata: Map::new(),
        source_event_ids: vec![item.id],
        created_at_ms: 10,
        updated_at_ms: 10,
    };
    store
        .apply_strategy_event(
            "test-strategy",
            first.domain_event_seq.unwrap(),
            &[MemoryMutation::Upsert(record)],
        )
        .await
        .unwrap();
    let memories = store
        .query_memories(&MemoryQuery {
            namespace: "test-strategy".into(),
            text: "durable".into(),
            limit: 10,
        })
        .await
        .unwrap();
    assert_eq!(memories.len(), 1);
    assert!(
        store
            .apply_strategy_event(
                "other-strategy",
                first.domain_event_seq.unwrap(),
                &[MemoryMutation::Delete {
                    namespace: "test-strategy".into(),
                    id: "decision-1".into()
                }]
            )
            .await
            .is_err()
    );
    assert_eq!(
        store.strategy_checkpoint("test-strategy").await.unwrap().0,
        first.domain_event_seq.unwrap()
    );
}

async fn assert_store_contract(store: &dyn EventStore) {
    let events = [
        event("use sqlx for database access", 1),
        event("keep the durable spool", 2),
    ];
    let first = store.ingest(&events).await.unwrap();
    assert_eq!(first.inserted, 2);
    let second = store.ingest(&events).await.unwrap();
    assert_eq!(second.deduplicated, 2);
    assert_eq!(store.count().await.unwrap(), 2);
    let query = SearchQuery {
        text: "sqlx".into(),
        repository: Some("/workspace/demo".into()),
        agent: None,
        since_ms: None,
        until_ms: None,
        limit: 10,
    };
    assert_eq!(
        retrieval::strategy("lexical")
            .unwrap()
            .retrieve(store, &query)
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        retrieval::strategy("recent")
            .unwrap()
            .retrieve(store, &query)
            .await
            .unwrap()
            .len(),
        2
    );
    assert!(
        !retrieval::strategy("hybrid")
            .unwrap()
            .retrieve(store, &query)
            .await
            .unwrap()
            .is_empty()
    );
    let recent_only = SearchQuery {
        since_ms: Some(2),
        ..query.clone()
    };
    let hits = retrieval::strategy("recent")
        .unwrap()
        .retrieve(store, &recent_only)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].event.search_text(), "keep the durable spool");

    let first_only = SearchQuery {
        until_ms: Some(1),
        ..query
    };
    assert_eq!(
        retrieval::strategy("lexical")
            .unwrap()
            .retrieve(store, &first_only)
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn memory_and_sqlx_sqlite_implement_the_same_store_contract() {
    assert_store_contract(&MemoryStore::default()).await;
    assert_store_contract(&SqliteStore::open_memory().await.unwrap()).await;
}

#[tokio::test]
async fn opendal_backend_is_content_addressed_and_provider_neutral() {
    let operator = Operator::new(Memory::default()).unwrap().finish();
    let store = OpenDalBlobStore::from_operator(operator);
    let first = store
        .put(
            b"binary-image".to_vec(),
            "image/png",
            Some("screen.png".into()),
        )
        .await
        .unwrap();
    let second = store
        .put(b"binary-image".to_vec(), "image/png", None)
        .await
        .unwrap();
    assert_eq!(first.object_key, second.object_key);
    assert_eq!(store.get(&first.object_key).await.unwrap(), b"binary-image");
}

#[tokio::test]
async fn opendal_backend_is_created_from_dynamic_config() {
    let store = OpenDalBlobStore::from_config(&BlobConfig {
        scheme: "memory".into(),
        config: Default::default(),
    })
    .unwrap();
    let blob = store
        .put(
            b"dynamic-backend".to_vec(),
            "application/octet-stream",
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        store.get(&blob.object_key).await.unwrap(),
        b"dynamic-backend"
    );
}

#[tokio::test]
async fn blob_gc_preserves_references_and_isolates_databases_sharing_one_backend() {
    let temp = TempDir::new().unwrap();
    let config = BlobConfig {
        scheme: "fs".into(),
        config: [(
            "root".into(),
            temp.path().join("blobs").display().to_string(),
        )]
        .into_iter()
        .collect(),
    };
    let store_a = SqliteStore::open(&temp.path().join("a.db")).await.unwrap();
    let blobs_a = OpenDalBlobStore::from_config_for_store(&config, &store_a)
        .await
        .unwrap();
    let attachment_a = blobs_a
        .put(b"shared bytes".to_vec(), "application/octet-stream", None)
        .await
        .unwrap();
    let mut old = event("old attachment", 10);
    old.source.session_id = "old-attachment-session".into();
    old.add_attachments(vec![attachment_a.clone()]);
    persist_event_with_raw(&store_a, &mut old).await;
    store_a
        .forget(&ForgetFilter {
            session_id: Some("old-attachment-session".into()),
            ..ForgetFilter::default()
        })
        .await
        .unwrap();
    assert_eq!(store_a.pending_blob_gc_count().await.unwrap(), 1);

    let attachment_reused = blobs_a
        .put(b"shared bytes".to_vec(), "application/octet-stream", None)
        .await
        .unwrap();
    assert_eq!(attachment_reused.object_key, attachment_a.object_key);
    let mut current = event("current attachment", 20);
    current.source.session_id = "current-attachment-session".into();
    current.add_attachments(vec![attachment_reused]);
    persist_event_with_raw(&store_a, &mut current).await;
    assert_eq!(store_a.pending_blob_gc_count().await.unwrap(), 0);
    assert_eq!(
        drain_blob_gc(&store_a, &blobs_a, 10).await.unwrap().deleted,
        0
    );
    assert_eq!(
        blobs_a.get(&attachment_a.object_key).await.unwrap(),
        b"shared bytes"
    );

    let store_b = SqliteStore::open(&temp.path().join("b.db")).await.unwrap();
    let blobs_b = OpenDalBlobStore::from_config_for_store(&config, &store_b)
        .await
        .unwrap();
    let attachment_b = blobs_b
        .put(b"shared bytes".to_vec(), "application/octet-stream", None)
        .await
        .unwrap();
    assert_ne!(attachment_a.object_key, attachment_b.object_key);
    let mut event_b = event("database B attachment", 30);
    event_b.source.session_id = "database-b-session".into();
    event_b.add_attachments(vec![attachment_b.clone()]);
    persist_event_with_raw(&store_b, &mut event_b).await;

    store_a
        .forget(&ForgetFilter {
            session_id: Some("current-attachment-session".into()),
            ..ForgetFilter::default()
        })
        .await
        .unwrap();
    let report = drain_blob_gc(&store_a, &blobs_a, 10).await.unwrap();
    assert_eq!(report.deleted, 1);
    assert!(blobs_a.get(&attachment_a.object_key).await.is_err());
    assert_eq!(
        blobs_b.get(&attachment_b.object_key).await.unwrap(),
        b"shared bytes"
    );
}

struct BlockingBlobStore {
    entered: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait]
impl BlobStore for BlockingBlobStore {
    async fn put(
        &self,
        bytes: Vec<u8>,
        media_type: &str,
        name: Option<String>,
    ) -> anyhow::Result<AttachmentRef> {
        self.entered.notify_one();
        self.release.notified().await;
        Ok(AttachmentRef {
            object_key: "blocked/object".into(),
            digest: "blocked-digest".into(),
            media_type: media_type.into(),
            size: bytes.len() as u64,
            name,
        })
    }

    async fn get(&self, _key: &str) -> anyhow::Result<Vec<u8>> {
        unreachable!()
    }

    async fn delete(&self, _key: &str) -> anyhow::Result<()> {
        unreachable!()
    }

    fn backend_id(&self) -> &str {
        "blocking-test"
    }

    fn manages_key(&self, _key: &str) -> bool {
        true
    }
}

#[tokio::test]
async fn spool_append_during_replay_is_left_for_the_next_generation() {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().join("spool");
    hooks::append_spool(
        &dir,
        AgentKind::Codex,
        "Stop",
        &serde_json::json!({
            "session_id":"first", "last_assistant_message":"first",
            "content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}]
        }),
    )
    .unwrap();

    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let blobs = Arc::new(BlockingBlobStore {
        entered: entered.clone(),
        release: release.clone(),
    });
    let replay_dir = dir.clone();
    let store = SqliteStore::open_memory().await.unwrap();
    let replay_store = store.clone();
    let task = tokio::spawn(async move {
        hooks::replay_spool(
            &replay_dir,
            &replay_store,
            Some(&*blobs),
            &Config::default(),
        )
        .await
    });
    entered.notified().await;
    hooks::append_spool(
        &dir,
        AgentKind::Codex,
        "Stop",
        &serde_json::json!({
            "session_id":"second", "last_assistant_message":"second"
        }),
    )
    .unwrap();
    release.notify_one();
    assert_eq!(task.await.unwrap().unwrap(), 1);
    assert!(dir.join("hooks.jsonl").is_file());

    let memory_blobs =
        OpenDalBlobStore::from_operator(Operator::new(Memory::default()).unwrap().finish());
    assert_eq!(
        hooks::replay_spool(&dir, &store, Some(&memory_blobs), &Config::default())
            .await
            .unwrap(),
        1
    );
    assert_eq!(store.count().await.unwrap(), 2);
}
