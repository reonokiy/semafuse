use async_trait::async_trait;
use opendal::{Operator, services::Memory};
use semafuse::{
    AgentKind, AttachmentRef, CanonicalEvent, DomainEventSequence, EventKind, MemoryMutation,
    MemoryQuery, MemoryRecord, RawRecord, SearchQuery,
    adapters::adapter,
    blob::{BlobStore, OpenDalBlobStore},
    canonical_archive_schema,
    config::{BlobConfig, Config},
    hooks, importer, retrieval,
    storage::{EventStore, IngestBatch, MemoryRepository, MemoryStore, SqliteStore},
};
use serde_json::Map;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::Notify;

fn event(content: &str, timestamp_ms: i64) -> CanonicalEvent {
    CanonicalEvent::new(
        AgentKind::Codex,
        "contract-session",
        Some("turn-1".into()),
        None,
        EventKind::AssistantMessage,
        Some("assistant".into()),
        content,
        None,
        Some("/workspace/demo".into()),
        Some("/workspace/demo".into()),
        Some(timestamp_ms),
        Map::new(),
    )
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
        let report = importer::import_path(
            &*adapter(agent),
            std::path::Path::new(fixture),
            None,
            &mut |_, _, _| {},
        )
        .await
        .unwrap();
        assert!(report.errors.is_empty(), "{agent}: {:?}", report.errors);
        let archive = report.canonical_archive();
        assert!(!archive.raw_records.is_empty());
        assert!(!archive.events.is_empty());
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
