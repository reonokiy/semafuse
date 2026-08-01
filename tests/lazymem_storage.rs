use semafuse::{
    AgentKind, CanonicalEvent, EventSource, MemoryMutation, MessageRole, SearchQuery,
    StrategyEventVector, VectorQuery,
    storage::{EventStore, ForgetFilter, SqliteStore, VectorRepository},
};

fn message(session: &str, text: &str, timestamp_ms: i64) -> CanonicalEvent {
    CanonicalEvent::message_from_text(
        EventSource::new(
            AgentKind::Codex,
            session,
            None,
            None,
            "message",
            None,
            Some("/repo".into()),
        ),
        MessageRole::User,
        text,
        Some(timestamp_ms),
    )
}

#[tokio::test]
async fn vector_artifacts_are_scoped_ranked_and_forgotten_with_their_events() {
    let store = SqliteStore::open_memory().await.unwrap();
    let first = message("session", "alpha", 20);
    let second = message("session", "beta", 10);
    let stats = store
        .ingest(&[first.clone(), second.clone()])
        .await
        .unwrap();
    store
        .apply_strategy_event(
            "lazymem",
            stats.domain_event_seq.unwrap(),
            &[
                MemoryMutation::UpsertVector(StrategyEventVector {
                    namespace: "lazymem".into(),
                    event_id: first.id.clone(),
                    model: "embedding@revision".into(),
                    values: vec![1.0, 0.0],
                }),
                MemoryMutation::UpsertVector(StrategyEventVector {
                    namespace: "lazymem".into(),
                    event_id: second.id.clone(),
                    model: "embedding@revision".into(),
                    values: vec![0.0, 1.0],
                }),
            ],
        )
        .await
        .unwrap();

    let hits = store
        .query_vectors(&VectorQuery {
            namespace: "lazymem".into(),
            model: "embedding@revision".into(),
            values: vec![0.9, 0.1],
            scope: SearchQuery {
                repository: Some("/repo".into()),
                limit: 10,
                ..SearchQuery::default()
            },
        })
        .await
        .unwrap();
    assert_eq!(
        hits.iter()
            .map(|hit| hit.event.id.as_str())
            .collect::<Vec<_>>(),
        [first.id.as_str(), second.id.as_str()]
    );

    store
        .forget(&ForgetFilter {
            session_id: Some("session".into()),
            ..ForgetFilter::default()
        })
        .await
        .unwrap();
    assert!(
        store
            .all_strategy_vectors("lazymem")
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn vector_dimensions_cannot_drift_within_one_namespace_and_model() {
    let store = SqliteStore::open_memory().await.unwrap();
    let first = message("session", "alpha", 10);
    let first_stats = store.ingest(std::slice::from_ref(&first)).await.unwrap();
    store
        .apply_strategy_event(
            "lazymem",
            first_stats.domain_event_seq.unwrap(),
            &[MemoryMutation::UpsertVector(StrategyEventVector {
                namespace: "lazymem".into(),
                event_id: first.id.clone(),
                model: "embedding@revision".into(),
                values: vec![1.0, 0.0],
            })],
        )
        .await
        .unwrap();

    let second = message("session", "beta", 20);
    let second_stats = store.ingest(std::slice::from_ref(&second)).await.unwrap();
    let error = store
        .apply_strategy_event(
            "lazymem",
            second_stats.domain_event_seq.unwrap(),
            &[MemoryMutation::UpsertVector(StrategyEventVector {
                namespace: "lazymem".into(),
                event_id: second.id,
                model: "embedding@revision".into(),
                values: vec![1.0, 0.0, 0.0],
            })],
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("dimensions changed"));
    let (checkpoint, _, _) = store.strategy_checkpoint("lazymem").await.unwrap();
    assert_eq!(checkpoint, first_stats.domain_event_seq.unwrap());
}

#[tokio::test]
async fn scoped_lazymem_corpus_excludes_non_message_events_before_ranking() {
    let store = SqliteStore::open_memory().await.unwrap();
    let message = message("session", "needle", 10);
    let compaction = CanonicalEvent::compaction_from_content(
        EventSource::new(
            AgentKind::Codex,
            "session",
            None,
            Some("compaction".into()),
            "compaction",
            None,
            Some("/repo".into()),
        ),
        "needle summary",
        Some(11),
    );
    store.ingest(&[message.clone(), compaction]).await.unwrap();
    let corpus = store
        .scoped_message_events(&SearchQuery {
            text: "needle".into(),
            repository: Some("/repo".into()),
            limit: 1,
            ..SearchQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(corpus.len(), 1);
    assert_eq!(corpus[0].id, message.id);
}

#[tokio::test]
async fn ordered_session_events_follow_ingestion_not_native_clock_order() {
    let store = SqliteStore::open_memory().await.unwrap();
    let first = message("session", "written first", 20);
    let second = message("session", "written second", 10);
    store
        .ingest(&[first.clone(), second.clone()])
        .await
        .unwrap();

    let ordered = store.ordered_session_events("session").await.unwrap();
    assert_eq!(ordered.len(), 2);
    assert_eq!(ordered[0].event.id, first.id);
    assert_eq!(ordered[1].event.id, second.id);
    assert!(ordered[0].ordinal < ordered[1].ordinal);
}
