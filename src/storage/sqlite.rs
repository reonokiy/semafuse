//! Durable SQLite storage for events, provenance, domain events, and memories.

use super::{EventStore, IngestStats, MemoryRepository};
use crate::model::{
    CanonicalEvent, ConversionDiagnostic, DiagnosticSeverity, DomainEventSequence, MemoryMutation,
    MemoryQuery, MemoryRecord, QueryMode, RawRecord, SearchHit, SearchQuery,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use sqlx::{
    Row, SqlitePool,
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions},
};
use std::{path::Path, str::FromStr};

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!();

/// Cloneable SQLite-backed store with transactional ingestion and strategy state.
#[derive(Clone)]
pub struct SqliteStore {
    pool: SqlitePool,
}

impl SqliteStore {
    /// Opens or creates a database, enables WAL, and runs bundled migrations.
    pub async fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(std::time::Duration::from_secs(5));
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .with_context(|| format!("open database {}", path.display()))?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }

    /// Opens a single-connection in-memory database and runs migrations.
    pub async fn open_memory() -> Result<Self> {
        let options = SqliteConnectOptions::from_str("sqlite::memory:")?.create_if_missing(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }
}

fn event_from_json(value: String) -> Result<CanonicalEvent> {
    serde_json::from_str(&value).context("decode formal canonical event")
}

impl SqliteStore {
    async fn ingest_formal(
        &self,
        raw_records: &[RawRecord],
        events: &[CanonicalEvent],
        diagnostics: &[ConversionDiagnostic],
    ) -> Result<IngestStats> {
        let mut tx = self.pool.begin().await?;
        let mut stats = IngestStats::default();
        for raw in raw_records {
            let payload_json = serde_json::to_string(&raw.payload)?;
            let result = sqlx::query(
                "INSERT OR IGNORE INTO raw_records(id,digest,agent,source,record_index,native_type,payload_json,ingested_at_ms) VALUES(?,?,?,?,?,?,?,?)",
            )
            .bind(&raw.id)
            .bind(&raw.digest)
            .bind(raw.agent.as_str())
            .bind(&raw.source)
            .bind(raw.record_index as i64)
            .bind(&raw.native_type)
            .bind(payload_json)
            .bind(raw.ingested_at_ms)
            .execute(&mut *tx)
            .await?;
            if result.rows_affected() == 1 {
                stats.raw_inserted += 1;
            } else {
                stats.raw_deduplicated += 1;
            }
        }
        for event in events {
            event.validate()?;
            let event_json = serde_json::to_string(event)?;
            let search_text = event.search_text();
            let ordering_timestamp_ms = event.ordering_timestamp_ms();
            let result = sqlx::query(
                "INSERT OR IGNORE INTO events(id,fingerprint,spec_version,agent,session_id,repository,event_type,occurred_at_ms,ordering_timestamp_ms,search_text,event_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            )
            .bind(&event.id)
            .bind(&event.fingerprint)
            .bind("semafuse.event/v1")
            .bind(event.source.agent.as_str())
            .bind(&event.source.session_id)
            .bind(&event.source.repository)
            .bind(event.event_type())
            .bind(event.occurred_at_ms)
            .bind(ordering_timestamp_ms)
            .bind(search_text)
            .bind(event_json)
            .execute(&mut *tx)
            .await?;
            if result.rows_affected() == 1 {
                stats.inserted += 1;
                stats.inserted_events.push(event.clone());
            } else {
                stats.deduplicated += 1;
            }
            for raw_record_id in &event.raw_record_ids {
                sqlx::query(
                    "INSERT OR IGNORE INTO event_raw_records(event_id,raw_record_id) SELECT ?,? WHERE EXISTS(SELECT 1 FROM raw_records WHERE id=?)",
                )
                .bind(&event.id)
                .bind(raw_record_id)
                .bind(raw_record_id)
                .execute(&mut *tx)
                .await?;
            }
        }
        for diagnostic in diagnostics {
            sqlx::query(
                "INSERT OR IGNORE INTO conversion_diagnostics(id,raw_record_id,severity,code,message,created_at_ms) VALUES(?,?,?,?,?,?)",
            )
            .bind(&diagnostic.id)
            .bind(&diagnostic.raw_record_id)
            .bind(match diagnostic.severity {
                DiagnosticSeverity::Warning => "warning",
                DiagnosticSeverity::Error => "error",
            })
            .bind(&diagnostic.code)
            .bind(&diagnostic.message)
            .bind(diagnostic.created_at_ms)
            .execute(&mut *tx)
            .await?;
        }
        if !stats.inserted_events.is_empty() {
            let ids = stats
                .inserted_events
                .iter()
                .map(|event| &event.id)
                .collect::<Vec<_>>();
            let payload_json = serde_json::to_string(&ids)?;
            let result = sqlx::query(
                "INSERT INTO domain_events(kind,payload_json,created_at_ms) VALUES('events_ingested',?,?)",
            )
            .bind(payload_json)
            .bind(crate::model::now_ms())
            .execute(&mut *tx)
            .await?;
            stats.domain_event_seq = Some(DomainEventSequence::from_database(
                result.last_insert_rowid(),
            ));
        }
        tx.commit().await?;
        Ok(stats)
    }
}

#[async_trait]
impl EventStore for SqliteStore {
    async fn ingest(&self, events: &[CanonicalEvent]) -> Result<IngestStats> {
        self.ingest_formal(&[], events, &[]).await
    }

    async fn ingest_batch(&self, batch: super::IngestBatch<'_>) -> Result<IngestStats> {
        self.ingest_formal(batch.raw_records, batch.events, batch.diagnostics)
            .await
    }

    async fn lexical_search(&self, query: &SearchQuery) -> Result<Vec<SearchHit>> {
        if query.text.trim().is_empty() {
            return self.recent(query).await;
        }
        let terms: Vec<String> = query
            .text
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .map(|s| format!("\"{}\"", s.replace('"', "\"\"")))
            .collect();
        let expression = if terms.is_empty() {
            format!("\"{}\"", query.text.replace('"', "\"\""))
        } else {
            terms.join(" OR ")
        };
        let agent = query.agent.map(|a| a.as_str());
        let limit = query.limit as i64;
        let rows = sqlx::query(
            "SELECT e.event_json,bm25(events_fts) AS rank FROM events_fts JOIN events e ON e.rowid=events_fts.rowid WHERE events_fts MATCH ? AND (? IS NULL OR e.repository=?) AND (? IS NULL OR e.agent=?) AND (? IS NULL OR e.ordering_timestamp_ms>=?) AND (? IS NULL OR e.ordering_timestamp_ms<=?) ORDER BY rank,e.ordering_timestamp_ms DESC LIMIT ?",
        )
        .bind(expression)
        .bind(query.repository.as_deref())
        .bind(query.repository.as_deref())
        .bind(agent)
        .bind(agent)
        .bind(query.since_ms)
        .bind(query.since_ms)
        .bind(query.until_ms)
        .bind(query.until_ms)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                let rank: f64 = row.try_get("rank")?;
                let event = event_from_json(row.try_get("event_json")?)?;
                Ok(SearchHit {
                    event,
                    score: 1.0 / (1.0 + rank.abs()),
                })
            })
            .collect()
    }

    async fn recent(&self, query: &SearchQuery) -> Result<Vec<SearchHit>> {
        let agent = query.agent.map(|a| a.as_str());
        let limit = query.limit as i64;
        let rows = sqlx::query(
            "SELECT event_json FROM events WHERE (? IS NULL OR repository=?) AND (? IS NULL OR agent=?) AND (? IS NULL OR ordering_timestamp_ms>=?) AND (? IS NULL OR ordering_timestamp_ms<=?) ORDER BY ordering_timestamp_ms DESC LIMIT ?",
        )
        .bind(query.repository.as_deref())
        .bind(query.repository.as_deref())
        .bind(agent)
        .bind(agent)
        .bind(query.since_ms)
        .bind(query.since_ms)
        .bind(query.until_ms)
        .bind(query.until_ms)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                Ok(SearchHit {
                    event: event_from_json(row.try_get("event_json")?)?,
                    score: 0.1,
                })
            })
            .collect()
    }

    async fn session_events(&self, session_id: &str) -> Result<Vec<CanonicalEvent>> {
        let rows = sqlx::query(
            "SELECT event_json FROM events WHERE session_id=? ORDER BY ordering_timestamp_ms",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| event_from_json(row.try_get("event_json")?))
            .collect()
    }

    async fn count(&self) -> Result<usize> {
        let row = sqlx::query("SELECT COUNT(*) AS count FROM events")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.try_get::<i64, _>("count")? as usize)
    }

    async fn query(&self, mode: QueryMode, query: &SearchQuery) -> Result<Vec<SearchHit>> {
        match mode {
            QueryMode::Lexical => self.lexical_search(query).await,
            QueryMode::Recent => self.recent(query).await,
            QueryMode::Hybrid => {
                crate::retrieval::strategy("hybrid")?
                    .retrieve(self, query)
                    .await
            }
        }
    }
}

struct MemoryRow {
    namespace: String,
    id: String,
    content: String,
    metadata_json: String,
    source_event_ids_json: String,
    created_at_ms: i64,
    updated_at_ms: i64,
}

fn memory_from_row(row: MemoryRow) -> Result<MemoryRecord> {
    Ok(MemoryRecord {
        namespace: row.namespace,
        id: row.id,
        content: row.content,
        metadata: serde_json::from_str(&row.metadata_json)?,
        source_event_ids: serde_json::from_str(&row.source_event_ids_json)?,
        created_at_ms: row.created_at_ms,
        updated_at_ms: row.updated_at_ms,
    })
}

#[async_trait]
impl MemoryRepository for SqliteStore {
    async fn get_memory(&self, namespace: &str, id: &str) -> Result<Option<MemoryRecord>> {
        let row = sqlx::query_as!(
            MemoryRow,
            "SELECT * FROM memories WHERE namespace=? AND id=?",
            namespace,
            id
        )
        .fetch_optional(&self.pool)
        .await?;
        row.map(memory_from_row).transpose()
    }

    async fn query_memories(&self, query: &MemoryQuery) -> Result<Vec<MemoryRecord>> {
        let limit = query.limit.clamp(1, 100) as i64;
        let rows = if query.text.trim().is_empty() {
            sqlx::query_as!(
                MemoryRow,
                "SELECT * FROM memories WHERE namespace=? ORDER BY updated_at_ms DESC LIMIT ?",
                query.namespace,
                limit
            )
            .fetch_all(&self.pool)
            .await?
        } else {
            let expression = format!("\"{}\"", query.text.replace('"', "\"\""));
            sqlx::query_as!(
                MemoryRow,
                "SELECT m.namespace AS 'namespace!', m.id AS 'id!', m.content AS 'content!', m.metadata_json AS 'metadata_json!', m.source_event_ids_json AS 'source_event_ids_json!', m.created_at_ms AS 'created_at_ms!', m.updated_at_ms AS 'updated_at_ms!' FROM memories_fts JOIN memories m ON m.rowid=memories_fts.rowid WHERE memories_fts MATCH ? AND m.namespace=? ORDER BY bm25(memories_fts),m.updated_at_ms DESC LIMIT ?",
                expression,
                query.namespace,
                limit
            )
            .fetch_all(&self.pool)
            .await?
        };
        rows.into_iter().map(memory_from_row).collect()
    }
}

impl SqliteStore {
    /// Returns a strategy's checkpoint, last error, and consecutive attempt count.
    ///
    /// An unseen strategy starts at [`DomainEventSequence::ORIGIN`] with no error.
    pub async fn strategy_checkpoint(
        &self,
        strategy: &str,
    ) -> Result<(DomainEventSequence, Option<String>, i64)> {
        let row = sqlx::query!(
            "SELECT last_seq,last_error,attempts FROM strategy_checkpoints WHERE strategy=?",
            strategy
        )
        .fetch_optional(&self.pool)
        .await?;
        Ok(row
            .map(|r| {
                (
                    DomainEventSequence::from_database(r.last_seq),
                    r.last_error,
                    r.attempts,
                )
            })
            .unwrap_or((DomainEventSequence::ORIGIN, None, 0)))
    }

    /// Returns domain-event batches strictly after `after`, in sequence order.
    pub async fn pending_domain_events(
        &self,
        after: DomainEventSequence,
    ) -> Result<Vec<(DomainEventSequence, Vec<String>)>> {
        let after = after.get();
        let rows = sqlx::query!(
            "SELECT seq,payload_json FROM domain_events WHERE seq>? ORDER BY seq",
            after
        )
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|r| {
                Ok((
                    DomainEventSequence::from_database(r.seq),
                    serde_json::from_str(&r.payload_json)?,
                ))
            })
            .collect()
    }

    /// Loads existing events in the exact order of the supplied IDs.
    ///
    /// IDs not present in the store are omitted.
    pub async fn events_by_ids(&self, ids: &[String]) -> Result<Vec<CanonicalEvent>> {
        let mut events = Vec::new();
        for id in ids {
            if let Some(row) = sqlx::query("SELECT event_json FROM events WHERE id=?")
                .bind(id)
                .fetch_optional(&self.pool)
                .await?
            {
                events.push(event_from_json(row.try_get("event_json")?)?);
            }
        }
        Ok(events)
    }

    /// Returns every event ordered by timestamp and then event ID.
    pub async fn all_events(&self) -> Result<Vec<CanonicalEvent>> {
        let rows = sqlx::query("SELECT event_json FROM events ORDER BY ordering_timestamp_ms,id")
            .fetch_all(&self.pool)
            .await?;
        rows.into_iter()
            .map(|row| event_from_json(row.try_get("event_json")?))
            .collect()
    }

    /// Loads one immutable native input record by ID.
    pub async fn raw_record(&self, id: &str) -> Result<Option<RawRecord>> {
        let row = sqlx::query(
            "SELECT id,digest,agent,source,record_index,native_type,payload_json,ingested_at_ms FROM raw_records WHERE id=?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        row.map(raw_record_from_row).transpose()
    }

    /// Loads every native input record linked to one canonical event.
    pub async fn raw_records_for_event(&self, event_id: &str) -> Result<Vec<RawRecord>> {
        let rows = sqlx::query(
            "SELECT r.id,r.digest,r.agent,r.source,r.record_index,r.native_type,r.payload_json,r.ingested_at_ms FROM event_raw_records p JOIN raw_records r ON r.id=p.raw_record_id WHERE p.event_id=? ORDER BY r.source,r.record_index",
        )
        .bind(event_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(raw_record_from_row).collect()
    }

    /// Returns the newest committed domain-event position, or origin when empty.
    pub async fn latest_domain_event_seq(&self) -> Result<DomainEventSequence> {
        let row = sqlx::query!("SELECT COALESCE(MAX(seq),0) AS 'seq!: i64' FROM domain_events")
            .fetch_one(&self.pool)
            .await?;
        Ok(DomainEventSequence::from_database(row.seq))
    }

    /// Returns every memory in one namespace in deterministic creation order.
    pub async fn all_memories(&self, namespace: &str) -> Result<Vec<MemoryRecord>> {
        let rows = sqlx::query_as!(
            MemoryRow,
            "SELECT * FROM memories WHERE namespace=? ORDER BY created_at_ms,id",
            namespace
        )
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(memory_from_row).collect()
    }

    /// Atomically replaces one strategy's namespace and advances its checkpoint.
    ///
    /// Every record must belong to `strategy`. Validation or persistence failure
    /// leaves both the old records and checkpoint intact; success clears failure state.
    pub async fn replace_strategy_state(
        &self,
        strategy: &str,
        records: &[MemoryRecord],
        checkpoint: DomainEventSequence,
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        sqlx::query!("DELETE FROM memories WHERE namespace=?", strategy)
            .execute(&mut *tx)
            .await?;
        for record in records {
            anyhow::ensure!(
                record.namespace == strategy,
                "strategy {strategy} cannot write namespace {}",
                record.namespace
            );
            let metadata_json = serde_json::to_string(&record.metadata)?;
            let source_event_ids_json = serde_json::to_string(&record.source_event_ids)?;
            sqlx::query!(
                "INSERT INTO memories(namespace,id,content,metadata_json,source_event_ids_json,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?,?,?)",
                record.namespace, record.id, record.content, metadata_json,
                source_event_ids_json, record.created_at_ms, record.updated_at_ms
            )
            .execute(&mut *tx)
            .await?;
        }
        let checkpoint = checkpoint.get();
        let now = crate::model::now_ms();
        sqlx::query!(
            "INSERT INTO strategy_checkpoints(strategy,last_seq,last_error,attempts,updated_at_ms) VALUES(?,?,NULL,0,?) ON CONFLICT(strategy) DO UPDATE SET last_seq=excluded.last_seq,last_error=NULL,attempts=0,updated_at_ms=excluded.updated_at_ms",
            strategy,
            checkpoint,
            now
        )
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Atomically applies mutations and advances a strategy checkpoint.
    ///
    /// Mutations may affect only the strategy's own namespace. The checkpoint
    /// advances and prior failure state clears only if every mutation commits.
    pub async fn apply_strategy_event(
        &self,
        strategy: &str,
        seq: DomainEventSequence,
        mutations: &[MemoryMutation],
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        for mutation in mutations {
            match mutation {
                MemoryMutation::Upsert(record) => {
                    anyhow::ensure!(
                        record.namespace == strategy,
                        "strategy {strategy} cannot write namespace {}",
                        record.namespace
                    );
                    let metadata_json = serde_json::to_string(&record.metadata)?;
                    let source_event_ids_json = serde_json::to_string(&record.source_event_ids)?;
                    sqlx::query!(
                        "INSERT INTO memories(namespace,id,content,metadata_json,source_event_ids_json,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?,?,?) ON CONFLICT(namespace,id) DO UPDATE SET content=excluded.content,metadata_json=excluded.metadata_json,source_event_ids_json=excluded.source_event_ids_json,updated_at_ms=excluded.updated_at_ms",
                        record.namespace, record.id, record.content, metadata_json,
                        source_event_ids_json, record.created_at_ms, record.updated_at_ms
                    )
                    .execute(&mut *tx)
                    .await?;
                }
                MemoryMutation::Delete { namespace, id } => {
                    anyhow::ensure!(
                        namespace == strategy,
                        "strategy {strategy} cannot delete namespace {namespace}"
                    );
                    sqlx::query!(
                        "DELETE FROM memories WHERE namespace=? AND id=?",
                        namespace,
                        id
                    )
                    .execute(&mut *tx)
                    .await?;
                }
            }
        }
        let seq = seq.get();
        let now = crate::model::now_ms();
        sqlx::query!(
            "INSERT INTO strategy_checkpoints(strategy,last_seq,last_error,attempts,updated_at_ms) VALUES(?,?,NULL,0,?) ON CONFLICT(strategy) DO UPDATE SET last_seq=excluded.last_seq,last_error=NULL,attempts=0,updated_at_ms=excluded.updated_at_ms",
            strategy,
            seq,
            now
        )
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Records a failed delivery without advancing its checkpoint.
    ///
    /// The latest error and attempt count are updated and an audit row is added
    /// for the currently checkpointed sequence in one transaction.
    pub async fn record_strategy_failure(&self, strategy: &str, error: &str) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        let now = crate::model::now_ms();
        sqlx::query!(
            "INSERT INTO strategy_checkpoints(strategy,last_seq,last_error,attempts,updated_at_ms) VALUES(?,0,?,1,?) ON CONFLICT(strategy) DO UPDATE SET last_error=excluded.last_error,attempts=attempts+1,updated_at_ms=excluded.updated_at_ms",
            strategy,
            error,
            now
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query!(
            "INSERT INTO strategy_failures(strategy,domain_event_seq,error,created_at_ms) SELECT strategy,last_seq,?,? FROM strategy_checkpoints WHERE strategy=?",
            error,
            now,
            strategy
        )
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }
}

fn raw_record_from_row(row: sqlx::sqlite::SqliteRow) -> Result<RawRecord> {
    Ok(RawRecord {
        id: row.try_get("id")?,
        digest: row.try_get("digest")?,
        agent: row.try_get::<String, _>("agent")?.parse()?,
        source: row.try_get("source")?,
        record_index: row.try_get::<i64, _>("record_index")? as u64,
        native_type: row.try_get("native_type")?,
        payload: serde_json::from_str(&row.try_get::<String, _>("payload_json")?)?,
        ingested_at_ms: row.try_get("ingested_at_ms")?,
    })
}
