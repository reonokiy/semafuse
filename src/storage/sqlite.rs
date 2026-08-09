//! Durable SQLite storage for events, provenance, domain events, and memories.

use super::{
    BlobGcEntry, EventStore, ForgetFilter, ForgetPreview, ForgetStats, IngestStats,
    MemoryRepository, OrderedEvent, RawContext, StrategyStateRepository, VectorRepository,
};
use crate::adapters::native_raw_context;
use crate::model::{
    AgentKind, ArchiveSpecVersion, CanonicalArchive, CanonicalEvent, ConversionDiagnostic,
    DiagnosticSeverity, DomainEventSequence, MemoryMutation, MemoryQuery, MemoryRecord, QueryMode,
    RawRecord, SearchHit, SearchQuery, StrategyEventVector, VectorHit, VectorQuery,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use fs2::FileExt;
use sha2::{Digest, Sha256};
use sqlx::{
    QueryBuilder, Row, Sqlite, SqlitePool,
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions},
};
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    str::FromStr,
};

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!();

/// Cloneable SQLite-backed store with transactional ingestion and strategy state.
#[derive(Clone)]
pub struct SqliteStore {
    pool: SqlitePool,
    lifecycle_lock: Option<PathBuf>,
}

/// Process-safe guard serializing blob writes, deletion, exports, and forget operations.
pub struct LifecycleGuard {
    file: Option<File>,
}

impl Drop for LifecycleGuard {
    fn drop(&mut self) {
        if let Some(file) = &self.file {
            let _ = file.unlock();
        }
    }
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
        let created = prepare_private_database(path)?;
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .foreign_keys(true)
            .pragma("secure_delete", "ON")
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(std::time::Duration::from_secs(5));
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .with_context(|| format!("open database {}", path.display()))?;
        MIGRATOR.run(&pool).await?;
        if created {
            enforce_private_database(path)?;
        }
        let canonical_path = path
            .canonicalize()
            .with_context(|| format!("resolve database location {}", path.display()))?;
        let mut lock_name = canonical_path.as_os_str().to_os_string();
        lock_name.push(".lifecycle.lock");
        let store = Self {
            pool,
            lifecycle_lock: Some(PathBuf::from(lock_name)),
        };
        store.bind_database_location(&canonical_path).await?;
        store.backfill_governance_indexes().await?;
        Ok(store)
    }

    /// Opens a single-connection in-memory database and runs migrations.
    pub async fn open_memory() -> Result<Self> {
        let options = SqliteConnectOptions::from_str("sqlite::memory:")?
            .create_if_missing(true)
            .foreign_keys(true)
            .pragma("secure_delete", "ON");
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await?;
        MIGRATOR.run(&pool).await?;
        let store = Self {
            pool,
            lifecycle_lock: None,
        };
        store.backfill_governance_indexes().await?;
        Ok(store)
    }

    /// Acquires the cross-process lifecycle lock used around blob/SQLite transitions.
    pub fn lock_lifecycle(&self) -> Result<LifecycleGuard> {
        let Some(path) = &self.lifecycle_lock else {
            return Ok(LifecycleGuard { file: None });
        };
        let file = open_private_lock(path)?;
        lock_exclusive_wait(&file)
            .with_context(|| format!("lock lifecycle file {}", path.display()))?;
        Ok(LifecycleGuard { file: Some(file) })
    }

    /// Attempts the lifecycle lock without waiting, allowing hooks to spool promptly.
    pub fn try_lock_lifecycle(&self) -> Result<Option<LifecycleGuard>> {
        let Some(path) = &self.lifecycle_lock else {
            return Ok(Some(LifecycleGuard { file: None }));
        };
        let file = open_private_lock(path)?;
        match file.try_lock_exclusive() {
            Ok(()) => Ok(Some(LifecycleGuard { file: Some(file) })),
            Err(error) if lock_is_contended(&error) => Ok(None),
            Err(error) => {
                Err(error).with_context(|| format!("try lifecycle lock {}", path.display()))
            }
        }
    }

    /// Returns this database's persistent attachment namespace identifier.
    pub async fn storage_id(&self) -> Result<String> {
        let row = sqlx::query("SELECT value FROM storage_metadata WHERE key='storage_id'")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.try_get("value")?)
    }

    async fn bind_database_location(&self, canonical: &Path) -> Result<()> {
        let sidecar_identity = database_location_identity(canonical)?;
        let location_hash = selector_hash(&format!(
            "{}\0{sidecar_identity}",
            canonical.to_string_lossy()
        ));
        let legacy_location_hash = selector_hash(canonical.to_string_lossy().as_ref());
        let mut tx = self.pool.begin().await?;
        let existing =
            sqlx::query("SELECT value FROM storage_metadata WHERE key='database_location_hash'")
                .fetch_optional(&mut *tx)
                .await?
                .map(|row| row.try_get::<String, _>("value"))
                .transpose()?;
        let version = sqlx::query(
            "SELECT value FROM storage_metadata WHERE key='database_location_identity_version'",
        )
        .fetch_optional(&mut *tx)
        .await?
        .map(|row| row.try_get::<String, _>("value"))
        .transpose()?;
        let changed = if let Some(existing) = existing {
            if version.as_deref() == Some("sidecar-v1") {
                anyhow::ensure!(
                    existing == location_hash,
                    "database copy or relocation detected at {}; raw SQLite copies cannot share an attachment namespace; use `semafuse export` or restore the database together with its location identity",
                    canonical.display()
                );
                false
            } else {
                anyhow::ensure!(
                    existing == legacy_location_hash,
                    "database copy or relocation detected at {}; raw SQLite copies cannot share an attachment namespace; use `semafuse export` or restore the database at its original canonical path",
                    canonical.display()
                );
                sqlx::query(
                    "UPDATE storage_metadata SET value=? WHERE key='database_location_hash'",
                )
                .bind(&location_hash)
                .execute(&mut *tx)
                .await?;
                sqlx::query(
                    "INSERT INTO storage_metadata(key,value) VALUES('database_location_identity_version','sidecar-v1')",
                )
                .execute(&mut *tx)
                .await?;
                true
            }
        } else {
            sqlx::query(
                "INSERT INTO storage_metadata(key,value) VALUES('database_location_hash',?)",
            )
            .bind(location_hash)
            .execute(&mut *tx)
            .await?;
            sqlx::query(
                "INSERT INTO storage_metadata(key,value) VALUES('database_location_identity_version','sidecar-v1')",
            )
            .execute(&mut *tx)
            .await?;
            true
        };
        if changed {
            sqlx::query(
                "INSERT INTO storage_metadata(key,value) VALUES('database_location_checkpoint_pending','1') ON CONFLICT(key) DO UPDATE SET value='1'",
            )
            .execute(&mut *tx)
            .await?;
        }
        let checkpoint_pending = changed
            || sqlx::query(
                "SELECT EXISTS(SELECT 1 FROM storage_metadata WHERE key='database_location_checkpoint_pending') AS pending",
            )
            .fetch_one(&mut *tx)
            .await?
            .try_get::<i64, _>("pending")?
                != 0;
        tx.commit().await?;
        if checkpoint_pending {
            // The clone guard must be present in the main database file, not
            // only in a WAL that a naive copy might omit. The pending marker
            // remains durable across a busy checkpoint and forces a later
            // open to retry this step.
            let row = sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
                .fetch_one(&self.pool)
                .await?;
            anyhow::ensure!(
                row.try_get::<i64, _>(0)? == 0,
                "could not durably checkpoint database location identity"
            );
            sqlx::query(
                "DELETE FROM storage_metadata WHERE key='database_location_checkpoint_pending'",
            )
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    /// Binds a database to exactly one attachment backend fingerprint.
    pub async fn bind_blob_backend(&self, backend_id: &str) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        let existing =
            sqlx::query("SELECT value FROM storage_metadata WHERE key='blob_backend_id'")
                .fetch_optional(&mut *tx)
                .await?
                .map(|row| row.try_get::<String, _>("value"))
                .transpose()?;
        if let Some(existing) = existing {
            anyhow::ensure!(
                existing == backend_id,
                "attachment backend changed for this database (expected {existing}, got {backend_id}); export or migrate attachments before changing blob configuration"
            );
        } else {
            sqlx::query("INSERT INTO storage_metadata(key,value) VALUES('blob_backend_id',?)")
                .bind(backend_id)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Returns the backend fingerprint previously bound to this database.
    pub async fn blob_backend_id(&self) -> Result<Option<String>> {
        let row = sqlx::query("SELECT value FROM storage_metadata WHERE key='blob_backend_id'")
            .fetch_optional(&self.pool)
            .await?;
        row.map(|row| row.try_get("value"))
            .transpose()
            .map_err(Into::into)
    }

    /// Checks durable forget rules before an importer or hook performs blob I/O.
    pub async fn event_is_tombstoned(&self, event: &CanonicalEvent) -> Result<bool> {
        let mut tx = self.pool.begin().await?;
        let exact_event = sqlx::query(
            "SELECT EXISTS(SELECT 1 FROM forgotten_events WHERE event_id=?) AS forgotten",
        )
        .bind(&event.id)
        .fetch_one(&mut *tx)
        .await?
        .try_get::<i64, _>("forgotten")?
            != 0;
        let exact_raw = !forgotten_raw_ids_for(&mut tx, &event.raw_record_ids)
            .await?
            .is_empty();
        let matches_rule = event_matches_forget_rule(&mut tx, event).await?;
        tx.commit().await?;
        Ok(exact_event || exact_raw || matches_rule)
    }

    /// Checks exact tombstones and durable selectors for a native record before
    /// a hook adapter or attachment decoder touches the payload.
    pub async fn raw_is_tombstoned(&self, raw: &RawRecord, context: &RawContext) -> Result<bool> {
        let mut tx = self.pool.begin().await?;
        let exact = !forgotten_raw_ids_for(&mut tx, std::slice::from_ref(&raw.id))
            .await?
            .is_empty();
        let matches_rule = raw_matches_forget_rule(&mut tx, raw, context).await?;
        tx.commit().await?;
        Ok(exact || matches_rule)
    }

    /// Registers a write-ahead lease before an attachment object is created.
    pub async fn register_blob_candidate(&self, backend_id: &str, object_key: &str) -> Result<()> {
        let now = crate::model::now_ms();
        let not_before = now.saturating_add(3_600_000);
        let result = sqlx::query(
            "INSERT INTO blob_gc_queue(backend_id,object_key,reason,enqueued_at_ms,not_before_ms,last_error,attempts,claimed_until_ms,claim_token) VALUES(?,?,'staged',?,?,NULL,0,NULL,NULL) ON CONFLICT(backend_id,object_key) DO UPDATE SET reason='staged',not_before_ms=MAX(blob_gc_queue.not_before_ms,excluded.not_before_ms),last_error=NULL WHERE blob_gc_queue.claim_token IS NULL",
        )
        .bind(backend_id)
        .bind(object_key)
        .bind(now)
        .bind(not_before)
        .execute(&self.pool)
        .await?;
        anyhow::ensure!(
            result.rows_affected() == 1,
            "attachment deletion is already in progress for {object_key}; retry after garbage collection completes"
        );
        Ok(())
    }

    async fn backfill_governance_indexes(&self) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        let complete = sqlx::query(
            "SELECT EXISTS(SELECT 1 FROM storage_metadata WHERE key='governance_backfill_v2') AS complete",
        )
        .fetch_one(&mut *tx)
        .await?
        .try_get::<i64, _>("complete")?
            != 0;
        if complete {
            tx.commit().await?;
            return Ok(());
        }
        let event_rows = sqlx::query("SELECT event_json FROM events ORDER BY id")
            .fetch_all(&mut *tx)
            .await?;
        for row in event_rows {
            let event = event_from_json(row.try_get("event_json")?)?;
            for attachment in event.attachment_refs() {
                sqlx::query(
                    "INSERT OR IGNORE INTO event_attachments(event_id,object_key,digest) VALUES(?,?,?)",
                )
                .bind(&event.id)
                .bind(&attachment.object_key)
                .bind(&attachment.digest)
                .execute(&mut *tx)
                .await?;
            }
            for raw_record_id in &event.raw_record_ids {
                sqlx::query(
                    "INSERT INTO raw_contexts(raw_record_id,session_id,repository,occurred_at_ms) SELECT ?,?,?,? WHERE EXISTS(SELECT 1 FROM raw_records WHERE id=?) ON CONFLICT(raw_record_id) DO UPDATE SET session_id=excluded.session_id,repository=COALESCE(excluded.repository,raw_contexts.repository),occurred_at_ms=COALESCE(excluded.occurred_at_ms,raw_contexts.occurred_at_ms)",
                )
                .bind(raw_record_id)
                .bind(&event.source.session_id)
                .bind(&event.source.repository)
                .bind(event.occurred_at_ms)
                .bind(raw_record_id)
                .execute(&mut *tx)
                .await?;
            }
        }
        let raw_rows =
            sqlx::query("SELECT id,agent,payload_json,ingested_at_ms FROM raw_records ORDER BY id")
                .fetch_all(&mut *tx)
                .await?;
        for row in raw_rows {
            let raw_record_id: String = row.try_get("id")?;
            let agent = row
                .try_get::<String, _>("agent")?
                .parse::<crate::model::AgentKind>()?;
            let payload = serde_json::from_str(row.try_get::<String, _>("payload_json")?.as_str())?;
            let ingested_at_ms: i64 = row.try_get("ingested_at_ms")?;
            let context = native_raw_context(agent, &payload);
            sqlx::query(
                "INSERT INTO raw_contexts(raw_record_id,session_id,repository,occurred_at_ms) VALUES(?,?,?,?) ON CONFLICT(raw_record_id) DO UPDATE SET session_id=COALESCE(raw_contexts.session_id,excluded.session_id),repository=COALESCE(raw_contexts.repository,excluded.repository),occurred_at_ms=CASE WHEN raw_contexts.occurred_at_ms IS NULL OR raw_contexts.occurred_at_ms=? THEN excluded.occurred_at_ms ELSE raw_contexts.occurred_at_ms END",
            )
            .bind(&raw_record_id)
            .bind(&context.session_id)
            .bind(&context.repository)
            .bind(context.occurred_at_ms.or(Some(ingested_at_ms)))
            .bind(ingested_at_ms)
            .execute(&mut *tx)
            .await?;
        }
        sqlx::query(
            "INSERT INTO storage_metadata(key,value) VALUES('governance_backfill_v2','complete')",
        )
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }
}

fn open_private_lock(path: &Path) -> Result<File> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(path)
        .with_context(|| format!("open lifecycle lock {}", path.display()))
}

fn lock_is_contended(error: &io::Error) -> bool {
    let expected = fs2::lock_contended_error();
    error.kind() == io::ErrorKind::WouldBlock
        || (error.raw_os_error().is_some() && error.raw_os_error() == expected.raw_os_error())
}

fn lock_exclusive_wait(file: &File) -> io::Result<()> {
    loop {
        match file.lock_exclusive() {
            Ok(()) => return Ok(()),
            Err(error) if lock_is_contended(&error) => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(error) => return Err(error),
        }
    }
}

fn database_location_identity(canonical_database: &Path) -> Result<String> {
    let mut sidecar_name = canonical_database.as_os_str().to_os_string();
    sidecar_name.push(".location-id");
    let sidecar = PathBuf::from(sidecar_name);
    loop {
        match std::fs::read_to_string(&sidecar) {
            Ok(identity) => {
                let identity = identity.trim();
                anyhow::ensure!(
                    identity.len() == 64 && identity.bytes().all(|byte| byte.is_ascii_hexdigit()),
                    "database location identity {} is malformed",
                    sidecar.display()
                );
                return Ok(identity.to_ascii_lowercase());
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("read database location identity {}", sidecar.display())
                });
            }
        }

        let entropy = format!(
            "{}\0{}\0{}",
            canonical_database.display(),
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let identity = selector_hash(&entropy);
        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&sidecar) {
            Ok(mut file) => {
                file.write_all(identity.as_bytes())?;
                file.write_all(b"\n")?;
                file.sync_all()?;
                enforce_private_database(&sidecar)?;
                sync_database_parent(
                    sidecar
                        .parent()
                        .context("database location identity has no parent")?,
                )?;
                return Ok(identity);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("create database location identity {}", sidecar.display())
                });
            }
        }
    }
}

#[cfg(unix)]
fn sync_database_parent(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_database_parent(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn prepare_private_database(path: &Path) -> Result<bool> {
    use std::os::unix::fs::OpenOptionsExt;

    if path.try_exists()? {
        return Ok(false);
    }
    match std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)
    {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => {
            Err(error).with_context(|| format!("create private database {}", path.display()))
        }
    }
}

#[cfg(not(unix))]
fn prepare_private_database(_path: &Path) -> Result<bool> {
    Ok(false)
}

#[cfg(unix)]
fn enforce_private_database(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("secure database permissions for {}", path.display()))
}

#[cfg(not(unix))]
fn enforce_private_database(_path: &Path) -> Result<()> {
    Ok(())
}

fn event_from_json(value: String) -> Result<CanonicalEvent> {
    serde_json::from_str(&value).context("decode formal canonical event")
}

fn selector_hash(value: &str) -> String {
    hex::encode(Sha256::digest(value.as_bytes()))
}

async fn event_matches_forget_rule(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
    event: &CanonicalEvent,
) -> Result<bool> {
    let session_hash = selector_hash(&event.source.session_id);
    let repository_hash = event.source.repository.as_deref().map(selector_hash);
    let row = sqlx::query(
        "SELECT EXISTS(SELECT 1 FROM forget_rules WHERE (session_hash IS NULL OR session_hash=?) AND (repository_hash IS NULL OR repository_hash=?) AND (agent IS NULL OR agent=?) AND (before_ms IS NULL OR ? < before_ms)) AS forgotten",
    )
    .bind(session_hash)
    .bind(repository_hash)
    .bind(event.source.agent.as_str())
    .bind(event.ordering_timestamp_ms())
    .fetch_one(&mut **tx)
    .await?;
    Ok(row.try_get::<i64, _>("forgotten")? != 0)
}

async fn raw_matches_forget_rule(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
    raw: &RawRecord,
    context: &RawContext,
) -> Result<bool> {
    let session_hash = context.session_id.as_deref().map(selector_hash);
    let repository_hash = context.repository.as_deref().map(selector_hash);
    let row = sqlx::query(
        "SELECT EXISTS(SELECT 1 FROM forget_rules WHERE (session_hash IS NULL OR session_hash=?) AND (repository_hash IS NULL OR repository_hash=?) AND (agent IS NULL OR agent=?) AND (before_ms IS NULL OR ? < before_ms)) AS forgotten",
    )
    .bind(session_hash)
    .bind(repository_hash)
    .bind(raw.agent.as_str())
    .bind(context.occurred_at_ms.unwrap_or(raw.ingested_at_ms))
    .fetch_one(&mut **tx)
    .await?;
    Ok(row.try_get::<i64, _>("forgotten")? != 0)
}

async fn forgotten_raw_ids_for(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
    ids: &[String],
) -> Result<HashSet<String>> {
    if ids.is_empty() {
        return Ok(HashSet::new());
    }
    let mut found = HashSet::new();
    for ids in ids.chunks(500) {
        let mut query = QueryBuilder::<Sqlite>::new(
            "SELECT raw_record_id FROM forgotten_raw_records WHERE raw_record_id IN (",
        );
        let mut values = query.separated(",");
        for id in ids {
            values.push_bind(id);
        }
        values.push_unseparated(")");
        for row in query.build().fetch_all(&mut **tx).await? {
            found.insert(row.try_get("raw_record_id")?);
        }
    }
    Ok(found)
}

async fn forgotten_event_ids_for(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
    ids: &[String],
) -> Result<HashSet<String>> {
    if ids.is_empty() {
        return Ok(HashSet::new());
    }
    let mut found = HashSet::new();
    for ids in ids.chunks(500) {
        let mut query = QueryBuilder::<Sqlite>::new(
            "SELECT event_id FROM forgotten_events WHERE event_id IN (",
        );
        let mut values = query.separated(",");
        for id in ids {
            values.push_bind(id);
        }
        values.push_unseparated(")");
        for row in query.build().fetch_all(&mut **tx).await? {
            found.insert(row.try_get("event_id")?);
        }
    }
    Ok(found)
}

async fn existing_raw_ids_for(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
    ids: &[String],
) -> Result<HashSet<String>> {
    if ids.is_empty() {
        return Ok(HashSet::new());
    }
    let mut found = HashSet::new();
    for ids in ids.chunks(500) {
        let mut query = QueryBuilder::<Sqlite>::new("SELECT id FROM raw_records WHERE id IN (");
        let mut values = query.separated(",");
        for id in ids {
            values.push_bind(id);
        }
        values.push_unseparated(")");
        for row in query.build().fetch_all(&mut **tx).await? {
            found.insert(row.try_get("id")?);
        }
    }
    Ok(found)
}

impl SqliteStore {
    async fn ingest_formal(
        &self,
        raw_records: &[RawRecord],
        raw_contexts: &[RawContext],
        events: &[CanonicalEvent],
        diagnostics: &[ConversionDiagnostic],
    ) -> Result<IngestStats> {
        let mut tx = self.pool.begin().await?;
        let raw_ids = raw_records
            .iter()
            .map(|raw| raw.id.clone())
            .collect::<Vec<_>>();
        let event_ids = events
            .iter()
            .map(|event| event.id.clone())
            .collect::<Vec<_>>();
        let mut rejected_raw_ids = forgotten_raw_ids_for(&mut tx, &raw_ids).await?;
        let mut rejected_event_ids = forgotten_event_ids_for(&mut tx, &event_ids).await?;
        let supplied_contexts = raw_contexts
            .iter()
            .map(|context| (context.raw_record_id.as_str(), context))
            .collect::<HashMap<_, _>>();
        anyhow::ensure!(
            supplied_contexts.len() == raw_contexts.len(),
            "ingest batch contains duplicate raw contexts"
        );
        let mut stats = IngestStats::default();
        let mut persisted_contexts = HashMap::new();
        for raw in raw_records {
            let linked_contexts = events
                .iter()
                .filter(|event| event.raw_record_ids.iter().any(|id| id == &raw.id))
                .map(|event| RawContext {
                    raw_record_id: raw.id.clone(),
                    session_id: Some(event.source.session_id.clone()),
                    repository: event.source.repository.clone(),
                    occurred_at_ms: event.occurred_at_ms,
                })
                .collect::<Vec<_>>();
            let supplied = supplied_contexts.get(raw.id.as_str()).copied();
            let native = native_raw_context(raw.agent, &raw.payload);
            let native_context = RawContext {
                raw_record_id: raw.id.clone(),
                session_id: native.session_id,
                repository: native.repository,
                occurred_at_ms: native.occurred_at_ms,
            };
            // Native ownership is recoverable from the immutable payload;
            // persist the source-supplied view when available so its extra
            // repository/time fields are not discarded. Event projections
            // remain durable in the normalized event rows.
            let context = supplied
                .cloned()
                .or_else(|| linked_contexts.first().cloned())
                .unwrap_or_else(|| native_context.clone());
            persisted_contexts.insert(raw.id.clone(), context);
            // A raw payload can legitimately carry several ownership views:
            // native metadata, source-supplied context, and one or more
            // projections. A forget match in any view rejects the plaintext;
            // no higher-priority view may mask a tombstoned one. Each view is
            // tested coherently so selectors are never assembled across
            // unrelated session/repository sources.
            for candidate in linked_contexts
                .iter()
                .chain(supplied)
                .chain(std::iter::once(&native_context))
            {
                if !rejected_raw_ids.contains(&raw.id)
                    && raw_matches_forget_rule(&mut tx, raw, candidate).await?
                {
                    rejected_raw_ids.insert(raw.id.clone());
                }
            }
        }
        for event in events {
            if !rejected_event_ids.contains(&event.id)
                && event_matches_forget_rule(&mut tx, event).await?
            {
                rejected_event_ids.insert(event.id.clone());
            }
        }
        // Close direct matches across the entire incoming provenance graph
        // before writing any plaintext. This makes rejection independent of
        // raw/event order and prevents a sibling raw from surviving a shared
        // forgotten projection.
        loop {
            let raw_count = rejected_raw_ids.len();
            let event_count = rejected_event_ids.len();
            for event in events {
                if rejected_event_ids.contains(&event.id)
                    || event
                        .raw_record_ids
                        .iter()
                        .any(|raw_id| rejected_raw_ids.contains(raw_id))
                {
                    rejected_event_ids.insert(event.id.clone());
                    rejected_raw_ids.extend(event.raw_record_ids.iter().cloned());
                }
            }
            if raw_count == rejected_raw_ids.len() && event_count == rejected_event_ids.len() {
                break;
            }
        }

        let mut legacy_cleanup_happened = false;
        if !rejected_raw_ids.is_empty() || !rejected_event_ids.is_empty() {
            // Older stores may already contain records without enough context
            // for the original forget operation. Close the same selection over
            // durable provenance so no projection, attachment, or derived
            // state can outlive the newly recognized plaintext.
            let existing_events = load_events_from(&mut tx).await?;
            let links = load_raw_links_from(&mut tx).await?;
            let storage_id = storage_id_from(&mut tx).await?;
            let mut plan = build_forget_plan_from_selection(
                existing_events,
                links,
                rejected_event_ids.clone(),
                rejected_raw_ids.clone(),
                &storage_id,
            );
            let applied = apply_forget_plan(&mut tx, &mut plan, crate::model::now_ms()).await?;
            legacy_cleanup_happened = applied.deleted_events > 0 || applied.deleted_raw > 0;
            stats.attachment_objects_queued = plan.orphan_attachment_keys.len();
            stats.unmanaged_attachment_objects = plan.unmanaged_attachment_keys.len();
            rejected_raw_ids.extend(plan.orphan_raw_record_ids);
            rejected_event_ids.extend(plan.selected_event_ids);
        }

        for raw in raw_records {
            if rejected_raw_ids.contains(&raw.id) {
                stats.raw_tombstoned += 1;
                continue;
            }
            let context = persisted_contexts
                .get(&raw.id)
                .expect("every raw record has a persistence context");
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
            sqlx::query(
                "INSERT INTO raw_contexts(raw_record_id,session_id,repository,occurred_at_ms) VALUES(?,?,?,?) ON CONFLICT(raw_record_id) DO UPDATE SET session_id=COALESCE(excluded.session_id,raw_contexts.session_id),repository=COALESCE(excluded.repository,raw_contexts.repository),occurred_at_ms=COALESCE(excluded.occurred_at_ms,raw_contexts.occurred_at_ms)",
            )
            .bind(&context.raw_record_id)
            .bind(&context.session_id)
            .bind(&context.repository)
            .bind(context.occurred_at_ms.or(Some(raw.ingested_at_ms)))
            .execute(&mut *tx)
            .await?;
        }
        let referenced_raw_ids = events
            .iter()
            .flat_map(|event| event.raw_record_ids.iter().cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let existing_raw_ids = existing_raw_ids_for(&mut tx, &referenced_raw_ids).await?;
        let enforce_provenance = self.lifecycle_lock.is_some() || !raw_records.is_empty();
        for event in events {
            if rejected_event_ids.contains(&event.id) {
                stats.tombstoned += 1;
                continue;
            }
            event.validate()?;
            if enforce_provenance {
                anyhow::ensure!(
                    !event.raw_record_ids.is_empty(),
                    "persistent canonical event {} has no raw provenance; use ingest_batch with its native records",
                    event.id
                );
                let missing = event
                    .raw_record_ids
                    .iter()
                    .filter(|id| !existing_raw_ids.contains(*id))
                    .cloned()
                    .collect::<Vec<_>>();
                anyhow::ensure!(
                    missing.is_empty(),
                    "canonical event {} references missing raw provenance: {}",
                    event.id,
                    missing.join(", ")
                );
            }
            for attachment in event.attachment_refs() {
                let active_claim = sqlx::query(
                    "SELECT EXISTS(SELECT 1 FROM blob_gc_queue WHERE object_key=? AND claim_token IS NOT NULL) AS active",
                )
                .bind(&attachment.object_key)
                .fetch_one(&mut *tx)
                .await?
                .try_get::<i64, _>("active")?
                    != 0;
                anyhow::ensure!(
                    !active_claim,
                    "attachment deletion is in progress for {}; retry ingestion after garbage collection completes",
                    attachment.object_key
                );
            }
            let event_json = serde_json::to_string(event)?;
            let search_text = event.search_text();
            let ordering_timestamp_ms = event.ordering_timestamp_ms();
            let result = sqlx::query(
                "INSERT OR IGNORE INTO events(id,fingerprint,spec_version,agent,session_id,repository,event_type,occurred_at_ms,ordering_timestamp_ms,ingest_seq,search_text,event_json) VALUES(?,?,?,?,?,?,?,?,?,(SELECT COALESCE(MAX(ingest_seq),0)+1 FROM events),?,?)",
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
            // Canonical identity intentionally ignores raw provenance, so a
            // later projection can deduplicate onto the same event. Keep the
            // serialized envelope synchronized with the normalized many-to-many
            // link table or an export/re-import would lose the later raw link.
            let linked_raw_ids = sqlx::query(
                "SELECT raw_record_id FROM event_raw_records WHERE event_id=? ORDER BY raw_record_id",
            )
            .bind(&event.id)
            .fetch_all(&mut *tx)
            .await?
            .into_iter()
            .map(|row| row.try_get::<String, _>("raw_record_id"))
            .collect::<std::result::Result<Vec<_>, _>>()?;
            let mut persisted_event = sqlx::query("SELECT event_json FROM events WHERE id=?")
                .bind(&event.id)
                .fetch_one(&mut *tx)
                .await
                .and_then(|row| row.try_get::<String, _>("event_json"))
                .map_err(anyhow::Error::from)
                .and_then(event_from_json)?;
            persisted_event.raw_record_ids = linked_raw_ids;
            sqlx::query("UPDATE events SET event_json=? WHERE id=?")
                .bind(serde_json::to_string(&persisted_event)?)
                .bind(&event.id)
                .execute(&mut *tx)
                .await?;
            if result.rows_affected() == 1
                && let Some(inserted) = stats.inserted_events.last_mut()
            {
                inserted.raw_record_ids = persisted_event.raw_record_ids.clone();
            }
            for attachment in event.attachment_refs() {
                sqlx::query(
                    "INSERT OR IGNORE INTO event_attachments(event_id,object_key,digest) VALUES(?,?,?)",
                )
                .bind(&event.id)
                .bind(&attachment.object_key)
                .bind(&attachment.digest)
                .execute(&mut *tx)
                .await?;
                sqlx::query("DELETE FROM blob_gc_queue WHERE object_key=?")
                    .bind(&attachment.object_key)
                    .execute(&mut *tx)
                    .await?;
            }
        }
        for diagnostic in diagnostics {
            sqlx::query(
                "INSERT OR IGNORE INTO conversion_diagnostics(id,raw_record_id,severity,code,message,created_at_ms) SELECT ?,?,?,?,?,? WHERE EXISTS(SELECT 1 FROM raw_records WHERE id=?)",
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
            .bind(&diagnostic.raw_record_id)
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
        if legacy_cleanup_happened {
            stats.sqlite_cleanup_error = self
                .scrub_deleted_pages()
                .await
                .err()
                .map(|error| format!("{error:#}"));
            stats.sqlite_cleanup_pending = self.sqlite_cleanup_is_pending().await?;
            if stats.sqlite_cleanup_pending && stats.sqlite_cleanup_error.is_none() {
                stats.sqlite_cleanup_error = Some(
                    "another deletion committed while SQLite cleanup was running; retry cleanup"
                        .into(),
                );
            }
        }
        Ok(stats)
    }
}

#[async_trait]
impl EventStore for SqliteStore {
    async fn ingest(&self, events: &[CanonicalEvent]) -> Result<IngestStats> {
        self.ingest_formal(&[], &[], events, &[]).await
    }

    async fn ingest_batch(&self, batch: super::IngestBatch<'_>) -> Result<IngestStats> {
        self.ingest_formal(
            batch.raw_records,
            batch.raw_contexts,
            batch.events,
            batch.diagnostics,
        )
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
            "SELECT e.event_json,bm25(events_fts) AS rank FROM events_fts JOIN events e ON e.rowid=events_fts.rowid WHERE events_fts MATCH ? AND (? IS NULL OR e.repository=?) AND (? IS NULL OR e.agent=?) AND (? IS NULL OR e.ordering_timestamp_ms>=?) AND (? IS NULL OR e.ordering_timestamp_ms<=?) ORDER BY rank,e.ordering_timestamp_ms DESC,e.id LIMIT ?",
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

    async fn scoped_message_events(&self, query: &SearchQuery) -> Result<Vec<CanonicalEvent>> {
        let agent = query.agent.map(|value| value.as_str());
        let rows = sqlx::query(
            "SELECT event_json FROM events WHERE event_type IN ('message','tool_call','tool_result') AND length(trim(search_text))>0 AND (? IS NULL OR repository=?) AND (? IS NULL OR agent=?) AND (? IS NULL OR ordering_timestamp_ms>=?) AND (? IS NULL OR ordering_timestamp_ms<=?) ORDER BY ingest_seq,id",
        )
        .bind(query.repository.as_deref())
        .bind(query.repository.as_deref())
        .bind(agent)
        .bind(agent)
        .bind(query.since_ms)
        .bind(query.since_ms)
        .bind(query.until_ms)
        .bind(query.until_ms)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| event_from_json(row.try_get("event_json")?))
            .collect()
    }

    async fn recent(&self, query: &SearchQuery) -> Result<Vec<SearchHit>> {
        let agent = query.agent.map(|a| a.as_str());
        let limit = query.limit as i64;
        let rows = sqlx::query(
            "SELECT event_json FROM events WHERE (? IS NULL OR repository=?) AND (? IS NULL OR agent=?) AND (? IS NULL OR ordering_timestamp_ms>=?) AND (? IS NULL OR ordering_timestamp_ms<=?) ORDER BY ordering_timestamp_ms DESC,id LIMIT ?",
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
        let rows =
            sqlx::query("SELECT event_json FROM events WHERE session_id=? ORDER BY ingest_seq,id")
                .bind(session_id)
                .fetch_all(&self.pool)
                .await?;
        rows.into_iter()
            .map(|row| event_from_json(row.try_get("event_json")?))
            .collect()
    }

    async fn ordered_session_events(&self, session_id: &str) -> Result<Vec<OrderedEvent>> {
        let rows = sqlx::query(
            "SELECT ingest_seq,event_json FROM events WHERE session_id=? ORDER BY ingest_seq,id",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                let ordinal = row.try_get::<i64, _>("ingest_seq")?;
                anyhow::ensure!(ordinal > 0, "event has an invalid ingestion ordinal");
                Ok(OrderedEvent {
                    ordinal: ordinal as u64,
                    event: event_from_json(row.try_get("event_json")?)?,
                })
            })
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

fn encode_vector(values: &[f32]) -> Result<Vec<u8>> {
    anyhow::ensure!(!values.is_empty(), "strategy vector cannot be empty");
    anyhow::ensure!(
        values.iter().all(|value| value.is_finite()),
        "strategy vector must contain only finite values"
    );
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

async fn ensure_vector_dimensions(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
    namespace: &str,
    model: &str,
    dimensions: usize,
) -> Result<()> {
    anyhow::ensure!(dimensions > 0, "strategy vector cannot be empty");
    let existing = sqlx::query(
        "SELECT dimensions FROM strategy_event_vectors WHERE namespace=? AND model=? LIMIT 1",
    )
    .bind(namespace)
    .bind(model)
    .fetch_optional(&mut **tx)
    .await?
    .map(|row| row.try_get::<i64, _>("dimensions"))
    .transpose()?;
    if let Some(existing) = existing {
        anyhow::ensure!(
            existing == dimensions as i64,
            "vector dimensions changed for namespace {namespace} model {model}: existing {existing}, new {dimensions}"
        );
    }
    Ok(())
}

fn decode_vector(bytes: Vec<u8>, dimensions: usize) -> Result<Vec<f32>> {
    let expected = dimensions
        .checked_mul(4)
        .context("strategy vector byte length overflow")?;
    anyhow::ensure!(
        bytes.len() == expected,
        "strategy vector has {} bytes, expected {expected}",
        bytes.len()
    );
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        values.iter().all(|value| value.is_finite()),
        "stored strategy vector contains a non-finite value"
    );
    Ok(values)
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

#[async_trait]
impl VectorRepository for SqliteStore {
    async fn query_vectors(&self, query: &VectorQuery) -> Result<Vec<VectorHit>> {
        if query.scope.limit == 0 {
            return Ok(Vec::new());
        }
        anyhow::ensure!(!query.values.is_empty(), "query vector cannot be empty");
        anyhow::ensure!(
            query.values.iter().all(|value| value.is_finite()),
            "query vector must contain only finite values"
        );
        let agent = query.scope.agent.map(|value| value.as_str());
        let rows = sqlx::query(
            "SELECT v.event_id,v.dimensions,v.vector,e.event_json FROM strategy_event_vectors v JOIN events e ON e.id=v.event_id WHERE v.namespace=? AND v.model=? AND e.event_type IN ('message','tool_call','tool_result') AND length(trim(e.search_text))>0 AND (? IS NULL OR e.repository=?) AND (? IS NULL OR e.agent=?) AND (? IS NULL OR e.ordering_timestamp_ms>=?) AND (? IS NULL OR e.ordering_timestamp_ms<=?)",
        )
        .bind(&query.namespace)
        .bind(&query.model)
        .bind(query.scope.repository.as_deref())
        .bind(query.scope.repository.as_deref())
        .bind(agent)
        .bind(agent)
        .bind(query.scope.since_ms)
        .bind(query.scope.since_ms)
        .bind(query.scope.until_ms)
        .bind(query.scope.until_ms)
        .fetch_all(&self.pool)
        .await?;

        let query_norm = query
            .values
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        anyhow::ensure!(query_norm > 0.0, "query vector norm must be positive");
        let mut hits = Vec::with_capacity(rows.len());
        for row in rows {
            let event_id: String = row.try_get("event_id")?;
            let dimensions: i64 = row.try_get("dimensions")?;
            anyhow::ensure!(
                dimensions == query.values.len() as i64,
                "stored vector {event_id} has {dimensions} dimensions, expected {}",
                query.values.len()
            );
            let stored = decode_vector(row.try_get("vector")?, query.values.len())
                .with_context(|| format!("decode stored vector for event {event_id}"))?;
            let mut dot = 0.0;
            let mut stored_norm_squared = 0.0;
            for (left, right) in query.values.iter().zip(stored) {
                let left = f64::from(*left);
                let right = f64::from(right);
                dot += left * right;
                stored_norm_squared += right * right;
            }
            if stored_norm_squared == 0.0 {
                continue;
            }
            hits.push(VectorHit {
                event: event_from_json(row.try_get("event_json")?)?,
                score: dot / (query_norm * stored_norm_squared.sqrt()),
            });
        }
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| {
                    right
                        .event
                        .ordering_timestamp_ms()
                        .cmp(&left.event.ordering_timestamp_ms())
                })
                .then_with(|| left.event.id.cmp(&right.event.id))
        });
        hits.truncate(query.scope.limit.min(10_000));
        Ok(hits)
    }

    async fn indexed_event_ids(
        &self,
        namespace: &str,
        model: &str,
        scope: &SearchQuery,
    ) -> Result<Vec<String>> {
        let agent = scope.agent.map(|value| value.as_str());
        let rows = sqlx::query(
            "SELECT v.event_id FROM strategy_event_vectors v JOIN events e ON e.id=v.event_id WHERE v.namespace=? AND v.model=? AND e.event_type IN ('message','tool_call','tool_result') AND length(trim(e.search_text))>0 AND (? IS NULL OR e.repository=?) AND (? IS NULL OR e.agent=?) AND (? IS NULL OR e.ordering_timestamp_ms>=?) AND (? IS NULL OR e.ordering_timestamp_ms<=?) ORDER BY e.ingest_seq,e.id",
        )
        .bind(namespace)
        .bind(model)
        .bind(scope.repository.as_deref())
        .bind(scope.repository.as_deref())
        .bind(agent)
        .bind(agent)
        .bind(scope.since_ms)
        .bind(scope.since_ms)
        .bind(scope.until_ms)
        .bind(scope.until_ms)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| row.try_get("event_id").map_err(Into::into))
            .collect()
    }
}

#[async_trait]
impl StrategyStateRepository for SqliteStore {
    async fn strategy_progress(
        &self,
        strategy: &str,
    ) -> Result<(DomainEventSequence, DomainEventSequence)> {
        let mut tx = self.pool.begin().await?;
        let checkpoint = sqlx::query("SELECT last_seq FROM strategy_checkpoints WHERE strategy=?")
            .bind(strategy)
            .fetch_optional(&mut *tx)
            .await?
            .map(|row| row.try_get::<i64, _>("last_seq"))
            .transpose()?
            .map(DomainEventSequence::from_database)
            .unwrap_or(DomainEventSequence::ORIGIN);
        let high_water = sqlx::query("SELECT COALESCE(MAX(seq),0) AS seq FROM domain_events")
            .fetch_one(&mut *tx)
            .await?
            .try_get::<i64, _>("seq")?;
        tx.commit().await?;
        Ok((checkpoint, DomainEventSequence::from_database(high_water)))
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

    /// Returns every event in stable ingestion order.
    pub async fn all_events(&self) -> Result<Vec<CanonicalEvent>> {
        let rows = sqlx::query("SELECT event_json FROM events ORDER BY ingest_seq,id")
            .fetch_all(&self.pool)
            .await?;
        rows.into_iter()
            .map(|row| event_from_json(row.try_get("event_json")?))
            .collect()
    }

    /// Reads all events and the matching domain-event high-water mark atomically.
    pub async fn events_and_latest_seq(
        &self,
    ) -> Result<(Vec<CanonicalEvent>, DomainEventSequence)> {
        let mut tx = self.pool.begin().await?;
        let events = load_events_from(&mut tx).await?;
        let row = sqlx::query("SELECT COALESCE(MAX(seq),0) AS seq FROM domain_events")
            .fetch_one(&mut *tx)
            .await?;
        let sequence = DomainEventSequence::from_database(row.try_get("seq")?);
        tx.commit().await?;
        Ok((events, sequence))
    }

    /// Exports every raw record, canonical event, and diagnostic in deterministic order.
    pub async fn canonical_archive(&self) -> Result<CanonicalArchive> {
        let mut tx = self.pool.begin().await?;
        let raw_rows = sqlx::query(
            "SELECT id,digest,agent,source,record_index,native_type,payload_json,ingested_at_ms FROM raw_records ORDER BY agent,source,record_index,id",
        )
        .fetch_all(&mut *tx)
        .await?;
        let event_rows =
            sqlx::query("SELECT id,event_json FROM events ORDER BY ordering_timestamp_ms,id")
                .fetch_all(&mut *tx)
                .await?;
        let mut raw_links_by_event = HashMap::<String, Vec<String>>::new();
        for (event_id, raw_record_id) in load_raw_links_from(&mut tx).await? {
            raw_links_by_event
                .entry(event_id)
                .or_default()
                .push(raw_record_id);
        }
        let diagnostic_rows = sqlx::query(
            "SELECT id,raw_record_id,severity,code,message,created_at_ms FROM conversion_diagnostics ORDER BY created_at_ms,id",
        )
        .fetch_all(&mut *tx)
        .await?;
        let archive = CanonicalArchive {
            spec_version: ArchiveSpecVersion::V1,
            raw_records: raw_rows
                .into_iter()
                .map(raw_record_from_row)
                .collect::<Result<_>>()?,
            events: event_rows
                .into_iter()
                .map(|row| {
                    let event_id: String = row.try_get("id")?;
                    let mut event = event_from_json(row.try_get("event_json")?)?;
                    if let Some(raw_record_ids) = raw_links_by_event.remove(&event_id) {
                        event.raw_record_ids = raw_record_ids;
                    }
                    Ok(event)
                })
                .collect::<Result<_>>()?,
            diagnostics: diagnostic_rows
                .into_iter()
                .map(diagnostic_from_row)
                .collect::<Result<_>>()?,
        };
        archive.validate()?;
        tx.commit().await?;
        Ok(archive)
    }

    /// Calculates the exact impact of a bounded forget operation without writing.
    pub async fn preview_forget(&self, filter: &ForgetFilter) -> Result<ForgetPreview> {
        filter.validate()?;
        let mut tx = self.pool.begin().await?;
        let events = load_events_from(&mut tx).await?;
        let links = load_raw_links_from(&mut tx).await?;
        let raw_records = load_raw_for_forget_from(&mut tx).await?;
        let storage_id = storage_id_from(&mut tx).await?;
        let preview = build_forget_plan(events, links, raw_records, filter, &storage_id)?.preview();
        tx.commit().await?;
        Ok(preview)
    }

    /// Atomically deletes matching events and their sensitive native records.
    ///
    /// Attachment objects are enqueued in the same transaction and must be
    /// drained separately. Derived memories and checkpoints are reset, then the
    /// remaining event set is emitted as a fresh replay batch.
    pub async fn forget(&self, filter: &ForgetFilter) -> Result<ForgetStats> {
        filter.validate()?;
        let _guard = self.lock_lifecycle()?;
        let mut tx = self.pool.begin().await?;
        let events = load_events_from(&mut tx).await?;
        let links = load_raw_links_from(&mut tx).await?;
        let raw_records = load_raw_for_forget_from(&mut tx).await?;
        let storage_id = storage_id_from(&mut tx).await?;
        let mut plan = build_forget_plan(events, links, raw_records, filter, &storage_id)?;
        let now = crate::model::now_ms();
        sqlx::query(
            "INSERT OR IGNORE INTO forget_rules(session_hash,repository_hash,agent,before_ms,created_at_ms) VALUES(?,?,?,?,?)",
        )
        .bind(filter.session_id.as_deref().map(selector_hash))
        .bind(filter.repository.as_deref().map(selector_hash))
        .bind(filter.agent.map(|agent| agent.as_str()))
        .bind(filter.before_ms)
        .bind(now)
        .execute(&mut *tx)
        .await?;
        let applied = apply_forget_plan(&mut tx, &mut plan, now).await?;
        tx.commit().await?;

        let cleanup_required = self.sqlite_cleanup_is_pending().await?;
        let mut sqlite_cleanup_error = if cleanup_required {
            self.scrub_deleted_pages()
                .await
                .err()
                .map(|error| format!("{error:#}"))
        } else {
            None
        };
        let sqlite_cleanup_pending = self.sqlite_cleanup_is_pending().await?;
        if sqlite_cleanup_pending && sqlite_cleanup_error.is_none() {
            sqlite_cleanup_error = Some(
                "another deletion committed while SQLite cleanup was running; retry cleanup".into(),
            );
        }

        Ok(ForgetStats {
            events: applied.deleted_events,
            raw_records: applied.deleted_raw,
            attachment_objects_queued: plan.orphan_attachment_keys.len(),
            unmanaged_attachment_objects: plan.unmanaged_attachment_keys.len(),
            remaining_events: plan.remaining_event_ids.len(),
            sqlite_cleanup_pending,
            sqlite_cleanup_error,
        })
    }

    /// Rebuilds FTS state and releases deleted SQLite/WAL pages where possible.
    ///
    /// Lifecycle locking avoids redundant work, while a generation-scoped
    /// marker keeps concurrent public ingestion from losing a later cleanup.
    pub async fn scrub_deleted_pages(&self) -> Result<()> {
        let cleanup_generation = self.sqlite_cleanup_generation().await?;
        let mut tx = self.pool.begin().await?;
        sqlx::query("INSERT INTO events_fts(events_fts) VALUES('rebuild')")
            .execute(&mut *tx)
            .await?;
        sqlx::query("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;

        for phase in ["before vacuum", "after vacuum"] {
            if phase == "after vacuum" {
                sqlx::query("VACUUM").execute(&self.pool).await?;
            }
            let row = sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
                .fetch_one(&self.pool)
                .await?;
            let busy: i64 = row.try_get(0)?;
            anyhow::ensure!(busy == 0, "WAL checkpoint remained busy {phase}");
        }
        if let Some(generation) = cleanup_generation {
            // A concurrent legacy cleanup may commit after this scrub starts.
            // Only clear the generation whose deletion this pass observed.
            self.clear_sqlite_cleanup_generation(&generation).await?;
        }
        Ok(())
    }

    async fn sqlite_cleanup_generation(&self) -> Result<Option<String>> {
        sqlx::query("SELECT value FROM storage_metadata WHERE key='sqlite_cleanup_pending'")
            .fetch_optional(&self.pool)
            .await?
            .map(|row| row.try_get::<String, _>("value"))
            .transpose()
            .map_err(Into::into)
    }

    async fn clear_sqlite_cleanup_generation(&self, generation: &str) -> Result<()> {
        sqlx::query("DELETE FROM storage_metadata WHERE key='sqlite_cleanup_pending' AND value=?")
            .bind(generation)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Returns whether a committed logical deletion still needs page cleanup.
    pub async fn sqlite_cleanup_is_pending(&self) -> Result<bool> {
        let row = sqlx::query(
            "SELECT EXISTS(SELECT 1 FROM storage_metadata WHERE key='sqlite_cleanup_pending') AS pending",
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(row.try_get::<i64, _>("pending")? != 0)
    }

    /// Claims pending attachment deletions in stable queue order.
    pub async fn claim_pending_blob_gc(&self, limit: usize) -> Result<Vec<BlobGcEntry>> {
        let now = crate::model::now_ms();
        let claimed_until = now.saturating_add(3_600_000);
        let rows = sqlx::query(
            "UPDATE blob_gc_queue SET claim_token=lower(hex(randomblob(16))),claimed_until_ms=? WHERE rowid IN (SELECT rowid FROM blob_gc_queue WHERE not_before_ms<=? AND (claim_token IS NULL OR claimed_until_ms<=?) ORDER BY not_before_ms,enqueued_at_ms,backend_id,object_key LIMIT ?) RETURNING backend_id,object_key,attempts,claim_token",
        )
        .bind(claimed_until)
        .bind(now)
        .bind(now)
        .bind(limit.min(10_000) as i64)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                Ok(BlobGcEntry {
                    backend_id: row.try_get("backend_id")?,
                    object_key: row.try_get("object_key")?,
                    attempts: row.try_get("attempts")?,
                    claim_token: row.try_get("claim_token")?,
                })
            })
            .collect()
    }

    /// Removes a successfully deleted attachment from the durable GC queue.
    pub async fn complete_blob_gc(
        &self,
        backend_id: &str,
        object_key: &str,
        claim_token: &str,
    ) -> Result<()> {
        let result = sqlx::query(
            "DELETE FROM blob_gc_queue WHERE backend_id=? AND object_key=? AND claim_token=?",
        )
        .bind(backend_id)
        .bind(object_key)
        .bind(claim_token)
        .execute(&self.pool)
        .await?;
        anyhow::ensure!(
            result.rows_affected() == 1,
            "attachment GC claim was lost for {object_key}"
        );
        Ok(())
    }

    /// Records a failed attachment deletion for later retry.
    pub async fn fail_blob_gc(
        &self,
        backend_id: &str,
        object_key: &str,
        claim_token: &str,
        error: &str,
    ) -> Result<()> {
        let result = sqlx::query("UPDATE blob_gc_queue SET attempts=attempts+1,last_error=?,claim_token=NULL,claimed_until_ms=NULL WHERE backend_id=? AND object_key=? AND claim_token=?")
            .bind(error)
            .bind(backend_id)
            .bind(object_key)
            .bind(claim_token)
            .execute(&self.pool)
            .await?;
        anyhow::ensure!(
            result.rows_affected() == 1,
            "attachment GC claim was lost for {object_key}"
        );
        Ok(())
    }

    /// Counts attachment objects still waiting for deletion.
    pub async fn pending_blob_gc_count(&self) -> Result<usize> {
        let row = sqlx::query("SELECT COUNT(*) AS count FROM blob_gc_queue")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.try_get::<i64, _>("count")? as usize)
    }

    /// Counts legacy or foreign attachment keys retained for manual cleanup.
    pub async fn unmanaged_blob_residue_count(&self) -> Result<usize> {
        let row = sqlx::query("SELECT COUNT(*) AS count FROM unmanaged_blob_residue")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.try_get::<i64, _>("count")? as usize)
    }

    /// Checks the normalized attachment index before an external delete.
    pub async fn attachment_is_referenced(&self, object_key: &str) -> Result<bool> {
        let row = sqlx::query(
            "SELECT EXISTS(SELECT 1 FROM event_attachments WHERE object_key=?) AS referenced",
        )
        .bind(object_key)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.try_get::<i64, _>("referenced")? != 0)
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

    /// Returns every event vector in one strategy namespace.
    pub async fn all_strategy_vectors(&self, namespace: &str) -> Result<Vec<StrategyEventVector>> {
        let rows = sqlx::query(
            "SELECT event_id,model,dimensions,vector FROM strategy_event_vectors WHERE namespace=? ORDER BY event_id,model",
        )
        .bind(namespace)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                let dimensions: i64 = row.try_get("dimensions")?;
                anyhow::ensure!(dimensions > 0, "stored strategy vector has no dimensions");
                Ok(StrategyEventVector {
                    namespace: namespace.to_owned(),
                    event_id: row.try_get("event_id")?,
                    model: row.try_get("model")?,
                    values: decode_vector(row.try_get("vector")?, dimensions as usize)?,
                })
            })
            .collect()
    }

    /// Atomically replaces one strategy's namespace and advances its checkpoint.
    ///
    /// Every record must belong to `strategy`. Validation or persistence failure
    /// leaves both the old records and checkpoint intact; success clears failure state.
    pub async fn replace_strategy_state(
        &self,
        strategy: &str,
        records: &[MemoryRecord],
        vectors: &[StrategyEventVector],
        checkpoint: DomainEventSequence,
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        let row = sqlx::query("SELECT COALESCE(MAX(seq),0) AS seq FROM domain_events")
            .fetch_one(&mut *tx)
            .await?;
        let current = row.try_get::<i64, _>("seq")?;
        anyhow::ensure!(
            current == checkpoint.get(),
            "strategy rebuild snapshot is stale (expected sequence {}, current {current})",
            checkpoint.get()
        );
        sqlx::query!("DELETE FROM memories WHERE namespace=?", strategy)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM strategy_event_vectors WHERE namespace=?")
            .bind(strategy)
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
        for vector in vectors {
            anyhow::ensure!(
                vector.namespace == strategy,
                "strategy {strategy} cannot write vector namespace {}",
                vector.namespace
            );
            anyhow::ensure!(
                !vector.model.trim().is_empty(),
                "vector model cannot be empty"
            );
            ensure_vector_dimensions(
                &mut tx,
                &vector.namespace,
                &vector.model,
                vector.values.len(),
            )
            .await?;
            let bytes = encode_vector(&vector.values)?;
            let now = crate::model::now_ms();
            sqlx::query(
                "INSERT INTO strategy_event_vectors(namespace,event_id,model,dimensions,vector,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?,?,?)",
            )
            .bind(&vector.namespace)
            .bind(&vector.event_id)
            .bind(&vector.model)
            .bind(vector.values.len() as i64)
            .bind(bytes)
            .bind(now)
            .bind(now)
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
        let seq_value = seq.get();
        let row = sqlx::query(
            "SELECT EXISTS(SELECT 1 FROM domain_events WHERE seq=? AND kind='events_ingested') AS present",
        )
        .bind(seq_value)
        .fetch_one(&mut *tx)
        .await?;
        anyhow::ensure!(
            row.try_get::<i64, _>("present")? != 0,
            "domain event sequence {seq_value} no longer exists"
        );
        let checkpoint_row =
            sqlx::query("SELECT last_seq FROM strategy_checkpoints WHERE strategy=?")
                .bind(strategy)
                .fetch_optional(&mut *tx)
                .await?;
        if let Some(checkpoint_row) = checkpoint_row {
            let previous: i64 = checkpoint_row.try_get("last_seq")?;
            anyhow::ensure!(
                seq_value > previous,
                "strategy {strategy} sequence {seq_value} does not advance checkpoint {previous}"
            );
        }
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
                MemoryMutation::UpsertVector(vector) => {
                    anyhow::ensure!(
                        vector.namespace == strategy,
                        "strategy {strategy} cannot write vector namespace {}",
                        vector.namespace
                    );
                    anyhow::ensure!(
                        !vector.model.trim().is_empty(),
                        "vector model cannot be empty"
                    );
                    ensure_vector_dimensions(
                        &mut tx,
                        &vector.namespace,
                        &vector.model,
                        vector.values.len(),
                    )
                    .await?;
                    let bytes = encode_vector(&vector.values)?;
                    let now = crate::model::now_ms();
                    sqlx::query(
                        "INSERT INTO strategy_event_vectors(namespace,event_id,model,dimensions,vector,created_at_ms,updated_at_ms) VALUES(?,?,?,?,?,?,?) ON CONFLICT(namespace,event_id,model) DO UPDATE SET dimensions=excluded.dimensions,vector=excluded.vector,updated_at_ms=excluded.updated_at_ms",
                    )
                    .bind(&vector.namespace)
                    .bind(&vector.event_id)
                    .bind(&vector.model)
                    .bind(vector.values.len() as i64)
                    .bind(bytes)
                    .bind(now)
                    .bind(now)
                    .execute(&mut *tx)
                    .await?;
                }
                MemoryMutation::DeleteVector {
                    namespace,
                    event_id,
                    model,
                } => {
                    anyhow::ensure!(
                        namespace == strategy,
                        "strategy {strategy} cannot delete vector namespace {namespace}"
                    );
                    sqlx::query(
                        "DELETE FROM strategy_event_vectors WHERE namespace=? AND event_id=? AND model=?",
                    )
                    .bind(namespace)
                    .bind(event_id)
                    .bind(model)
                    .execute(&mut *tx)
                    .await?;
                }
            }
        }
        let seq = seq_value;
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

fn diagnostic_from_row(row: sqlx::sqlite::SqliteRow) -> Result<ConversionDiagnostic> {
    let severity = match row.try_get::<String, _>("severity")?.as_str() {
        "warning" => DiagnosticSeverity::Warning,
        "error" => DiagnosticSeverity::Error,
        value => anyhow::bail!("unsupported diagnostic severity {value:?}"),
    };
    Ok(ConversionDiagnostic {
        id: row.try_get("id")?,
        raw_record_id: row.try_get("raw_record_id")?,
        severity,
        code: row.try_get("code")?,
        message: row.try_get("message")?,
        created_at_ms: row.try_get("created_at_ms")?,
    })
}

async fn load_events_from(tx: &mut sqlx::Transaction<'_, Sqlite>) -> Result<Vec<CanonicalEvent>> {
    let rows = sqlx::query("SELECT event_json FROM events ORDER BY ingest_seq,id")
        .fetch_all(&mut **tx)
        .await?;
    rows.into_iter()
        .map(|row| event_from_json(row.try_get("event_json")?))
        .collect()
}

async fn load_raw_links_from(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
) -> Result<Vec<(String, String)>> {
    let rows = sqlx::query(
        "SELECT event_id,raw_record_id FROM event_raw_records ORDER BY event_id,raw_record_id",
    )
    .fetch_all(&mut **tx)
    .await?;
    rows.into_iter()
        .map(|row| Ok((row.try_get("event_id")?, row.try_get("raw_record_id")?)))
        .collect()
}

async fn load_raw_for_forget_from(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
) -> Result<Vec<RawForForget>> {
    let rows = sqlx::query(
        "SELECT r.id,r.agent,r.source,r.record_index,r.payload_json,r.ingested_at_ms,c.session_id,c.repository,c.occurred_at_ms FROM raw_records r LEFT JOIN raw_contexts c ON c.raw_record_id=r.id ORDER BY r.agent,r.source,r.record_index,r.id",
    )
    .fetch_all(&mut **tx)
    .await?;

    struct LoadedRaw {
        id: String,
        agent: AgentKind,
        source: String,
        record_index: u64,
        native_context: RawContext,
        stored_context: RawContext,
        is_codex_session_meta: bool,
        ingested_at_ms: i64,
    }

    let mut loaded = Vec::with_capacity(rows.len());
    for row in rows {
        let agent = row.try_get::<String, _>("agent")?.parse()?;
        let payload = serde_json::from_str::<serde_json::Value>(
            row.try_get::<String, _>("payload_json")?.as_str(),
        )?;
        let raw_record_id: String = row.try_get("id")?;
        let payload_context = native_raw_context(agent, &payload);
        loaded.push(LoadedRaw {
            id: raw_record_id.clone(),
            agent,
            source: row.try_get("source")?,
            record_index: row.try_get::<i64, _>("record_index")? as u64,
            native_context: RawContext {
                raw_record_id: raw_record_id.clone(),
                session_id: payload_context.session_id,
                repository: payload_context.repository,
                occurred_at_ms: payload_context.occurred_at_ms,
            },
            stored_context: RawContext {
                raw_record_id,
                session_id: row.try_get("session_id")?,
                repository: row.try_get("repository")?,
                occurred_at_ms: row.try_get("occurred_at_ms")?,
            },
            is_codex_session_meta: agent == AgentKind::Codex
                && payload.get("type").and_then(serde_json::Value::as_str) == Some("session_meta"),
            ingested_at_ms: row.try_get("ingested_at_ms")?,
        });
    }

    let mut result = Vec::with_capacity(loaded.len());
    let mut start = 0;
    while start < loaded.len() {
        let mut end = start + 1;
        while end < loaded.len()
            && loaded[end].agent == loaded[start].agent
            && loaded[end].source == loaded[start].source
        {
            end += 1;
        }

        // A legacy Codex rollout may have been retained after whole-source
        // decoding failed. Reconstruct an additional coherent ownership view
        // from session_meta boundaries without overwriting stored/native views.
        // Conflicting markers at the same record index remain explicit
        // candidates. A bounded selector that distinguishes between them is
        // rejected below instead of silently retaining data or over-deleting
        // another session.
        let mut markers = HashMap::<u64, HashSet<(Option<String>, Option<String>)>>::new();
        if loaded[start].agent == AgentKind::Codex {
            for raw in &loaded[start..end] {
                if raw.is_codex_session_meta {
                    markers.entry(raw.record_index).or_default().insert((
                        raw.native_context.session_id.clone(),
                        raw.native_context.repository.clone(),
                    ));
                }
            }
        }
        let mut active_segments = Vec::new();
        for raw in &loaded[start..end] {
            if let Some(boundaries) = markers.get(&raw.record_index) {
                active_segments = boundaries.iter().cloned().collect();
                active_segments.sort();
            }
            let mut contexts = vec![raw.native_context.clone()];
            if !contexts.contains(&raw.stored_context) {
                contexts.push(raw.stored_context.clone());
            }
            let mut ownership_candidates = Vec::new();
            if !raw.is_codex_session_meta {
                for (session_id, repository) in &active_segments {
                    let reconstructed = RawContext {
                        raw_record_id: raw.id.clone(),
                        session_id: session_id.clone(),
                        repository: repository.clone(),
                        occurred_at_ms: raw
                            .native_context
                            .occurred_at_ms
                            .or(raw.stored_context.occurred_at_ms),
                    };
                    if !contexts.contains(&reconstructed) {
                        contexts.push(reconstructed.clone());
                    }
                    ownership_candidates.push(reconstructed);
                }
            }
            result.push(RawForForget {
                id: raw.id.clone(),
                agent: raw.agent,
                contexts,
                ownership_candidates,
                ingested_at_ms: raw.ingested_at_ms,
            });
        }
        start = end;
    }
    Ok(result)
}

async fn storage_id_from(tx: &mut sqlx::Transaction<'_, Sqlite>) -> Result<String> {
    let row = sqlx::query("SELECT value FROM storage_metadata WHERE key='storage_id'")
        .fetch_one(&mut **tx)
        .await?;
    Ok(row.try_get("value")?)
}

struct RawForForget {
    id: String,
    agent: crate::model::AgentKind,
    contexts: Vec<RawContext>,
    ownership_candidates: Vec<RawContext>,
    ingested_at_ms: i64,
}

struct ForgetPlan {
    selected_event_ids: Vec<String>,
    remaining_event_ids: Vec<String>,
    orphan_raw_record_ids: Vec<String>,
    orphan_attachment_keys: Vec<String>,
    unmanaged_attachment_keys: Vec<String>,
}

impl ForgetPlan {
    fn preview(&self) -> ForgetPreview {
        ForgetPreview {
            events: self.selected_event_ids.len(),
            raw_records: self.orphan_raw_record_ids.len(),
            attachment_objects: self.orphan_attachment_keys.len(),
            unmanaged_attachment_objects: self.unmanaged_attachment_keys.len(),
        }
    }
}

#[derive(Default)]
struct AppliedForget {
    deleted_events: usize,
    deleted_raw: usize,
}

async fn apply_forget_plan(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
    plan: &mut ForgetPlan,
    now: i64,
) -> Result<AppliedForget> {
    for event_id in &plan.selected_event_ids {
        sqlx::query("INSERT OR IGNORE INTO forgotten_events(event_id,created_at_ms) VALUES(?,?)")
            .bind(event_id)
            .bind(now)
            .execute(&mut **tx)
            .await?;
    }
    for raw_record_id in &plan.orphan_raw_record_ids {
        sqlx::query(
            "INSERT OR IGNORE INTO forgotten_raw_records(raw_record_id,created_at_ms) VALUES(?,?)",
        )
        .bind(raw_record_id)
        .bind(now)
        .execute(&mut **tx)
        .await?;
    }

    if plan.selected_event_ids.is_empty() && plan.orphan_raw_record_ids.is_empty() {
        return Ok(AppliedForget::default());
    }

    let backend_id = sqlx::query("SELECT value FROM storage_metadata WHERE key='blob_backend_id'")
        .fetch_optional(&mut **tx)
        .await?
        .map(|row| row.try_get::<String, _>("value"))
        .transpose()?;
    if let Some(backend_id) = &backend_id {
        for object_key in &plan.orphan_attachment_keys {
            sqlx::query(
                "INSERT INTO blob_gc_queue(backend_id,object_key,reason,enqueued_at_ms,not_before_ms,last_error,attempts) VALUES(?,?,'forgotten',?,?,NULL,0) ON CONFLICT(backend_id,object_key) DO UPDATE SET reason='forgotten',not_before_ms=MIN(blob_gc_queue.not_before_ms,excluded.not_before_ms),last_error=NULL",
            )
            .bind(backend_id)
            .bind(object_key)
            .bind(now)
            .bind(now)
            .execute(&mut **tx)
            .await?;
        }
    } else {
        plan.unmanaged_attachment_keys
            .append(&mut plan.orphan_attachment_keys);
    }
    let residue_backend = backend_id.as_deref().unwrap_or("unbound");
    for object_key in &plan.unmanaged_attachment_keys {
        sqlx::query(
            "INSERT OR IGNORE INTO unmanaged_blob_residue(backend_id,object_key,reason,noted_at_ms) VALUES(?,?,'legacy or foreign attachment key cannot be deleted safely',?)",
        )
        .bind(residue_backend)
        .bind(object_key)
        .bind(now)
        .execute(&mut **tx)
        .await?;
    }

    let mut deleted_events = 0_u64;
    for ids in plan.selected_event_ids.chunks(500) {
        let mut query = QueryBuilder::<Sqlite>::new("DELETE FROM events WHERE id IN (");
        let mut values = query.separated(",");
        for id in ids {
            values.push_bind(id);
        }
        values.push_unseparated(")");
        deleted_events += query.build().execute(&mut **tx).await?.rows_affected();
    }

    let mut deleted_raw = 0_u64;
    for ids in plan.orphan_raw_record_ids.chunks(500) {
        let mut query = QueryBuilder::<Sqlite>::new("DELETE FROM raw_records WHERE id IN (");
        let mut values = query.separated(",");
        for id in ids {
            values.push_bind(id);
        }
        values.push_unseparated(")");
        deleted_raw += query.build().execute(&mut **tx).await?.rows_affected();
    }

    if deleted_events > 0 {
        sqlx::query("DELETE FROM memories")
            .execute(&mut **tx)
            .await?;
        sqlx::query("DELETE FROM strategy_failures")
            .execute(&mut **tx)
            .await?;
        sqlx::query("DELETE FROM strategy_checkpoints")
            .execute(&mut **tx)
            .await?;
        sqlx::query("DELETE FROM domain_events")
            .execute(&mut **tx)
            .await?;
        if !plan.remaining_event_ids.is_empty() {
            sqlx::query(
                "INSERT INTO domain_events(kind,payload_json,created_at_ms) VALUES('events_ingested',?,?)",
            )
            .bind(serde_json::to_string(&plan.remaining_event_ids)?)
            .bind(now)
            .execute(&mut **tx)
            .await?;
        }
    }
    if deleted_events > 0 || deleted_raw > 0 {
        sqlx::query(
            "INSERT INTO storage_metadata(key,value) VALUES('sqlite_cleanup_pending',lower(hex(randomblob(16)))) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        )
        .execute(&mut **tx)
        .await?;
    }
    Ok(AppliedForget {
        deleted_events: deleted_events as usize,
        deleted_raw: deleted_raw as usize,
    })
}

fn build_forget_plan(
    events: Vec<CanonicalEvent>,
    raw_links: Vec<(String, String)>,
    raw_records: Vec<RawForForget>,
    filter: &ForgetFilter,
    storage_id: &str,
) -> Result<ForgetPlan> {
    filter.validate()?;
    for (name, value) in [
        ("session", filter.session_id.as_deref()),
        ("repository", filter.repository.as_deref()),
    ] {
        anyhow::ensure!(
            value.is_none_or(|value| !value.trim().is_empty()),
            "forget {name} selector cannot be empty"
        );
    }

    let directly_selected_event_ids = events
        .iter()
        .filter(|event| {
            filter
                .session_id
                .as_deref()
                .is_none_or(|value| event.source.session_id == value)
                && filter
                    .repository
                    .as_deref()
                    .is_none_or(|value| event.source.repository.as_deref() == Some(value))
                && filter.agent.is_none_or(|value| event.source.agent == value)
                && filter
                    .before_ms
                    .is_none_or(|value| event.ordering_timestamp_ms() < value)
        })
        .map(|event| event.id.clone())
        .collect::<Vec<_>>();
    let context_matches = |raw: &RawForForget, context: &RawContext| {
        filter.agent.is_none_or(|agent| raw.agent == agent)
            && filter
                .session_id
                .as_deref()
                .is_none_or(|value| context.session_id.as_deref() == Some(value))
            && filter
                .repository
                .as_deref()
                .is_none_or(|value| context.repository.as_deref() == Some(value))
            && filter
                .before_ms
                .is_none_or(|before| context.occurred_at_ms.unwrap_or(raw.ingested_at_ms) < before)
    };
    let context_selected_raw = raw_records
        .iter()
        .filter(|raw| {
            raw.contexts
                .iter()
                .any(|context| context_matches(raw, context))
        })
        .map(|raw| raw.id.clone())
        .collect::<HashSet<_>>();
    let plan = build_forget_plan_from_selection(
        events,
        raw_links,
        directly_selected_event_ids.into_iter().collect(),
        context_selected_raw,
        storage_id,
    );
    let selected_raw = plan.orphan_raw_record_ids.iter().collect::<HashSet<_>>();
    for raw in &raw_records {
        if selected_raw.contains(&raw.id) && raw.ownership_candidates.len() > 1 {
            let matching = raw
                .ownership_candidates
                .iter()
                .filter(|context| context_matches(raw, context))
                .count();
            anyhow::ensure!(
                matching == raw.ownership_candidates.len(),
                "raw record {} has ambiguous Codex session ownership; use a broader forget selector that covers every candidate session",
                raw.id
            );
        }
    }
    Ok(plan)
}

fn build_forget_plan_from_selection(
    events: Vec<CanonicalEvent>,
    raw_links: Vec<(String, String)>,
    mut selected: HashSet<String>,
    mut selected_raw: HashSet<String>,
    storage_id: &str,
) -> ForgetPlan {
    // Deletion is closed over the event/raw provenance graph. If a native
    // record produced events on both sides of a time bound, retaining either
    // the plaintext raw payload or a projection derived from it would violate
    // the forget request (and the archive's mandatory provenance contract).
    loop {
        let event_count = selected.len();
        let raw_count = selected_raw.len();
        for (event_id, raw_id) in &raw_links {
            if selected.contains(event_id) {
                selected_raw.insert(raw_id.clone());
            }
            if selected_raw.contains(raw_id) {
                selected.insert(event_id.clone());
            }
        }
        if selected.len() == event_count && selected_raw.len() == raw_count {
            break;
        }
    }
    let mut selected_event_ids = selected.iter().cloned().collect::<Vec<_>>();
    selected_event_ids.sort();
    let remaining_event_ids = events
        .iter()
        .filter(|event| !selected.contains(&event.id))
        .map(|event| event.id.clone())
        .collect::<Vec<_>>();
    let mut orphan_raw_record_ids = selected_raw.into_iter().collect::<Vec<_>>();
    orphan_raw_record_ids.sort();

    let selected_attachments = events
        .iter()
        .filter(|event| selected.contains(&event.id))
        .flat_map(CanonicalEvent::attachment_refs)
        .map(|attachment| attachment.object_key.clone())
        .collect::<HashSet<_>>();
    let remaining_attachments = events
        .iter()
        .filter(|event| !selected.contains(&event.id))
        .flat_map(CanonicalEvent::attachment_refs)
        .map(|attachment| attachment.object_key.clone())
        .collect::<HashSet<_>>();
    let mut orphan_attachment_keys = selected_attachments
        .difference(&remaining_attachments)
        .cloned()
        .collect::<Vec<_>>();
    orphan_attachment_keys.sort();
    let managed_prefix = format!("semafuse/{storage_id}/");
    let (orphan_attachment_keys, unmanaged_attachment_keys) = orphan_attachment_keys
        .into_iter()
        .partition(|key| key.starts_with(&managed_prefix));

    ForgetPlan {
        selected_event_ids,
        remaining_event_ids,
        orphan_raw_record_ids,
        orphan_attachment_keys,
        unmanaged_attachment_keys,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AgentKind, AttachmentRef, EventSource, MessageRole};
    use serde_json::json;

    fn message(
        session: &str,
        native_id: &str,
        raw_id: &str,
        timestamp: i64,
        attachments: Vec<AttachmentRef>,
    ) -> CanonicalEvent {
        let source = EventSource::new(
            AgentKind::Codex,
            session,
            None,
            Some(native_id.into()),
            "message",
            None,
            Some("/repo".into()),
        );
        let mut event = CanonicalEvent::message_from_text(
            source,
            MessageRole::User,
            native_id,
            Some(timestamp),
        );
        event.attach_raw_record(raw_id);
        event.add_attachments(attachments);
        event
    }

    fn attachment(key: &str, marker: char) -> AttachmentRef {
        AttachmentRef {
            object_key: key.into(),
            digest: marker.to_string().repeat(64),
            media_type: "application/octet-stream".into(),
            size: 1,
            name: None,
        }
    }

    #[tokio::test]
    async fn forget_is_bounded_cascades_raw_data_and_preserves_shared_blobs() {
        let store = SqliteStore::open_memory().await.unwrap();
        store.bind_blob_backend("test-backend").await.unwrap();
        let prefix = format!("semafuse/{}/", store.storage_id().await.unwrap());
        let raw_old = RawRecord::new(AgentKind::Codex, "old.json", 0, json!({"old":true}));
        let raw_old_context =
            RawRecord::new(AgentKind::Codex, "old.json", 1, json!({"type":"context"}));
        let raw_new = RawRecord::new(AgentKind::Codex, "new.json", 0, json!({"new":true}));
        let unique = attachment(&format!("{prefix}sha256/aa/aa/unique"), 'a');
        let shared = attachment(&format!("{prefix}sha256/bb/bb/shared"), 'b');
        let events = vec![
            message(
                "old-session",
                "old-message",
                &raw_old.id,
                100,
                vec![unique.clone(), shared.clone()],
            ),
            message("new-session", "new-message", &raw_new.id, 200, vec![shared]),
        ];
        let diagnostic = ConversionDiagnostic::new(
            &raw_old.id,
            DiagnosticSeverity::Warning,
            "test_warning",
            "test warning",
        );
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: &[raw_old.clone(), raw_old_context.clone(), raw_new.clone()],
                raw_contexts: &[
                    RawContext {
                        raw_record_id: raw_old.id.clone(),
                        session_id: Some("old-session".into()),
                        ..RawContext::default()
                    },
                    RawContext {
                        raw_record_id: raw_old_context.id.clone(),
                        session_id: Some("old-session".into()),
                        ..RawContext::default()
                    },
                    RawContext {
                        raw_record_id: raw_new.id.clone(),
                        session_id: Some("new-session".into()),
                        ..RawContext::default()
                    },
                ],
                events: &events,
                diagnostics: &[diagnostic],
            })
            .await
            .unwrap();

        assert!(
            store
                .preview_forget(&ForgetFilter::default())
                .await
                .is_err()
        );
        let filter = ForgetFilter {
            session_id: Some("old-session".into()),
            ..ForgetFilter::default()
        };
        assert_eq!(
            store.preview_forget(&filter).await.unwrap(),
            ForgetPreview {
                events: 1,
                raw_records: 2,
                attachment_objects: 1,
                unmanaged_attachment_objects: 0,
            }
        );

        let stats = store.forget(&filter).await.unwrap();
        assert_eq!(stats.events, 1);
        assert_eq!(stats.raw_records, 2);
        assert_eq!(stats.attachment_objects_queued, 1);
        assert_eq!(stats.remaining_events, 1);
        assert!(store.raw_record(&raw_old.id).await.unwrap().is_none());
        assert!(
            store
                .raw_record(&raw_old_context.id)
                .await
                .unwrap()
                .is_none()
        );
        assert!(store.raw_record(&raw_new.id).await.unwrap().is_some());
        let archive = store.canonical_archive().await.unwrap();
        assert_eq!(archive.events.len(), 1);
        assert!(archive.diagnostics.is_empty());
        let queue = store.claim_pending_blob_gc(10).await.unwrap();
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0].object_key, unique.object_key);
    }

    #[tokio::test]
    async fn forget_rules_block_history_and_unprojected_raw_from_returning() {
        let store = SqliteStore::open_memory().await.unwrap();
        let raw = RawRecord::new(
            AgentKind::Codex,
            "history.jsonl",
            0,
            json!({"session_id":"forgotten-session","message":"private"}),
        );
        let unprojected = RawRecord::new(
            AgentKind::Codex,
            "history.jsonl",
            1,
            json!({"session_id":"forgotten-session","future":"private"}),
        );
        let event = message(
            "forgotten-session",
            "private-message",
            &raw.id,
            100,
            Vec::new(),
        );
        let contexts = [
            RawContext {
                raw_record_id: raw.id.clone(),
                session_id: Some("forgotten-session".into()),
                occurred_at_ms: Some(100),
                ..RawContext::default()
            },
            RawContext {
                raw_record_id: unprojected.id.clone(),
                session_id: Some("history-file-fallback".into()),
                occurred_at_ms: Some(100),
                ..RawContext::default()
            },
        ];
        let batch = super::super::IngestBatch {
            raw_records: &[raw.clone(), unprojected.clone()],
            raw_contexts: &contexts,
            events: std::slice::from_ref(&event),
            diagnostics: &[],
        };
        store.ingest_batch(batch).await.unwrap();
        store
            .forget(&ForgetFilter {
                session_id: Some("forgotten-session".into()),
                ..ForgetFilter::default()
            })
            .await
            .unwrap();

        let stats = store
            .ingest_batch(super::super::IngestBatch {
                raw_records: &[raw.clone(), unprojected.clone()],
                raw_contexts: &contexts,
                events: std::slice::from_ref(&event),
                diagnostics: &[],
            })
            .await
            .unwrap();
        assert_eq!(stats.raw_tombstoned, 2);
        assert_eq!(stats.tombstoned, 1);
        assert_eq!(store.count().await.unwrap(), 0);
        assert!(store.raw_record(&raw.id).await.unwrap().is_none());
        assert!(store.raw_record(&unprojected.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn replay_rule_match_removes_legacy_event_provenance_as_one_closure() {
        let store = SqliteStore::open_memory().await.unwrap();
        store.bind_blob_backend("legacy-backend").await.unwrap();
        let attachment_key = format!(
            "semafuse/{}/sha256/aa/bb/legacy",
            store.storage_id().await.unwrap()
        );
        let raw = RawRecord::new(
            AgentKind::Codex,
            "legacy-without-context.jsonl",
            0,
            json!({"secret":"legacy plaintext"}),
        );
        let event = message(
            "legacy-unknown-session",
            "legacy-event",
            &raw.id,
            100,
            vec![attachment(&attachment_key, 'a')],
        );
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw),
                raw_contexts: &[],
                events: std::slice::from_ref(&event),
                diagnostics: &[],
            })
            .await
            .unwrap();
        sqlx::query("INSERT INTO forget_rules(session_hash,created_at_ms) VALUES(?,?)")
            .bind(selector_hash("forgotten-session"))
            .bind(crate::model::now_ms())
            .execute(&store.pool)
            .await
            .unwrap();

        let stats = store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw),
                raw_contexts: &[RawContext {
                    raw_record_id: raw.id.clone(),
                    session_id: Some("forgotten-session".into()),
                    ..RawContext::default()
                }],
                events: &[],
                diagnostics: &[],
            })
            .await
            .unwrap();

        assert_eq!(stats.raw_tombstoned, 1);
        assert_eq!(stats.attachment_objects_queued, 1);
        assert_eq!(stats.unmanaged_attachment_objects, 0);
        assert_eq!(store.pending_blob_gc_count().await.unwrap(), 1);
        assert!(!stats.sqlite_cleanup_pending);
        assert_eq!(store.count().await.unwrap(), 0);
        assert!(store.raw_record(&raw.id).await.unwrap().is_none());
        assert!(store.canonical_archive().await.unwrap().events.is_empty());
        assert!(store.event_is_tombstoned(&event).await.unwrap());
    }

    #[tokio::test]
    async fn replay_batch_cannot_reinsert_a_raw_tombstoned_by_an_earlier_closure() {
        let store = SqliteStore::open_memory().await.unwrap();
        let raw_a = RawRecord::new(
            AgentKind::Codex,
            "legacy-shared.jsonl",
            0,
            json!({"secret":"selected plaintext"}),
        );
        let raw_b = RawRecord::new(
            AgentKind::Codex,
            "legacy-shared.jsonl",
            1,
            json!({"secret":"closure plaintext"}),
        );
        let mut event = message(
            "legacy-unknown-session",
            "legacy-shared-event",
            &raw_a.id,
            100,
            Vec::new(),
        );
        event.attach_raw_record(raw_b.id.clone());
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: &[raw_a.clone(), raw_b.clone()],
                raw_contexts: &[],
                events: std::slice::from_ref(&event),
                diagnostics: &[],
            })
            .await
            .unwrap();
        sqlx::query("INSERT INTO forget_rules(session_hash,created_at_ms) VALUES(?,?)")
            .bind(selector_hash("forgotten-session"))
            .bind(crate::model::now_ms())
            .execute(&store.pool)
            .await
            .unwrap();

        let stats = store
            .ingest_batch(super::super::IngestBatch {
                raw_records: &[raw_a.clone(), raw_b.clone()],
                raw_contexts: &[
                    RawContext {
                        raw_record_id: raw_a.id.clone(),
                        session_id: Some("forgotten-session".into()),
                        ..RawContext::default()
                    },
                    RawContext {
                        raw_record_id: raw_b.id.clone(),
                        session_id: Some("other-session".into()),
                        ..RawContext::default()
                    },
                ],
                events: &[],
                diagnostics: &[],
            })
            .await
            .unwrap();

        assert_eq!(stats.raw_tombstoned, 2);
        assert!(store.raw_record(&raw_a.id).await.unwrap().is_none());
        assert!(store.raw_record(&raw_b.id).await.unwrap().is_none());
        assert_eq!(store.count().await.unwrap(), 0);
        assert!(store.canonical_archive().await.unwrap().events.is_empty());
    }

    #[tokio::test]
    async fn incoming_provenance_closure_rejects_chained_raws_and_all_projection_contexts() {
        let store = SqliteStore::open_memory().await.unwrap();
        store
            .forget(&ForgetFilter {
                session_id: Some("forgotten-projection".into()),
                ..ForgetFilter::default()
            })
            .await
            .unwrap();
        let raws = (0..3)
            .map(|index| {
                RawRecord::new(
                    AgentKind::Codex,
                    "incoming-chain.jsonl",
                    index,
                    json!({"session_id":"allowed","index":index,"secret":"plaintext"}),
                )
            })
            .collect::<Vec<_>>();
        let mut event_ab = message("allowed", "event-ab", &raws[0].id, 100, Vec::new());
        event_ab.attach_raw_record(raws[1].id.clone());
        let mut event_bc = message(
            "forgotten-projection",
            "event-bc",
            &raws[1].id,
            200,
            Vec::new(),
        );
        event_bc.attach_raw_record(raws[2].id.clone());
        let contexts = raws
            .iter()
            .map(|raw| RawContext {
                raw_record_id: raw.id.clone(),
                session_id: Some("allowed".into()),
                ..RawContext::default()
            })
            .collect::<Vec<_>>();

        let stats = store
            .ingest_batch(super::super::IngestBatch {
                raw_records: &raws,
                raw_contexts: &contexts,
                events: &[event_ab, event_bc],
                diagnostics: &[],
            })
            .await
            .unwrap();

        assert_eq!(stats.raw_tombstoned, 3);
        assert_eq!(stats.tombstoned, 2);
        assert_eq!(store.count().await.unwrap(), 0);
        assert!(
            store
                .canonical_archive()
                .await
                .unwrap()
                .raw_records
                .is_empty()
        );
    }

    #[tokio::test]
    async fn raw_ownership_views_remain_coherent_and_preserve_supplied_repository() {
        let store = SqliteStore::open_memory().await.unwrap();
        let raw = RawRecord::new(
            AgentKind::Codex,
            "coherent.jsonl",
            0,
            json!({"session_id":"native-session","secret":"plaintext"}),
        );
        let supplied = RawContext {
            raw_record_id: raw.id.clone(),
            session_id: Some("supplied-session".into()),
            repository: Some("/supplied/repository".into()),
            occurred_at_ms: Some(100),
        };
        sqlx::query(
            "INSERT INTO forget_rules(session_hash,repository_hash,created_at_ms) VALUES(?,?,?)",
        )
        .bind(selector_hash("native-session"))
        .bind(selector_hash("/supplied/repository"))
        .bind(crate::model::now_ms())
        .execute(&store.pool)
        .await
        .unwrap();

        let stats = store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw),
                raw_contexts: std::slice::from_ref(&supplied),
                events: &[],
                diagnostics: &[],
            })
            .await
            .unwrap();
        assert_eq!(stats.raw_inserted, 1);
        assert_eq!(stats.raw_tombstoned, 0);
        assert_eq!(
            store
                .preview_forget(&ForgetFilter {
                    session_id: Some("native-session".into()),
                    repository: Some("/supplied/repository".into()),
                    ..ForgetFilter::default()
                })
                .await
                .unwrap()
                .raw_records,
            0
        );
        assert_eq!(
            store
                .forget(&ForgetFilter {
                    repository: Some("/supplied/repository".into()),
                    ..ForgetFilter::default()
                })
                .await
                .unwrap()
                .raw_records,
            1
        );
    }

    #[tokio::test]
    async fn deduplicated_event_exports_every_normalized_raw_link() {
        let store = SqliteStore::open_memory().await.unwrap();
        let raw_a = RawRecord::new(AgentKind::Codex, "a.jsonl", 0, json!({"value":"A"}));
        let raw_b = RawRecord::new(AgentKind::Codex, "b.jsonl", 0, json!({"value":"B"}));
        let event_a = message("dedup-session", "same-message", &raw_a.id, 100, Vec::new());
        let mut event_b = event_a.clone();
        event_b.raw_record_ids.clear();
        event_b.attach_raw_record(raw_b.id.clone());
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw_a),
                raw_contexts: &[],
                events: std::slice::from_ref(&event_a),
                diagnostics: &[],
            })
            .await
            .unwrap();
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw_b),
                raw_contexts: &[],
                events: std::slice::from_ref(&event_b),
                diagnostics: &[],
            })
            .await
            .unwrap();
        // Simulate a legacy row whose JSON predates link synchronization.
        sqlx::query("UPDATE events SET event_json=? WHERE id=?")
            .bind(serde_json::to_string(&event_a).unwrap())
            .bind(&event_a.id)
            .execute(&store.pool)
            .await
            .unwrap();

        let archive = store.canonical_archive().await.unwrap();
        assert_eq!(archive.events.len(), 1);
        let raw_ids = archive.events[0]
            .raw_record_ids
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        assert_eq!(raw_ids, HashSet::from([raw_a.id, raw_b.id]));
    }

    #[tokio::test]
    async fn persistent_ingest_rejects_missing_raw_provenance() {
        let temp = tempfile::TempDir::new().unwrap();
        let store = SqliteStore::open(&temp.path().join("provenance.db"))
            .await
            .unwrap();
        let without_raw = CanonicalEvent::message_from_text(
            EventSource::new(
                AgentKind::Codex,
                "persistent-session",
                None,
                Some("missing".into()),
                "message",
                None,
                None,
            ),
            MessageRole::User,
            "missing provenance",
            Some(100),
        );
        assert!(
            store
                .ingest(std::slice::from_ref(&without_raw))
                .await
                .unwrap_err()
                .to_string()
                .contains("has no raw provenance")
        );

        let mut unknown_raw = without_raw.clone();
        unknown_raw.attach_raw_record("raw_does_not_exist");
        assert!(
            store
                .ingest(std::slice::from_ref(&unknown_raw))
                .await
                .unwrap_err()
                .to_string()
                .contains("missing raw provenance")
        );
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn active_blob_gc_claim_blocks_a_new_attachment_reference() {
        let store = SqliteStore::open_memory().await.unwrap();
        store.bind_blob_backend("test-backend").await.unwrap();
        let key = format!(
            "semafuse/{}/sha256/aa/bb/claimed",
            store.storage_id().await.unwrap()
        );
        store
            .register_blob_candidate("test-backend", &key)
            .await
            .unwrap();
        sqlx::query("UPDATE blob_gc_queue SET not_before_ms=0 WHERE object_key=?")
            .bind(&key)
            .execute(&store.pool)
            .await
            .unwrap();
        let claim = store.claim_pending_blob_gc(1).await.unwrap().remove(0);
        assert!(
            store
                .register_blob_candidate("test-backend", &key)
                .await
                .unwrap_err()
                .to_string()
                .contains("deletion is already in progress")
        );
        let raw = RawRecord::new(AgentKind::Codex, "claimed.json", 0, json!({"value":1}));
        let event = message(
            "claimed-session",
            "claimed-event",
            &raw.id,
            100,
            vec![attachment(&key, 'c')],
        );
        assert!(
            store
                .ingest_batch(super::super::IngestBatch {
                    raw_records: std::slice::from_ref(&raw),
                    raw_contexts: &[],
                    events: std::slice::from_ref(&event),
                    diagnostics: &[],
                })
                .await
                .unwrap_err()
                .to_string()
                .contains("deletion is in progress")
        );
        store
            .fail_blob_gc(
                &claim.backend_id,
                &claim.object_key,
                &claim.claim_token,
                "retry",
            )
            .await
            .unwrap();
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw),
                raw_contexts: &[],
                events: std::slice::from_ref(&event),
                diagnostics: &[],
            })
            .await
            .unwrap();
        assert_eq!(store.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn cutoff_forget_cascades_across_shared_raw_provenance() {
        let store = SqliteStore::open_memory().await.unwrap();
        let raw = RawRecord::new(
            AgentKind::Codex,
            "shared.jsonl",
            0,
            json!({"session_id":"shared-session","secret":"old plaintext"}),
        );
        let older = message("shared-session", "older", &raw.id, 100, Vec::new());
        let newer = message("shared-session", "newer", &raw.id, 200, Vec::new());
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw),
                raw_contexts: &[],
                events: &[older, newer.clone()],
                diagnostics: &[],
            })
            .await
            .unwrap();

        let filter = ForgetFilter {
            before_ms: Some(150),
            ..ForgetFilter::default()
        };
        let preview = store.preview_forget(&filter).await.unwrap();
        assert_eq!(preview.events, 2);
        assert_eq!(preview.raw_records, 1);
        let stats = store.forget(&filter).await.unwrap();
        assert_eq!(stats.events, 2);
        assert_eq!(stats.raw_records, 1);
        assert!(store.raw_record(&raw.id).await.unwrap().is_none());
        assert!(
            store
                .session_events("shared-session")
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            store
                .canonical_archive()
                .await
                .unwrap()
                .raw_records
                .is_empty()
        );
    }

    #[tokio::test]
    async fn forget_reads_agent_native_context_when_projection_never_succeeded() {
        let store = SqliteStore::open_memory().await.unwrap();
        let raw = RawRecord::new(
            AgentKind::Amp,
            "misleading-file-name.json",
            0,
            json!({"id":"T-native-session","messages":"malformed","secret":"plaintext"}),
        );
        let misleading_context = RawContext {
            raw_record_id: raw.id.clone(),
            session_id: Some("misleading-file-name".into()),
            ..RawContext::default()
        };
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw),
                raw_contexts: std::slice::from_ref(&misleading_context),
                events: &[],
                diagnostics: &[],
            })
            .await
            .unwrap();

        let filter = ForgetFilter {
            session_id: Some("T-native-session".into()),
            ..ForgetFilter::default()
        };
        assert_eq!(store.preview_forget(&filter).await.unwrap().raw_records, 1);
        assert_eq!(store.forget(&filter).await.unwrap().raw_records, 1);
        assert!(store.raw_record(&raw.id).await.unwrap().is_none());

        let replay = store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw),
                raw_contexts: std::slice::from_ref(&misleading_context),
                events: &[],
                diagnostics: &[],
            })
            .await
            .unwrap();
        assert_eq!(replay.raw_tombstoned, 1);
        assert!(store.raw_record(&raw.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn repeated_forget_selector_creates_one_durable_rule() {
        let store = SqliteStore::open_memory().await.unwrap();
        let filter = ForgetFilter {
            session_id: Some("repeat-session".into()),
            agent: Some(AgentKind::Codex),
            ..ForgetFilter::default()
        };
        store.forget(&filter).await.unwrap();
        store.forget(&filter).await.unwrap();

        let count: i64 = sqlx::query("SELECT COUNT(*) AS count FROM forget_rules")
            .fetch_one(&store.pool)
            .await
            .unwrap()
            .try_get("count")
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn repeated_empty_forget_retries_pending_sqlite_cleanup() {
        let store = SqliteStore::open_memory().await.unwrap();
        sqlx::query("INSERT INTO storage_metadata(key,value) VALUES('sqlite_cleanup_pending','1')")
            .execute(&store.pool)
            .await
            .unwrap();
        let stats = store
            .forget(&ForgetFilter {
                session_id: Some("already-empty-session".into()),
                ..ForgetFilter::default()
            })
            .await
            .unwrap();

        assert!(!stats.sqlite_cleanup_pending);
        assert!(!store.sqlite_cleanup_is_pending().await.unwrap());
    }

    #[tokio::test]
    async fn cleanup_generation_compare_delete_preserves_a_later_request() {
        let store = SqliteStore::open_memory().await.unwrap();
        sqlx::query(
            "INSERT INTO storage_metadata(key,value) VALUES('sqlite_cleanup_pending','generation-a')",
        )
        .execute(&store.pool)
        .await
        .unwrap();
        let captured = store.sqlite_cleanup_generation().await.unwrap().unwrap();
        sqlx::query(
            "UPDATE storage_metadata SET value='generation-b' WHERE key='sqlite_cleanup_pending'",
        )
        .execute(&store.pool)
        .await
        .unwrap();

        store
            .clear_sqlite_cleanup_generation(&captured)
            .await
            .unwrap();

        assert_eq!(
            store.sqlite_cleanup_generation().await.unwrap().as_deref(),
            Some("generation-b")
        );
    }

    #[tokio::test]
    async fn busy_scrub_is_reported_and_retried_without_losing_logical_deletion() {
        let temp = tempfile::TempDir::new().unwrap();
        let store = SqliteStore::open(&temp.path().join("busy-scrub.db"))
            .await
            .unwrap();
        let raw = RawRecord::new(
            AgentKind::Codex,
            "busy.json",
            0,
            json!({"session_id":"busy-session","secret":"private"}),
        );
        let event = message("busy-session", "busy-event", &raw.id, 100, Vec::new());
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw),
                raw_contexts: &[],
                events: std::slice::from_ref(&event),
                diagnostics: &[],
            })
            .await
            .unwrap();

        let mut reader = store.pool.begin().await.unwrap();
        sqlx::query("SELECT event_json FROM events")
            .fetch_one(&mut *reader)
            .await
            .unwrap();
        let filter = ForgetFilter {
            session_id: Some("busy-session".into()),
            ..ForgetFilter::default()
        };
        let first = store.forget(&filter).await.unwrap();
        assert_eq!(first.events, 1);
        assert!(first.sqlite_cleanup_pending);
        assert!(first.sqlite_cleanup_error.is_some());
        assert_eq!(store.count().await.unwrap(), 0);
        reader.commit().await.unwrap();

        let retry = store.forget(&filter).await.unwrap();
        assert_eq!(retry.events, 0);
        assert!(!retry.sqlite_cleanup_pending);
        assert!(!store.sqlite_cleanup_is_pending().await.unwrap());
    }

    #[tokio::test]
    async fn copied_database_cannot_reuse_the_original_attachment_namespace() {
        let temp = tempfile::TempDir::new().unwrap();
        let original = temp.path().join("original.db");
        let clone = temp.path().join("clone.db");
        let store = SqliteStore::open(&original).await.unwrap();
        let storage_id = store.storage_id().await.unwrap();
        store.pool.close().await;
        std::fs::copy(&original, &clone).unwrap();

        let error = match SqliteStore::open(&clone).await {
            Ok(_) => panic!("a database clone unexpectedly opened"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("database copy or relocation"));
        let reopened = SqliteStore::open(&original).await.unwrap();
        assert_eq!(reopened.storage_id().await.unwrap(), storage_id);
    }

    #[tokio::test]
    async fn busy_location_checkpoint_remains_pending_and_retries_on_next_bind() {
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("location-retry.db");
        let store = SqliteStore::open(&path).await.unwrap();
        let canonical = path.canonicalize().unwrap();
        sqlx::query(
            "DELETE FROM storage_metadata WHERE key IN ('database_location_hash','database_location_identity_version','database_location_checkpoint_pending')",
        )
        .execute(&store.pool)
        .await
        .unwrap();
        let checkpoint = sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
            .fetch_one(&store.pool)
            .await
            .unwrap();
        assert_eq!(checkpoint.try_get::<i64, _>(0).unwrap(), 0);
        let mut reader = store.pool.begin().await.unwrap();
        sqlx::query("SELECT COUNT(*) FROM storage_metadata")
            .fetch_one(&mut *reader)
            .await
            .unwrap();

        let error = store.bind_database_location(&canonical).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("could not durably checkpoint database location identity")
        );
        let pending: i64 = sqlx::query(
            "SELECT EXISTS(SELECT 1 FROM storage_metadata WHERE key='database_location_checkpoint_pending')",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
        assert_eq!(pending, 1);

        reader.commit().await.unwrap();
        store.bind_database_location(&canonical).await.unwrap();
        let pending: i64 = sqlx::query(
            "SELECT EXISTS(SELECT 1 FROM storage_metadata WHERE key='database_location_checkpoint_pending')",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap()
        .try_get(0)
        .unwrap();
        assert_eq!(pending, 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_and_real_path_share_one_lifecycle_lock() {
        let temp = tempfile::TempDir::new().unwrap();
        let original = temp.path().join("original.db");
        let alias = temp.path().join("alias.db");
        let direct = SqliteStore::open(&original).await.unwrap();
        std::os::unix::fs::symlink(&original, &alias).unwrap();
        let through_alias = SqliteStore::open(&alias).await.unwrap();

        assert_eq!(direct.lifecycle_lock, through_alias.lifecycle_lock);
        assert_eq!(
            direct.lifecycle_lock,
            Some(PathBuf::from(format!(
                "{}.lifecycle.lock",
                original.canonicalize().unwrap().display()
            )))
        );
    }

    #[tokio::test]
    async fn governance_backfill_recovers_unprojected_agent_native_context_once() {
        let store = SqliteStore::open_memory().await.unwrap();
        let raw = RawRecord::new(
            AgentKind::OpenCode,
            "legacy-opencode.json",
            0,
            json!({"info":{"id":"legacy-open-session","directory":"/legacy/repo"},"messages":"malformed"}),
        );
        sqlx::query(
            "INSERT INTO raw_records(id,digest,agent,source,record_index,native_type,payload_json,ingested_at_ms) VALUES(?,?,?,?,?,?,?,?)",
        )
        .bind(&raw.id)
        .bind(&raw.digest)
        .bind(raw.agent.as_str())
        .bind(&raw.source)
        .bind(raw.record_index as i64)
        .bind(&raw.native_type)
        .bind(serde_json::to_string(&raw.payload).unwrap())
        .bind(raw.ingested_at_ms)
        .execute(&store.pool)
        .await
        .unwrap();
        sqlx::query("DELETE FROM storage_metadata WHERE key='governance_backfill_v2'")
            .execute(&store.pool)
            .await
            .unwrap();

        store.backfill_governance_indexes().await.unwrap();
        let row =
            sqlx::query("SELECT session_id,repository FROM raw_contexts WHERE raw_record_id=?")
                .bind(&raw.id)
                .fetch_one(&store.pool)
                .await
                .unwrap();
        assert_eq!(
            row.try_get::<String, _>("session_id").unwrap(),
            "legacy-open-session"
        );
        assert_eq!(
            row.try_get::<String, _>("repository").unwrap(),
            "/legacy/repo"
        );
        assert_eq!(
            sqlx::query("SELECT value FROM storage_metadata WHERE key='governance_backfill_v2'")
                .fetch_one(&store.pool)
                .await
                .unwrap()
                .try_get::<String, _>("value")
                .unwrap(),
            "complete"
        );
    }

    #[tokio::test]
    async fn legacy_attachment_residue_remains_auditable_after_event_cascade() {
        let store = SqliteStore::open_memory().await.unwrap();
        store.bind_blob_backend("legacy-backend").await.unwrap();
        let raw = RawRecord::new(
            AgentKind::Codex,
            "legacy.jsonl",
            0,
            json!({"session_id":"legacy-session"}),
        );
        let event = message(
            "legacy-session",
            "legacy-message",
            &raw.id,
            100,
            vec![attachment("sha256/legacy-object", 'c')],
        );
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw),
                raw_contexts: &[],
                events: std::slice::from_ref(&event),
                diagnostics: &[],
            })
            .await
            .unwrap();
        let stats = store
            .forget(&ForgetFilter {
                session_id: Some("legacy-session".into()),
                ..ForgetFilter::default()
            })
            .await
            .unwrap();

        assert_eq!(stats.unmanaged_attachment_objects, 1);
        assert_eq!(store.unmanaged_blob_residue_count().await.unwrap(), 1);
        let repeated = store
            .forget(&ForgetFilter {
                session_id: Some("legacy-session".into()),
                ..ForgetFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(repeated.unmanaged_attachment_objects, 0);
        assert_eq!(store.unmanaged_blob_residue_count().await.unwrap(), 1);
        assert!(store.canonical_archive().await.unwrap().events.is_empty());
    }

    #[tokio::test]
    async fn exact_tombstone_blocks_timestamp_free_raw_after_before_cutoff() {
        let store = SqliteStore::open_memory().await.unwrap();
        let raw = RawRecord::new(
            AgentKind::Codex,
            "timestamp-free.jsonl",
            0,
            json!({"future_native_record":"private"}),
        );
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw),
                raw_contexts: &[],
                events: &[],
                diagnostics: &[],
            })
            .await
            .unwrap();
        store
            .forget(&ForgetFilter {
                before_ms: Some(raw.ingested_at_ms.saturating_add(1)),
                ..ForgetFilter::default()
            })
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let replayed = RawRecord::new(
            AgentKind::Codex,
            "timestamp-free.jsonl",
            0,
            json!({"future_native_record":"private"}),
        );
        assert_eq!(replayed.id, raw.id);
        let stats = store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&replayed),
                raw_contexts: &[],
                events: &[],
                diagnostics: &[],
            })
            .await
            .unwrap();
        assert_eq!(stats.raw_tombstoned, 1);
        assert!(store.raw_record(&raw.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn stale_strategy_delivery_cannot_restore_memory_after_forget() {
        let store = SqliteStore::open_memory().await.unwrap();
        let event = message("strategy-forget", "message", "raw_missing", 100, Vec::new());
        let sequence = store
            .ingest(std::slice::from_ref(&event))
            .await
            .unwrap()
            .domain_event_seq
            .unwrap();
        store
            .forget(&ForgetFilter {
                session_id: Some("strategy-forget".into()),
                ..ForgetFilter::default()
            })
            .await
            .unwrap();
        let record = MemoryRecord {
            namespace: "history".into(),
            id: "stale".into(),
            content: "must not return".into(),
            metadata: serde_json::Map::new(),
            source_event_ids: vec![event.id],
            created_at_ms: 100,
            updated_at_ms: 100,
        };
        assert!(
            store
                .apply_strategy_event("history", sequence, &[MemoryMutation::Upsert(record)])
                .await
                .is_err()
        );
        assert!(store.all_memories("history").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn confirmed_forget_scrubs_plaintext_from_sqlite_and_wal() {
        let temp = tempfile::TempDir::new().unwrap();
        let path = temp.path().join("scrub.db");
        let marker = "SEMAFUSE_PHYSICAL_ERASURE_MARKER_91A7";
        let store = SqliteStore::open(&path).await.unwrap();
        let raw = RawRecord::new(
            AgentKind::Codex,
            "sensitive.jsonl",
            0,
            json!({"session_id":"erase-session","secret":marker}),
        );
        let event = message(
            "erase-session",
            "sensitive-message",
            &raw.id,
            100,
            Vec::new(),
        );
        let diagnostic = ConversionDiagnostic::new(
            &raw.id,
            DiagnosticSeverity::Warning,
            "sensitive_marker",
            marker,
        );
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: std::slice::from_ref(&raw),
                raw_contexts: &[RawContext {
                    raw_record_id: raw.id.clone(),
                    session_id: Some("erase-session".into()),
                    ..RawContext::default()
                }],
                events: std::slice::from_ref(&event),
                diagnostics: std::slice::from_ref(&diagnostic),
            })
            .await
            .unwrap();
        store
            .forget(&ForgetFilter {
                session_id: Some("erase-session".into()),
                ..ForgetFilter::default()
            })
            .await
            .unwrap();
        let integrity: String = sqlx::query("PRAGMA integrity_check")
            .fetch_one(&store.pool)
            .await
            .unwrap()
            .try_get(0)
            .unwrap();
        assert_eq!(integrity, "ok");
        store.pool.close().await;
        drop(store);

        for suffix in ["", "-wal", "-shm"] {
            let mut name = path.as_os_str().to_os_string();
            name.push(suffix);
            let candidate = PathBuf::from(name);
            if let Ok(bytes) = std::fs::read(&candidate) {
                assert!(
                    !bytes
                        .windows(marker.len())
                        .any(|window| window == marker.as_bytes()),
                    "marker remained in {}",
                    candidate.display()
                );
            }
        }
    }

    #[tokio::test]
    async fn legacy_failed_codex_rollout_context_is_reconstructed_for_forget() {
        let store = SqliteStore::open_memory().await.unwrap();
        let raw_meta = RawRecord::new(
            AgentKind::Codex,
            "legacy-failed-rollout.jsonl",
            0,
            json!({
                "type":"session_meta",
                "payload":{
                    "id":"legacy-private-session",
                    "secret":"first legacy plaintext"
                }
            }),
        );
        let raw_broken = RawRecord::new(
            AgentKind::Codex,
            "legacy-failed-rollout.jsonl",
            1,
            json!({"type":123,"payload":{"secret":"second legacy plaintext"}}),
        );
        let records = [raw_meta.clone(), raw_broken.clone()];
        let misleading_contexts = records
            .iter()
            .map(|raw| RawContext {
                raw_record_id: raw.id.clone(),
                session_id: Some("legacy-file-name-fallback".into()),
                ..RawContext::default()
            })
            .collect::<Vec<_>>();
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: &records,
                raw_contexts: &misleading_contexts,
                events: &[],
                diagnostics: &[],
            })
            .await
            .unwrap();

        let filter = ForgetFilter {
            session_id: Some("legacy-private-session".into()),
            ..ForgetFilter::default()
        };
        assert_eq!(store.preview_forget(&filter).await.unwrap().raw_records, 2);
        assert_eq!(store.forget(&filter).await.unwrap().raw_records, 2);
        assert!(
            store
                .canonical_archive()
                .await
                .unwrap()
                .raw_records
                .is_empty()
        );

        let replay = store
            .ingest_batch(super::super::IngestBatch {
                raw_records: &records,
                raw_contexts: &misleading_contexts,
                events: &[],
                diagnostics: &[],
            })
            .await
            .unwrap();
        assert_eq!(replay.raw_tombstoned, 2);
        assert!(
            store
                .canonical_archive()
                .await
                .unwrap()
                .raw_records
                .is_empty()
        );
    }

    #[tokio::test]
    async fn legacy_codex_context_reconstruction_respects_session_meta_boundaries() {
        let store = SqliteStore::open_memory().await.unwrap();
        let records = [
            RawRecord::new(
                AgentKind::Codex,
                "legacy-multi-session.jsonl",
                0,
                json!({"type":"session_meta","payload":{"id":"session-a"}}),
            ),
            RawRecord::new(
                AgentKind::Codex,
                "legacy-multi-session.jsonl",
                1,
                json!({"type":123,"payload":{"secret":"session-a plaintext"}}),
            ),
            RawRecord::new(
                AgentKind::Codex,
                "legacy-multi-session.jsonl",
                2,
                json!({"type":"session_meta","payload":{"id":"session-b"}}),
            ),
            RawRecord::new(
                AgentKind::Codex,
                "legacy-multi-session.jsonl",
                3,
                json!({"type":123,"payload":{"secret":"session-b plaintext"}}),
            ),
        ];
        let misleading_contexts = records
            .iter()
            .map(|raw| RawContext {
                raw_record_id: raw.id.clone(),
                session_id: Some("legacy-file-name-fallback".into()),
                ..RawContext::default()
            })
            .collect::<Vec<_>>();
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: &records,
                raw_contexts: &misleading_contexts,
                events: &[],
                diagnostics: &[],
            })
            .await
            .unwrap();

        let stats = store
            .forget(&ForgetFilter {
                session_id: Some("session-a".into()),
                ..ForgetFilter::default()
            })
            .await
            .unwrap();
        assert_eq!(stats.raw_records, 2);
        let retained = store.canonical_archive().await.unwrap().raw_records;
        assert_eq!(retained.len(), 2);
        assert!(
            retained
                .iter()
                .all(|raw| raw.payload.to_string().contains("session-b"))
        );
    }

    #[tokio::test]
    async fn ambiguous_legacy_codex_ownership_blocks_bounded_forget() {
        let store = SqliteStore::open_memory().await.unwrap();
        let records = [
            RawRecord::new(
                AgentKind::Codex,
                "reused-rollout-path.jsonl",
                0,
                json!({"type":"session_meta","payload":{"id":"session-a"}}),
            ),
            RawRecord::new(
                AgentKind::Codex,
                "reused-rollout-path.jsonl",
                0,
                json!({"type":"session_meta","payload":{"id":"session-b"}}),
            ),
            RawRecord::new(
                AgentKind::Codex,
                "reused-rollout-path.jsonl",
                1,
                json!({"type":123,"payload":{"secret":"ambiguous plaintext"}}),
            ),
        ];
        let misleading_contexts = records
            .iter()
            .map(|raw| RawContext {
                raw_record_id: raw.id.clone(),
                session_id: Some("reused-file-name-fallback".into()),
                ..RawContext::default()
            })
            .collect::<Vec<_>>();
        let event = message(
            "event-only-session",
            "ambiguous-event",
            &records[2].id,
            100,
            Vec::new(),
        );
        store
            .ingest_batch(super::super::IngestBatch {
                raw_records: &records,
                raw_contexts: &misleading_contexts,
                events: std::slice::from_ref(&event),
                diagnostics: &[],
            })
            .await
            .unwrap();

        let bounded = ForgetFilter {
            session_id: Some("session-a".into()),
            ..ForgetFilter::default()
        };
        let preview_error = store.preview_forget(&bounded).await.unwrap_err();
        assert!(
            preview_error
                .to_string()
                .contains("ambiguous Codex session ownership")
        );
        assert!(store.forget(&bounded).await.is_err());
        assert_eq!(
            store.canonical_archive().await.unwrap().raw_records.len(),
            3
        );

        let fallback = ForgetFilter {
            session_id: Some("reused-file-name-fallback".into()),
            ..ForgetFilter::default()
        };
        assert!(store.preview_forget(&fallback).await.is_err());
        assert!(store.forget(&fallback).await.is_err());
        assert_eq!(
            store.canonical_archive().await.unwrap().raw_records.len(),
            3
        );

        let provenance = ForgetFilter {
            session_id: Some("event-only-session".into()),
            ..ForgetFilter::default()
        };
        assert!(store.preview_forget(&provenance).await.is_err());
        assert!(store.forget(&provenance).await.is_err());
        let retained = store.canonical_archive().await.unwrap();
        assert_eq!(retained.raw_records.len(), 3);
        assert_eq!(retained.events.len(), 1);

        let broad = ForgetFilter {
            agent: Some(AgentKind::Codex),
            ..ForgetFilter::default()
        };
        assert_eq!(store.preview_forget(&broad).await.unwrap().raw_records, 3);
        assert_eq!(store.forget(&broad).await.unwrap().raw_records, 3);
    }
}
