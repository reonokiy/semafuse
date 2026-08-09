//! Crash-safe, fail-closed JSONL journaling for benchmark records.
//!
//! Every committed record is one newline-terminated JSON object.  Opening an
//! existing journal validates the complete file, rejects truncated tails, and
//! requires one run identity plus unique `(run_key, method, question_id,
//! record_kind)` keys.  This module never logs record contents or secret values.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    error::Error,
    ffi::OsString,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Typed envelope stored as one JSONL record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JournalRecord<T = Value> {
    /// Stable identity for the complete benchmark run.
    pub run_key: String,
    /// Stable method/configuration identifier.
    pub method: String,
    /// Stable benchmark question identifier.
    pub question_id: String,
    /// Record class, such as `retrieval` or `evaluation`.
    pub record_kind: String,
    /// Caller-defined, serde-compatible record payload.
    pub payload: T,
}

/// Unique resume key for one journal record.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct JournalRecordKey {
    /// Stable identity for the complete benchmark run.
    pub run_key: String,
    /// Stable method/configuration identifier.
    pub method: String,
    /// Stable benchmark question identifier.
    pub question_id: String,
    /// Record class.
    pub record_kind: String,
}

impl<T> From<&JournalRecord<T>> for JournalRecordKey {
    fn from(record: &JournalRecord<T>) -> Self {
        Self {
            run_key: record.run_key.clone(),
            method: record.method.clone(),
            question_id: record.question_id.clone(),
            record_kind: record.record_kind.clone(),
        }
    }
}

/// Top-level envelope written by [`write_summary_atomic`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummaryEnvelope<T = Value> {
    /// Stable identity of the run summarized by `summary`.
    pub run_key: String,
    /// Caller-defined summary payload.
    pub summary: T,
}

/// Journal or summary validation failure.
#[derive(Debug)]
pub enum BenchmarkJournalError {
    /// A filesystem operation failed.
    Io {
        /// Stable description of the failed operation; never record contents.
        operation: &'static str,
        /// Underlying filesystem error.
        source: io::Error,
    },
    /// Expected run identity was empty or whitespace-only.
    EmptyExpectedRunKey,
    /// A record identity field was empty or whitespace-only.
    EmptyRecordField {
        /// One-based source line, or `None` for a record being appended.
        line: Option<usize>,
        /// Name of the required identity field.
        field: &'static str,
    },
    /// A JSONL line was not valid JSON or did not match the typed envelope.
    InvalidRecord {
        /// One-based source line containing the invalid record.
        line: usize,
    },
    /// A non-empty final line lacked its commit newline.
    TruncatedRecord {
        /// One-based source line containing the incomplete record.
        line: usize,
    },
    /// A record used a different run identity from the opened journal.
    MixedRunKey {
        /// One-based source line, or `None` for a record being appended.
        line: Option<usize>,
    },
    /// A complete record key occurred more than once.
    DuplicateRecordKey {
        /// One-based duplicate line, or `None` for a record being appended.
        line: Option<usize>,
    },
    /// A payload contained a sensitive field or recognizable credential value.
    SensitiveMaterial {
        /// One-based source line, or `None` for pending output.
        line: Option<usize>,
    },
    /// A record or summary could not be serialized safely.
    Serialization,
    /// A previous append may have partially modified the open file.
    Poisoned,
    /// An existing summary was invalid or lacked a run identity.
    InvalidExistingSummary,
    /// An existing summary belongs to a different run identity.
    SummaryRunKeyMismatch,
    /// Unique temporary summary file creation repeatedly collided.
    TemporaryFileExhausted,
}

impl fmt::Display for BenchmarkJournalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, source } => write!(formatter, "{operation}: {source}"),
            Self::EmptyExpectedRunKey => formatter.write_str("expected run key must not be empty"),
            Self::EmptyRecordField { line, field } => match line {
                Some(line) => write!(formatter, "journal line {line} has an empty {field}"),
                None => write!(formatter, "record has an empty {field}"),
            },
            Self::InvalidRecord { line } => {
                write!(
                    formatter,
                    "journal line {line} is not a valid typed JSON record"
                )
            }
            Self::TruncatedRecord { line } => {
                write!(
                    formatter,
                    "journal line {line} is missing its commit newline"
                )
            }
            Self::MixedRunKey { line } => match line {
                Some(line) => write!(formatter, "journal line {line} has a different run key"),
                None => formatter.write_str("record run key differs from the open journal"),
            },
            Self::DuplicateRecordKey { line } => match line {
                Some(line) => write!(formatter, "journal line {line} duplicates a completed key"),
                None => formatter.write_str("record key is already completed"),
            },
            Self::SensitiveMaterial { line } => match line {
                Some(line) => {
                    write!(formatter, "journal line {line} contains sensitive material")
                }
                None => formatter.write_str("record contains sensitive material"),
            },
            Self::Serialization => formatter.write_str("record serialization failed"),
            Self::Poisoned => formatter.write_str(
                "journal append state is uncertain; close and reopen the journal before resuming",
            ),
            Self::InvalidExistingSummary => {
                formatter.write_str("existing summary is invalid or lacks a run key")
            }
            Self::SummaryRunKeyMismatch => {
                formatter.write_str("existing summary belongs to a different run key")
            }
            Self::TemporaryFileExhausted => {
                formatter.write_str("could not allocate a unique summary temporary file")
            }
        }
    }
}

impl Error for BenchmarkJournalError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Open append-only journal with validated in-memory resume state.
pub struct JsonlJournal<T = Value> {
    path: PathBuf,
    expected_run_key: String,
    file: File,
    records: Vec<JournalRecord<T>>,
    completed: BTreeSet<JournalRecordKey>,
    poisoned: bool,
}

impl<T> JsonlJournal<T>
where
    T: Serialize + DeserializeOwned,
{
    /// Opens or creates a journal and validates every existing committed line.
    pub fn open(
        path: impl AsRef<Path>,
        expected_run_key: impl Into<String>,
    ) -> Result<Self, BenchmarkJournalError> {
        let path = path.as_ref().to_path_buf();
        let expected_run_key = expected_run_key.into();
        if expected_run_key.trim().is_empty() {
            return Err(BenchmarkJournalError::EmptyExpectedRunKey);
        }
        ensure_output_directory(&path)?;
        let file = open_private_append_file(&path)?;
        let mut reader = BufReader::new(
            file.try_clone()
                .map_err(|source| io_error("clone journal file", source))?,
        );
        let mut records = Vec::new();
        let mut completed = BTreeSet::new();
        let mut bytes = Vec::new();
        let mut line_number = 0;

        loop {
            bytes.clear();
            let count = reader
                .read_until(b'\n', &mut bytes)
                .map_err(|source| io_error("read journal", source))?;
            if count == 0 {
                break;
            }
            line_number += 1;
            if !bytes.ends_with(b"\n") {
                return Err(BenchmarkJournalError::TruncatedRecord { line: line_number });
            }
            let raw: Value = serde_json::from_slice(&bytes)
                .map_err(|_| BenchmarkJournalError::InvalidRecord { line: line_number })?;
            if contains_sensitive_material(&raw) {
                return Err(BenchmarkJournalError::SensitiveMaterial {
                    line: Some(line_number),
                });
            }
            let record: JournalRecord<T> = serde_json::from_value(raw)
                .map_err(|_| BenchmarkJournalError::InvalidRecord { line: line_number })?;
            validate_record(&record, Some(line_number))?;
            if record.run_key != expected_run_key {
                return Err(BenchmarkJournalError::MixedRunKey {
                    line: Some(line_number),
                });
            }
            let key = JournalRecordKey::from(&record);
            if !completed.insert(key) {
                return Err(BenchmarkJournalError::DuplicateRecordKey {
                    line: Some(line_number),
                });
            }
            records.push(record);
        }

        Ok(Self {
            path,
            expected_run_key,
            file,
            records,
            completed,
            poisoned: false,
        })
    }

    /// Path backing this journal.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Validated single run identity for this journal.
    pub fn run_key(&self) -> &str {
        &self.expected_run_key
    }

    /// Completed unique record keys available for deterministic resume.
    pub fn completed_keys(&self) -> &BTreeSet<JournalRecordKey> {
        &self.completed
    }

    /// All committed records in on-disk order.
    pub fn records(&self) -> &[JournalRecord<T>] {
        &self.records
    }

    /// Appends, flushes, and data-syncs exactly one newline-terminated record.
    ///
    /// On any write, flush, or sync failure the handle becomes poisoned.  The
    /// caller must drop and reopen it so the complete file is revalidated.
    pub fn append(&mut self, record: JournalRecord<T>) -> Result<(), BenchmarkJournalError> {
        if self.poisoned {
            return Err(BenchmarkJournalError::Poisoned);
        }
        validate_record(&record, None)?;
        if record.run_key != self.expected_run_key {
            return Err(BenchmarkJournalError::MixedRunKey { line: None });
        }
        let key = JournalRecordKey::from(&record);
        if self.completed.contains(&key) {
            return Err(BenchmarkJournalError::DuplicateRecordKey { line: None });
        }

        let raw =
            serde_json::to_value(&record).map_err(|_| BenchmarkJournalError::Serialization)?;
        if contains_sensitive_material(&raw) {
            return Err(BenchmarkJournalError::SensitiveMaterial { line: None });
        }
        let mut encoded =
            serde_json::to_vec(&raw).map_err(|_| BenchmarkJournalError::Serialization)?;
        encoded.push(b'\n');

        self.poisoned = true;
        let committed = (|| {
            self.file
                .write_all(&encoded)
                .map_err(|source| io_error("append journal record", source))?;
            self.file
                .flush()
                .map_err(|source| io_error("flush journal record", source))?;
            self.file
                .sync_data()
                .map_err(|source| io_error("sync journal record", source))?;
            Ok(())
        })();
        committed?;
        self.poisoned = false;
        self.completed.insert(key);
        self.records.push(record);
        Ok(())
    }
}

/// Writes a run-bound summary through a same-directory temporary file + rename.
///
/// An existing destination may be atomically replaced only when its top-level
/// `run_key` matches `summary.run_key`.  Existing invalid summaries and summaries
/// from another run are left untouched.  The function rejects common secret
/// field names recursively and never includes values in its errors.
pub fn write_summary_atomic<T: Serialize>(
    path: impl AsRef<Path>,
    summary: &SummaryEnvelope<T>,
) -> Result<(), BenchmarkJournalError> {
    let path = path.as_ref();
    if summary.run_key.trim().is_empty() {
        return Err(BenchmarkJournalError::EmptyExpectedRunKey);
    }
    ensure_output_directory(path)?;
    validate_existing_summary_identity(path, &summary.run_key)?;

    let raw = serde_json::to_value(summary).map_err(|_| BenchmarkJournalError::Serialization)?;
    if contains_sensitive_material(&raw) {
        return Err(BenchmarkJournalError::SensitiveMaterial { line: None });
    }
    let mut encoded =
        serde_json::to_vec_pretty(&raw).map_err(|_| BenchmarkJournalError::Serialization)?;
    encoded.push(b'\n');

    let (temporary_path, mut temporary) = create_summary_temporary(path)?;
    let write_result = (|| {
        temporary
            .write_all(&encoded)
            .map_err(|source| io_error("write summary temporary file", source))?;
        temporary
            .flush()
            .map_err(|source| io_error("flush summary temporary file", source))?;
        temporary
            .sync_data()
            .map_err(|source| io_error("sync summary temporary file", source))?;
        drop(temporary);

        // Recheck immediately before replacement.  This is a single-writer
        // primitive; the second check prevents an ordinary stale caller from
        // replacing a summary whose identity changed during preparation.
        validate_existing_summary_identity(path, &summary.run_key)?;
        fs::rename(&temporary_path, path)
            .map_err(|source| io_error("rename summary temporary file", source))?;
        sync_parent_directory(path)?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    write_result
}

/// Reads and validates an existing run-bound summary without modifying it.
///
/// The path is opened read-only and is never created. Invalid JSON, a typed
/// payload mismatch, or an empty top-level `run_key` is rejected without
/// including file contents in the error. Sensitive field names and recognizable
/// credential values are scanned recursively before typed deserialization.
pub fn read_summary<T: DeserializeOwned>(
    path: impl AsRef<Path>,
) -> Result<SummaryEnvelope<T>, BenchmarkJournalError> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|source| io_error("open summary", source))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|source| io_error("read summary", source))?;
    let raw: Value = serde_json::from_slice(&bytes)
        .map_err(|_| BenchmarkJournalError::InvalidExistingSummary)?;
    if contains_sensitive_material(&raw) {
        return Err(BenchmarkJournalError::SensitiveMaterial { line: None });
    }
    let summary: SummaryEnvelope<T> =
        serde_json::from_value(raw).map_err(|_| BenchmarkJournalError::InvalidExistingSummary)?;
    if summary.run_key.trim().is_empty() {
        return Err(BenchmarkJournalError::InvalidExistingSummary);
    }
    Ok(summary)
}

fn validate_record<T>(
    record: &JournalRecord<T>,
    line: Option<usize>,
) -> Result<(), BenchmarkJournalError> {
    for (field, value) in [
        ("run_key", record.run_key.as_str()),
        ("method", record.method.as_str()),
        ("question_id", record.question_id.as_str()),
        ("record_kind", record.record_kind.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(BenchmarkJournalError::EmptyRecordField { line, field });
        }
    }
    Ok(())
}

fn ensure_output_directory(path: &Path) -> Result<(), BenchmarkJournalError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }
    fs::create_dir_all(parent).map_err(|source| io_error("create output directory", source))
}

fn open_private_append_file(path: &Path) -> Result<File, BenchmarkJournalError> {
    let mut options = OpenOptions::new();
    options.create(true).read(true).append(true);
    #[cfg(unix)]
    options.mode(0o600);
    options
        .open(path)
        .map_err(|source| io_error("open journal", source))
}

fn validate_existing_summary_identity(
    path: &Path,
    expected_run_key: &str,
) -> Result<(), BenchmarkJournalError> {
    if !path.exists() {
        return Ok(());
    }
    let summary = read_summary::<Value>(path)?;
    if summary.run_key != expected_run_key {
        return Err(BenchmarkJournalError::SummaryRunKeyMismatch);
    }
    Ok(())
}

fn create_summary_temporary(path: &Path) -> Result<(PathBuf, File), BenchmarkJournalError> {
    let parent = path.parent().filter(|value| !value.as_os_str().is_empty());
    let parent = parent.unwrap_or_else(|| Path::new("."));
    let base = path
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("summary"));
    for _ in 0..100 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut name = OsString::from(".");
        name.push(base);
        name.push(format!(".tmp-{}-{sequence}", std::process::id()));
        let temporary_path = parent.join(name);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&temporary_path) {
            Ok(file) => return Ok((temporary_path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(io_error("create summary temporary file", source)),
        }
    }
    Err(BenchmarkJournalError::TemporaryFileExhausted)
}

#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> Result<(), BenchmarkJournalError> {
    let parent = path.parent().filter(|value| !value.as_os_str().is_empty());
    let parent = parent.unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| io_error("sync summary output directory", source))
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) -> Result<(), BenchmarkJournalError> {
    Ok(())
}

fn contains_sensitive_material(value: &Value) -> bool {
    match value {
        Value::Object(object) => object.iter().any(|(name, value)| {
            is_sensitive_field_name(name) || contains_sensitive_material(value)
        }),
        Value::Array(values) => values.iter().any(contains_sensitive_material),
        Value::String(value) => looks_like_credential(value),
        _ => false,
    }
}

fn is_sensitive_field_name(name: &str) -> bool {
    let normalized = name.trim().to_ascii_lowercase().replace('-', "_");
    if normalized.ends_with("_env") || normalized.ends_with("_env_name") {
        // Environment-variable names are identities, not credential values.
        return false;
    }
    matches!(
        normalized.as_str(),
        "apikey" | "bearer" | "credential" | "credentials" | "passwd" | "private_key" | "token"
    ) || normalized.contains("api_key")
        || normalized.contains("authorization")
        || normalized.contains("secret")
        || normalized.ends_with("_token")
        || normalized.ends_with("_password")
        || normalized.ends_with("_cookie")
}

fn looks_like_credential(value: &str) -> bool {
    let value = value.trim();
    let long_prefixed_credential = value.len() >= 20
        && ["sk-", "ghp_", "github_pat_", "hf_", "AKIA"]
            .iter()
            .any(|prefix| value.starts_with(prefix));
    long_prefixed_credential
        || value.starts_with("Bearer ")
        || value.contains("-----BEGIN PRIVATE KEY-----")
        || value.contains("-----BEGIN RSA PRIVATE KEY-----")
}

fn io_error(operation: &'static str, source: io::Error) -> BenchmarkJournalError {
    BenchmarkJournalError::Io { operation, source }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn record(run_key: &str, method: &str, question_id: &str) -> JournalRecord<Value> {
        JournalRecord {
            run_key: run_key.into(),
            method: method.into(),
            question_id: question_id.into(),
            record_kind: "retrieval".into(),
            payload: json!({"score": 1.0}),
        }
    }

    fn encoded_line(record: &JournalRecord<Value>) -> Vec<u8> {
        let mut encoded = serde_json::to_vec(record).unwrap();
        encoded.push(b'\n');
        encoded
    }

    #[test]
    fn resume_restores_records_and_completed_keys() {
        let directory = tempdir().unwrap();
        let journal_path = directory.path().join("nested/results.jsonl");
        let first = record("run-a", "baseline", "q1");
        let second = record("run-a", "candidate", "q1");

        {
            let mut journal = JsonlJournal::open(&journal_path, "run-a").unwrap();
            journal.append(first.clone()).unwrap();
            journal.append(second.clone()).unwrap();
            assert_eq!(journal.records(), &[first.clone(), second.clone()]);
            assert_eq!(journal.completed_keys().len(), 2);
        }

        let mut resumed = JsonlJournal::open(&journal_path, "run-a").unwrap();
        assert_eq!(resumed.records(), &[first.clone(), second]);
        assert!(
            resumed
                .completed_keys()
                .contains(&JournalRecordKey::from(&first))
        );
        assert!(matches!(
            resumed.append(first),
            Err(BenchmarkJournalError::DuplicateRecordKey { line: None })
        ));
    }

    #[test]
    fn truncated_final_record_is_rejected_even_when_json_is_otherwise_valid() {
        let directory = tempdir().unwrap();
        let journal_path = directory.path().join("results.jsonl");
        let encoded = serde_json::to_vec(&record("run-a", "baseline", "q1")).unwrap();
        fs::write(&journal_path, encoded).unwrap();

        assert!(matches!(
            JsonlJournal::<Value>::open(&journal_path, "run-a"),
            Err(BenchmarkJournalError::TruncatedRecord { line: 1 })
        ));
    }

    #[test]
    fn duplicate_completed_key_is_rejected_on_open() {
        let directory = tempdir().unwrap();
        let journal_path = directory.path().join("results.jsonl");
        let line = encoded_line(&record("run-a", "baseline", "q1"));
        let mut contents = line.clone();
        contents.extend(line);
        fs::write(&journal_path, contents).unwrap();

        assert!(matches!(
            JsonlJournal::<Value>::open(&journal_path, "run-a"),
            Err(BenchmarkJournalError::DuplicateRecordKey { line: Some(2) })
        ));
    }

    #[test]
    fn mixed_run_keys_are_rejected_on_open() {
        let directory = tempdir().unwrap();
        let journal_path = directory.path().join("results.jsonl");
        let mut contents = encoded_line(&record("run-a", "baseline", "q1"));
        contents.extend(encoded_line(&record("run-b", "candidate", "q1")));
        fs::write(&journal_path, contents).unwrap();

        assert!(matches!(
            JsonlJournal::<Value>::open(&journal_path, "run-a"),
            Err(BenchmarkJournalError::MixedRunKey { line: Some(2) })
        ));
    }

    #[test]
    fn atomic_summary_refuses_to_replace_another_run() {
        let directory = tempdir().unwrap();
        let summary_path = directory.path().join("reports/summary.json");
        let first = SummaryEnvelope {
            run_key: "run-a".into(),
            summary: json!({"questions": 10}),
        };
        write_summary_atomic(&summary_path, &first).unwrap();
        let original = fs::read(&summary_path).unwrap();

        let updated = SummaryEnvelope {
            run_key: "run-a".into(),
            summary: json!({"questions": 20}),
        };
        write_summary_atomic(&summary_path, &updated).unwrap();
        let parsed: SummaryEnvelope<Value> =
            serde_json::from_slice(&fs::read(&summary_path).unwrap()).unwrap();
        assert_eq!(parsed, updated);

        let wrong_run = SummaryEnvelope {
            run_key: "run-b".into(),
            summary: json!({"questions": 30}),
        };
        assert!(matches!(
            write_summary_atomic(&summary_path, &wrong_run),
            Err(BenchmarkJournalError::SummaryRunKeyMismatch)
        ));
        assert_ne!(fs::read(&summary_path).unwrap(), original);
        assert_eq!(
            serde_json::from_slice::<SummaryEnvelope<Value>>(&fs::read(&summary_path).unwrap())
                .unwrap(),
            updated
        );
    }

    #[test]
    fn summary_reader_returns_a_valid_typed_envelope() {
        #[derive(Debug, PartialEq, Deserialize)]
        struct TestSummary {
            questions: usize,
        }

        let directory = tempdir().unwrap();
        let summary_path = directory.path().join("summary.json");
        fs::write(
            &summary_path,
            serde_json::to_vec(&json!({
                "run_key": "run-a",
                "summary": {"questions": 10}
            }))
            .unwrap(),
        )
        .unwrap();

        let summary = read_summary::<TestSummary>(&summary_path).unwrap();
        assert_eq!(summary.run_key, "run-a");
        assert_eq!(summary.summary, TestSummary { questions: 10 });
    }

    #[test]
    fn summary_reader_rejects_invalid_json() {
        let directory = tempdir().unwrap();
        let summary_path = directory.path().join("summary.json");
        fs::write(&summary_path, b"not json").unwrap();

        assert!(matches!(
            read_summary::<Value>(&summary_path),
            Err(BenchmarkJournalError::InvalidExistingSummary)
        ));
    }

    #[test]
    fn summary_reader_rejects_an_empty_run_key() {
        let directory = tempdir().unwrap();
        let summary_path = directory.path().join("summary.json");
        fs::write(
            &summary_path,
            serde_json::to_vec(&json!({"run_key": "  ", "summary": {}})).unwrap(),
        )
        .unwrap();

        assert!(matches!(
            read_summary::<Value>(&summary_path),
            Err(BenchmarkJournalError::InvalidExistingSummary)
        ));
    }

    #[test]
    fn summary_reader_rejects_nested_sensitive_material() {
        let directory = tempdir().unwrap();
        let summary_path = directory.path().join("summary.json");
        fs::write(
            &summary_path,
            serde_json::to_vec(&json!({
                "run_key": "run-a",
                "summary": {"metadata": {"api_key": "redacted"}}
            }))
            .unwrap(),
        )
        .unwrap();

        assert!(matches!(
            read_summary::<Value>(&summary_path),
            Err(BenchmarkJournalError::SensitiveMaterial { line: None })
        ));
    }

    #[test]
    fn summary_reader_does_not_create_a_missing_path() {
        let directory = tempdir().unwrap();
        let summary_path = directory.path().join("missing/reports/summary.json");

        assert!(matches!(
            read_summary::<Value>(&summary_path),
            Err(BenchmarkJournalError::Io { .. })
        ));
        assert!(!summary_path.exists());
        assert!(!directory.path().join("missing").exists());
    }

    #[test]
    fn sensitive_fields_are_rejected_without_committing_a_line() {
        let directory = tempdir().unwrap();
        let journal_path = directory.path().join("results.jsonl");
        let mut journal = JsonlJournal::open(&journal_path, "run-a").unwrap();
        let mut unsafe_record = record("run-a", "baseline", "q1");
        unsafe_record.payload = json!({"api_key": "redacted"});
        assert!(matches!(
            journal.append(unsafe_record),
            Err(BenchmarkJournalError::SensitiveMaterial { line: None })
        ));
        assert!(journal.records().is_empty());
        assert!(fs::read(&journal_path).unwrap().is_empty());
    }
}
