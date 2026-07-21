use super::AgentKind;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Provenance fields shared by every canonical event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EventSource {
    /// Coding agent that emitted the native record.
    pub agent: AgentKind,
    /// Native session identifier or an explicit import fallback.
    pub session_id: String,
    /// Native turn or message grouping identifier.
    pub turn_id: Option<String>,
    /// Native event, message, or tool-call identifier.
    pub native_id: Option<String>,
    /// Native event discriminator used by the adapter.
    pub native_event: String,
    /// Working directory reported by the source.
    pub cwd: Option<String>,
    /// Explicit or locally detected repository/worktree.
    pub repository: Option<String>,
}

impl EventSource {
    /// Builds normalized provenance for a canonical event.
    pub fn new(
        agent: AgentKind,
        session_id: impl Into<String>,
        turn_id: Option<String>,
        native_id: Option<String>,
        native_event: impl Into<String>,
        cwd: Option<String>,
        repository: Option<String>,
    ) -> Self {
        Self {
            agent,
            session_id: session_id.into(),
            turn_id,
            native_id,
            native_event: native_event.into(),
            cwd: cwd.clone(),
            repository: normalize_repository(cwd.as_deref(), repository.as_deref()),
        }
    }
}

/// Resolve a stable repository/worktree identifier independently from the event cwd.
///
/// An explicit worktree wins and is retained even when it is no longer available
/// locally. Otherwise, the nearest ancestor of `cwd` containing `.git` is used.
/// A working directory outside a Git repository does not become a repository ID.
pub fn normalize_repository(cwd: Option<&str>, repository: Option<&str>) -> Option<String> {
    if let Some(repository) = repository {
        let path = Path::new(repository);
        return path
            .canonicalize()
            .unwrap_or_else(|_| path.to_path_buf())
            .to_str()
            .map(str::to_owned)
            .or_else(|| Some(repository.to_owned()));
    }

    let value = cwd?;
    let path = Path::new(value);
    let normalized = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let start = if normalized.is_file() {
        normalized.parent().unwrap_or(&normalized)
    } else {
        normalized.as_path()
    };
    start
        .ancestors()
        .find(|ancestor| ancestor.join(".git").exists())?
        .to_str()
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::normalize_repository;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn repository_is_detected_separately_from_cwd() {
        let temp = TempDir::new().unwrap();
        let repo = temp.path().join("repo");
        let cwd = repo.join("src/nested");
        let outside = std::path::PathBuf::from(format!(
            "/semafuse-non-repository-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(repo.join(".git")).unwrap();
        fs::create_dir_all(&cwd).unwrap();

        assert_eq!(
            normalize_repository(cwd.to_str(), None),
            Some(repo.canonicalize().unwrap().to_string_lossy().into_owned())
        );
        assert_eq!(normalize_repository(outside.to_str(), None), None);
    }

    #[test]
    fn explicit_worktree_is_preserved_when_unavailable_locally() {
        assert_eq!(
            normalize_repository(Some("/current/cwd"), Some("/remote/worktree")),
            Some("/remote/worktree".into())
        );
    }
}
