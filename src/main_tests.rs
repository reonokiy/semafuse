use super::command_exists_in;
use std::ffi::OsStr;
use tempfile::TempDir;

#[test]
fn command_lookup_uses_pathext_with_windows_semantics() {
    let temp = TempDir::new().unwrap();
    std::fs::write(temp.path().join("codex.cmd"), b"").unwrap();
    std::fs::write(temp.path().join("amp.exe"), b"").unwrap();

    assert!(command_exists_in(
        "codex",
        Some(temp.path().as_os_str()),
        Some(OsStr::new(".EXE;.CMD")),
        true,
    ));
    assert!(command_exists_in(
        "amp",
        Some(temp.path().as_os_str()),
        Some(OsStr::new("EXE;CMD")),
        true,
    ));
    assert!(!command_exists_in(
        "claude",
        Some(temp.path().as_os_str()),
        Some(OsStr::new(".EXE;.CMD")),
        true,
    ));
}

#[test]
fn command_lookup_uses_default_windows_extensions_and_preserves_explicit_ones() {
    let temp = TempDir::new().unwrap();
    std::fs::write(temp.path().join("codex.EXE"), b"").unwrap();
    std::fs::write(temp.path().join("claude.cmd"), b"").unwrap();

    assert!(command_exists_in(
        "codex",
        Some(temp.path().as_os_str()),
        None,
        true,
    ));
    assert!(command_exists_in(
        "claude.CMD",
        Some(temp.path().as_os_str()),
        Some(OsStr::new(".EXE")),
        true,
    ));
}

#[test]
fn command_lookup_ignores_pathext_on_posix() {
    let temp = TempDir::new().unwrap();
    std::fs::write(temp.path().join("codex"), b"").unwrap();

    assert!(command_exists_in(
        "codex",
        Some(temp.path().as_os_str()),
        Some(OsStr::new(".EXE")),
        false,
    ));
    assert!(!command_exists_in(
        "codex.EXE",
        Some(temp.path().as_os_str()),
        Some(OsStr::new(".EXE")),
        false,
    ));
}
