use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use std::{
    fs,
    io::Write,
    process::{Command as StdCommand, Stdio},
};
use tempfile::TempDir;

fn bin() -> Command {
    Command::cargo_bin("semafuse").unwrap()
}

fn session_json(db: &std::path::Path, session_id: &str) -> Value {
    let output = bin()
        .args(["session", session_id, "--db"])
        .arg(db)
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success());
    serde_json::from_slice(&output.stdout).unwrap()
}

fn event_type(event: &Value) -> &str {
    event["data"]["type"].as_str().unwrap()
}

fn event_text(event: &Value) -> String {
    fn part_text(part: &Value) -> Option<String> {
        match part["type"].as_str() {
            Some("text" | "reasoning_summary") => part["text"].as_str().map(str::to_owned),
            Some("json") => Some(part["value"].to_string()),
            _ => None,
        }
    }
    if let Some(parts) = event["data"]["content"]
        .as_array()
        .or_else(|| event["data"]["output"].as_array())
    {
        return parts
            .iter()
            .filter_map(part_text)
            .collect::<Vec<_>>()
            .join("\n");
    }
    event["data"]["summary"]
        .as_str()
        .or_else(|| event["data"]["reason"].as_str())
        .or_else(|| event["data"]["error"]["message"].as_str())
        .unwrap_or_default()
        .to_owned()
}

fn event_attachments(event: &Value) -> Vec<&Value> {
    let mut attachments = event["data"]["attachments"]
        .as_array()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if let Some(parts) = event["data"]["content"]
        .as_array()
        .or_else(|| event["data"]["output"].as_array())
    {
        attachments.extend(
            parts
                .iter()
                .filter(|part| part["type"] == "attachment")
                .map(|part| &part["attachment"]),
        );
    }
    attachments
}

fn raw_for_event(db: &std::path::Path, event_id: &str) -> Value {
    let output = bin()
        .args(["raw", event_id, "--db"])
        .arg(db)
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success());
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn imports_four_agents_deduplicates_and_searches() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("memory.db");
    for (agent, fixture) in [
        ("amp", "tests/fixtures/amp-thread.json"),
        ("codex", "tests/fixtures/codex-session.jsonl"),
        ("claude-code", "tests/fixtures/claude-session.jsonl"),
        ("opencode", "tests/fixtures/opencode-session.json"),
    ] {
        bin()
            .args(["import", "--agent", agent, "--source", fixture, "--db"])
            .arg(&db)
            .arg("--json")
            .assert()
            .success()
            .stdout(predicate::str::contains("\"inserted\":"));
    }
    bin()
        .args(["search", "MCP search tools", "--db"])
        .arg(&db)
        .args(["--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"agent\": \"amp\""));
    bin()
        .args([
            "import",
            "--agent",
            "codex",
            "--source",
            "tests/fixtures/codex-session.jsonl",
            "--db",
        ])
        .arg(&db)
        .arg("--json")
        .assert()
        .success()
        .stdout(
            predicate::str::contains("\"inserted\": 0")
                .and(predicate::str::contains("\"deduplicated\": 3")),
        );
    bin()
        .args(["search", "sqlx", "--db"])
        .arg(&db)
        .args(["--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Use sqlx"));
    bin()
        .args(["search", "OpenDAL", "--db"])
        .arg(&db)
        .args(["--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("claude-code"));
    bin()
        .args(["search", "durable spool", "--db"])
        .arg(&db)
        .args(["--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("opencode"));
    bin()
        .args([
            "import",
            "--agent",
            "codex",
            "--source",
            "tests/fixtures/codex-app-server.json",
            "--db",
        ])
        .arg(&db)
        .arg("--json")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"inserted\": 3"));
    bin()
        .args(["search", "canonical store trait", "--db"])
        .arg(&db)
        .assert()
        .success()
        .stdout(predicate::str::contains("Use sqlx"));
}

#[test]
fn search_accepts_time_window_filters() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("time-window.db");
    bin()
        .args([
            "import",
            "--agent",
            "codex",
            "--source",
            "tests/fixtures/codex-rollout-complete.jsonl",
            "--db",
        ])
        .arg(&db)
        .assert()
        .success();

    let after_window = bin()
        .args([
            "search",
            "parser",
            "--mode",
            "recent",
            "--since",
            "2026-07-18T02:01:00Z",
            "--db",
        ])
        .arg(&db)
        .arg("--json")
        .output()
        .unwrap();
    assert!(after_window.status.success());
    let after_window: Value = serde_json::from_slice(&after_window.stdout).unwrap();
    assert_eq!(after_window["count"], 3);
    assert_eq!(
        after_window["since_ms"],
        serde_json::json!(1_784_340_060_000_i64)
    );
    assert!(after_window["results"].to_string().contains("Compacted"));
    assert!(!after_window["results"].to_string().contains("read_file"));

    let before_window = bin()
        .args([
            "search",
            "parser",
            "--mode",
            "lexical",
            "--until",
            "1784340003000",
            "--db",
        ])
        .arg(&db)
        .arg("--json")
        .output()
        .unwrap();
    assert!(before_window.status.success());
    let before_window: Value = serde_json::from_slice(&before_window.stdout).unwrap();
    assert_eq!(before_window["count"], 1);
    assert_eq!(
        before_window["until_ms"],
        serde_json::json!(1_784_340_003_000_i64)
    );
    assert!(
        before_window["results"]
            .to_string()
            .contains("Find the parser decision")
    );
    assert!(!before_window["results"].to_string().contains("read_file"));
}

#[test]
fn imports_realistic_complete_histories_and_preserves_native_outcomes() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("memory.db");
    let blob_dir = temp.path().join("blobs");
    for (agent, fixture, session_id, expected_kinds, marker) in [
        (
            "amp",
            "tests/fixtures/amp-thread-complete.json",
            "T-amp-complete-session",
            vec![
                "message",
                "tool_call",
                "tool_result",
                "message",
                "tool_call",
                "tool_result",
                "message",
            ],
            "command exited with status 1",
        ),
        (
            "codex",
            "tests/fixtures/codex-rollout-complete.jsonl",
            "codex-complete-session",
            vec![
                "session_started",
                "unknown",
                "message",
                "unknown",
                "tool_call",
                "tool_result",
                "tool_call",
                "tool_result",
                "message",
                "unknown",
                "message",
                "compaction",
                "compaction",
            ],
            "*** Begin Patch",
        ),
        (
            "claude-code",
            "tests/fixtures/claude-session-complete.jsonl",
            "claude-complete-session",
            vec![
                "compaction",
                "unknown",
                "message",
                "tool_call",
                "tool_result",
                "message",
                "tool_call",
                "tool_result",
                "compaction",
                "message",
            ],
            "Fixture session about robust history parsing",
        ),
        (
            "claude-code",
            "tests/fixtures/claude-transcript-complete.jsonl",
            "claude-transcript-complete",
            vec!["message", "tool_call", "tool_result", "message"],
            "fallback session ID",
        ),
        (
            "opencode",
            "tests/fixtures/opencode-session-complete.json",
            "opencode-complete-session",
            vec![
                "message",
                "message",
                "tool_call",
                "tool_result",
                "tool_call",
                "tool_result",
                "compaction",
            ],
            "command exited with status 1",
        ),
    ] {
        let output = bin()
            .env("SEMAFUSE_BLOB_DIR", &blob_dir)
            .args(["import", "--agent", agent, "--source", fixture, "--db"])
            .arg(&db)
            .arg("--json")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["errors"], serde_json::json!([]), "{report:#}");

        let session = session_json(&db, session_id);
        let mut kinds = session["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(event_type)
            .collect::<Vec<_>>();
        let mut expected_kinds = expected_kinds;
        kinds.sort_unstable();
        expected_kinds.sort_unstable();
        assert_eq!(kinds, expected_kinds, "{session:#}");
        assert!(
            session["events"]
                .as_array()
                .unwrap()
                .iter()
                .any(|event| event.to_string().contains(marker)),
            "missing {marker:?} in {session:#}"
        );
    }

    let opencode = session_json(&db, "opencode-complete-session");
    assert!(
        opencode["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|event| { !event_attachments(event).is_empty() })
    );
}

#[test]
fn amp_plugin_events_record_and_recall() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("memory.db");
    bin()
        .args([
            "import",
            "--agent",
            "codex",
            "--source",
            "tests/fixtures/codex-session.jsonl",
            "--db",
        ])
        .arg(&db)
        .assert()
        .success();

    let start = serde_json::json!({
        "thread":{"id":"T-amp-live"}, "id":7, "message":"Which database layer?",
        "cwd":"/workspace/demo"
    });
    bin()
        .args(["hook", "--agent", "amp", "--event", "agent.start", "--db"])
        .arg(&db)
        .args(["--recall", "--protocol"])
        .write_stdin(start.to_string())
        .assert()
        .success()
        .stdout(predicate::str::contains("Use sqlx"));

    let end = serde_json::json!({
        "thread":{"id":"T-amp-live"}, "id":7, "status":"done",
        "cwd":"/workspace/demo", "messages":[
            {"role":"user","id":7,"content":[{"type":"text","text":"Which database layer?"}]},
            {"role":"assistant","id":8,"content":[{"type":"text","text":"Reuse the recalled sqlx decision."}]}
        ]
    });
    bin()
        .args(["hook", "--agent", "amp", "--event", "agent.end", "--db"])
        .arg(&db)
        .write_stdin(end.to_string())
        .assert()
        .success();
    bin()
        .args(["session", "T-amp-live", "--db"])
        .arg(&db)
        .assert()
        .success()
        .stdout(predicate::str::contains("Reuse the recalled sqlx decision"));
}

#[test]
fn hook_recalls_cross_agent_history_and_stores_image_blob() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("memory.db");
    let blob_dir = temp.path().join("blobs");
    bin()
        .env("SEMAFUSE_BLOB_DIR", &blob_dir)
        .args([
            "import",
            "--agent",
            "codex",
            "--source",
            "tests/fixtures/codex-session.jsonl",
            "--db",
        ])
        .arg(&db)
        .assert()
        .success();
    let hook = serde_json::json!({
        "session_id":"claude-live-1","cwd":"/workspace/demo","hook_event_name":"UserPromptSubmit",
        "prompt":"Which database layer should we use?",
        "content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}]
    });
    let output = bin()
        .env("SEMAFUSE_BLOB_DIR", &blob_dir)
        .args(["hook", "--agent", "claude-code", "--event", "auto", "--db"])
        .arg(&db)
        .arg("--recall")
        .arg("--json")
        .write_stdin(hook.to_string())
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("Use sqlx"), "{text}");
    assert!(!text.contains("claude-live-1"), "{text}");
    bin()
        .args(["session", "claude-live-1", "--db"])
        .arg(&db)
        .arg("--json")
        .assert()
        .success()
        .stdout(predicate::str::contains("image/png"));
    assert!(
        fs::read_dir(blob_dir.join("sha256"))
            .unwrap()
            .next()
            .is_some()
    );
}

#[test]
fn hooks_preserve_native_fields_and_normalize_repository_and_tool_events() {
    let temp = TempDir::new().unwrap();
    let repo = temp.path().join("repo");
    let cwd = repo.join("src/nested");
    fs::create_dir_all(repo.join(".git")).unwrap();
    fs::create_dir_all(&cwd).unwrap();
    let db = temp.path().join("normalized.db");
    let blob_dir = temp.path().join("blobs");

    let call = serde_json::json!({
        "session_id":"claude-normalized", "turn_id":"turn-1",
        "tool_use_id":"tool-1", "cwd":cwd, "hook_event_name":"PreToolUse",
        "timestamp":"2025-01-01T00:00:00Z", "tool_name":"Bash",
        "tool_input":{"command":"cargo test"},
        "content":[
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}},
            {"type":"image","mimeType":"image/png","data":"iVBORw0KGgo="},
            {"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo="}
        ]
    });
    bin()
        .env("SEMAFUSE_BLOB_DIR", &blob_dir)
        .args(["hook", "--agent", "claude-code", "--event", "auto", "--db"])
        .arg(&db)
        .write_stdin(call.to_string())
        .assert()
        .success();

    let failure = serde_json::json!({
        "session_id":"claude-normalized", "turn_id":"turn-1",
        "tool_use_id":"tool-1", "cwd":cwd, "hook_event_name":"PostToolUseFailure",
        "timestamp":1735689601, "tool_name":"Bash", "error":"tests failed"
    });
    bin()
        .env("SEMAFUSE_BLOB_DIR", &blob_dir)
        .args(["hook", "--agent", "claude-code", "--event", "auto", "--db"])
        .arg(&db)
        .write_stdin(failure.to_string())
        .assert()
        .success();

    let session = session_json(&db, "claude-normalized");
    let events = session["events"].as_array().unwrap();
    assert_eq!(events.len(), 2);
    let call = events
        .iter()
        .find(|event| event_type(event) == "tool_call")
        .unwrap();
    assert_eq!(call["data"]["tool"], "Bash");
    assert_eq!(call["source"]["native_id"], "tool-1");
    assert_eq!(call["occurred_at_ms"], 1_735_689_600_000_i64);
    assert_eq!(call["source"]["cwd"], cwd.to_string_lossy().as_ref());
    assert_eq!(
        call["source"]["repository"],
        repo.canonicalize().unwrap().to_string_lossy().as_ref()
    );
    assert_eq!(call["source"]["native_event"], "PreToolUse");
    assert_eq!(event_attachments(call).len(), 1);
    let raw = raw_for_event(&db, call["id"].as_str().unwrap());
    assert_eq!(
        raw["records"][0]["payload"]["content"][0]["source"]["data"],
        "iVBORw0KGgo="
    );
    assert_eq!(
        raw["records"][0]["payload"]["content"][1]["data"],
        "iVBORw0KGgo="
    );
    assert_eq!(
        raw["records"][0]["payload"]["content"][2]["image_url"],
        "data:image/png;base64,iVBORw0KGgo="
    );
    let error = events
        .iter()
        .find(|event| event_type(event) == "tool_result" && event["data"]["status"] == "failed")
        .unwrap();
    assert_eq!(error["data"]["tool"], "Bash");
    assert_eq!(error["data"]["error"]["message"], "tests failed");
    assert_eq!(error["occurred_at_ms"], 1_735_689_601_000_i64);
}

#[test]
fn hook_attachments_only_use_adapter_trusted_sources() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("hook-attachments.db");
    let blob_dir = temp.path().join("blobs");
    let stray = temp.path().join("stray-secret.txt");
    fs::write(&stray, b"do not ingest from arbitrary nested payload").unwrap();

    let payload = serde_json::json!({
        "session_id":"trusted-attachments",
        "prompt_id":"prompt-1",
        "prompt":"inspect this image",
        "content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}],
        "untrusted_nested":{"type":"file","source":{"path":stray,"media_type":"text/plain"}}
    });
    bin()
        .env("SEMAFUSE_BLOB_DIR", &blob_dir)
        .args([
            "hook",
            "--agent",
            "claude-code",
            "--event",
            "UserPromptSubmit",
            "--db",
        ])
        .arg(&db)
        .write_stdin(payload.to_string())
        .assert()
        .success();

    let session = session_json(&db, "trusted-attachments");
    let attachments = event_attachments(&session["events"][0]);
    assert_eq!(attachments.len(), 1);
    assert_eq!(attachments[0]["media_type"], "image/png");
    assert_ne!(attachments[0]["name"], "stray-secret.txt");
}

#[test]
fn opencode_history_normalizes_structured_tool_parts() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("opencode-tool.json");
    let db = temp.path().join("opencode.db");
    fs::write(
        &source,
        serde_json::json!({
            "info":{"id":"open-tools","directory":"/workspace/tools"},
            "messages":[{
                "info":{"id":"msg-1","sessionID":"open-tools","role":"assistant","time":{"created":1750000000000_i64}},
                "parts":[{
                    "id":"part-1","messageID":"msg-1","sessionID":"open-tools",
                    "type":"tool","callID":"call-1","tool":"bash",
                    "state":{"status":"completed","input":{"command":"cargo test"},"output":"all tests passed","time":{"start":1750000001000_i64,"end":1750000002000_i64}}
                }]
            }]
        })
        .to_string(),
    )
    .unwrap();

    bin()
        .args(["import", "--agent", "opencode", "--source"])
        .arg(&source)
        .arg("--db")
        .arg(&db)
        .assert()
        .success();

    let session = session_json(&db, "open-tools");
    let events = session["events"].as_array().unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(event_type(&events[0]), "tool_call");
    assert_eq!(events[0]["data"]["tool"], "bash");
    assert_eq!(events[0]["source"]["native_id"], "call-1");
    assert_eq!(events[0]["occurred_at_ms"], 1_750_000_001_000_i64);
    let result = events
        .iter()
        .find(|event| event_type(event) == "tool_result")
        .unwrap();
    assert_eq!(result["data"]["status"], "succeeded");
    assert_eq!(event_text(result), "all tests passed");
    assert_eq!(result["occurred_at_ms"], 1_750_000_002_000_i64);
    assert_eq!(result["source"]["native_event"], "tool");
}

#[test]
fn opencode_part_hooks_use_real_session_and_history_identity() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("opencode-text.json");
    let db = temp.path().join("opencode-text.db");
    fs::write(
        &source,
        serde_json::json!({
            "info":{"id":"open-text","directory":"/workspace/text"},
            "messages":[{
                "info":{"id":"msg-1","sessionID":"open-text","role":"user"},
                "parts":[{"id":"part-1","messageID":"msg-1","sessionID":"open-text","type":"text","text":"same prompt"}]
            }]
        })
        .to_string(),
    )
    .unwrap();
    bin()
        .args(["import", "--agent", "opencode", "--source"])
        .arg(&source)
        .arg("--db")
        .arg(&db)
        .assert()
        .success();

    let payload = serde_json::json!({
        "directory":"/workspace/text", "worktree":"/workspace/text",
        "event":{"type":"message.part.updated","properties":{
            "messageRole":"user",
            "part":{"id":"part-1","messageID":"msg-1","sessionID":"open-text","type":"text","text":"same prompt"}
        }}
    });
    bin()
        .args(["hook", "--agent", "opencode", "--event", "auto", "--db"])
        .arg(&db)
        .write_stdin(payload.to_string())
        .assert()
        .success();

    let session = session_json(&db, "open-text");
    assert_eq!(session["count"], 1);
    assert_eq!(session["events"][0]["source"]["native_id"], "msg-1");
    assert_eq!(event_type(&session["events"][0]), "message");
    assert_eq!(session["events"][0]["data"]["role"], "user");
}

#[test]
fn phase_and_prompt_ids_prevent_lifecycle_event_collisions() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("lifecycle.db");
    for event in ["PreCompact", "PostCompact"] {
        bin()
            .args(["hook", "--agent", "codex", "--event", event, "--db"])
            .arg(&db)
            .write_stdin(
                serde_json::json!({
                    "session_id":"codex-compact", "turn_id":"turn-1", "trigger":"auto"
                })
                .to_string(),
            )
            .assert()
            .success();
    }
    let compact = session_json(&db, "codex-compact");
    assert_eq!(compact["count"], 2);
    assert_ne!(
        compact["events"][0]["source"]["native_id"],
        compact["events"][1]["source"]["native_id"]
    );

    for prompt_id in ["prompt-1", "prompt-2", "prompt-1"] {
        bin()
            .args([
                "hook",
                "--agent",
                "claude-code",
                "--event",
                "UserPromptSubmit",
                "--db",
            ])
            .arg(&db)
            .write_stdin(
                serde_json::json!({
                    "session_id":"claude-prompts", "prompt_id":prompt_id,
                    "prompt":"repeat this"
                })
                .to_string(),
            )
            .assert()
            .success();
    }
    assert_eq!(session_json(&db, "claude-prompts")["count"], 2);
}

#[test]
fn unknown_history_items_are_preserved_as_other_events() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("codex-unknown.json");
    let db = temp.path().join("unknown.db");
    fs::write(
        &source,
        serde_json::json!({
            "thread":{
                "id":"codex-unknown-history", "cwd":"/workspace/unknown",
                "turns":[{"id":"turn-1","items":[{
                    "id":"item-new", "type":"futureArtifact", "detail":"preserve me",
                    "time":{"created":1750000000000_i64}
                }]}]
            }
        })
        .to_string(),
    )
    .unwrap();

    bin()
        .args(["import", "--agent", "codex", "--source"])
        .arg(&source)
        .arg("--db")
        .arg(&db)
        .assert()
        .success();

    let session = session_json(&db, "codex-unknown-history");
    assert_eq!(session["count"], 1);
    assert_eq!(event_type(&session["events"][0]), "unknown");
    assert_eq!(session["events"][0]["source"]["native_id"], "item-new");
    assert_eq!(
        session["events"][0]["occurred_at_ms"],
        1_750_000_000_000_i64
    );
    assert_eq!(
        session["events"][0]["source"]["native_event"],
        "futureArtifact"
    );
    let raw = raw_for_event(&db, session["events"][0]["id"].as_str().unwrap());
    assert_eq!(
        raw["records"][0]["payload"]["thread"]["turns"][0]["items"][0]["detail"],
        "preserve me"
    );
}

#[test]
fn amp_import_and_hooks_share_stable_message_identity() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("identity.db");
    bin()
        .args([
            "import",
            "--agent",
            "amp",
            "--source",
            "tests/fixtures/amp-thread.json",
            "--db",
        ])
        .arg(&db)
        .assert()
        .success();

    for (event, payload) in [
        (
            "agent.start",
            serde_json::json!({
                "thread":{"id":"T-amp-session-1"}, "id":0,
                "message":"How should agent memory be queried?", "cwd":"/workspace/demo"
            }),
        ),
        (
            "agent.end",
            serde_json::json!({
                "thread":{"id":"T-amp-session-1"}, "id":0, "status":"done",
                "cwd":"/workspace/demo", "messages":[
                    {"role":"assistant","messageId":1,"content":[{"type":"text","text":"Use Semafuse through its MCP search tools."}]}
                ]
            }),
        ),
    ] {
        bin()
            .args(["hook", "--agent", "amp", "--event", event, "--db"])
            .arg(&db)
            .write_stdin(payload.to_string())
            .assert()
            .success();
    }
    assert_eq!(
        session_json(&db, "T-amp-session-1")["count"],
        4,
        "hook messages should deduplicate against imported messages"
    );
}

#[test]
fn spool_replay_and_mcp_protocol_work_end_to_end() {
    let temp = TempDir::new().unwrap();
    let invalid_db = temp.path().join("db-directory");
    fs::create_dir(&invalid_db).unwrap();
    let spool = temp.path().join("spool");
    let db = temp.path().join("valid.db");
    let blob_dir = temp.path().join("blobs");
    let payload = r#"{"session_id":"codex-live-1","turn_id":"turn-1","cwd":"/workspace/demo","hook_event_name":"Stop","last_assistant_message":"Recovered from the offline spool.","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}]}"#;
    bin()
        .args(["hook", "--agent", "codex", "--event", "auto", "--db"])
        .arg(&invalid_db)
        .arg("--spool-dir")
        .arg(&spool)
        .write_stdin(payload)
        .assert()
        .success();
    bin()
        .env("SEMAFUSE_BLOB_DIR", &blob_dir)
        .args(["spool", "replay", "--db"])
        .arg(&db)
        .arg("--spool-dir")
        .arg(&spool)
        .arg("--json")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"replayed\": 1"));
    let session = bin()
        .args(["session", "codex-live-1", "--db"])
        .arg(&db)
        .arg("--json")
        .output()
        .unwrap();
    let session: Value = serde_json::from_slice(&session.stdout).unwrap();
    assert_eq!(event_type(&session["events"][0]), "message");
    assert_eq!(session["events"][0]["data"]["role"], "assistant");
    assert_eq!(
        event_text(&session["events"][0]),
        "Recovered from the offline spool."
    );
    assert_eq!(
        event_attachments(&session["events"][0])[0]["media_type"],
        "image/png"
    );
    bin()
        .args(["search", "offline spool", "--db"])
        .arg(&db)
        .assert()
        .success()
        .stdout(predicate::str::contains("Recovered"));

    let executable = assert_cmd::cargo::cargo_bin("semafuse");
    let mut child = StdCommand::new(executable)
        .args(["mcp", "--db"])
        .arg(&db)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let input = concat!(
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n",
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{}}\n",
        "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"semafuse_search\",\"arguments\":{\"query\":\"offline spool\"}}}\n"
    );
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    drop(child.stdin.take());
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let lines: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 3);
    assert!(
        lines[1]["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "semafuse_search")
    );
    assert!(
        lines[1]["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "semafuse_recall")
    );
    assert!(
        lines[2]
            .to_string()
            .contains("Recovered from the offline spool")
    );
}

#[test]
fn generated_integrations_have_agent_native_locations() {
    let temp = TempDir::new().unwrap();
    for agent in ["amp", "codex", "claude-code", "opencode"] {
        bin()
            .args(["hooks", "install", "--agent", agent, "--target"])
            .arg(temp.path())
            .arg("--binary")
            .arg("/usr/local/bin/semafuse")
            .arg("--db")
            .arg(temp.path().join("db.sqlite"))
            .assert()
            .success();
    }
    assert!(temp.path().join(".codex/hooks.json").is_file());
    assert!(temp.path().join(".amp/settings.json").is_file());
    assert!(temp.path().join(".amp/plugins/semafuse.ts").is_file());
    assert!(temp.path().join(".claude/settings.json").is_file());
    assert!(temp.path().join(".opencode/plugins/semafuse.js").is_file());
    assert!(temp.path().join(".mcp.json").is_file());
    assert!(temp.path().join("opencode.json").is_file());
    let amp_settings: Value =
        serde_json::from_str(&fs::read_to_string(temp.path().join(".amp/settings.json")).unwrap())
            .unwrap();
    assert_eq!(
        amp_settings["amp.mcpServers"]["semafuse"]["command"],
        "/usr/local/bin/semafuse"
    );
    assert!(
        fs::read_to_string(temp.path().join(".amp/plugins/semafuse.ts"))
            .unwrap()
            .contains("agent.start")
    );
    assert!(
        fs::read_to_string(temp.path().join(".codex/hooks.json"))
            .unwrap()
            .contains("--protocol")
    );
    assert!(
        fs::read_to_string(temp.path().join(".codex/hooks.json"))
            .unwrap()
            .contains("PreToolUse")
    );
    let codex_hooks: Value =
        serde_json::from_str(&fs::read_to_string(temp.path().join(".codex/hooks.json")).unwrap())
            .unwrap();
    assert!(codex_hooks["hooks"].get("SessionEnd").is_none());
    assert!(codex_hooks["hooks"].get("StopFailure").is_none());
    for (name, entries) in codex_hooks["hooks"].as_object().unwrap() {
        let handler = &entries[0]["hooks"][0];
        let command = handler["command"].as_str().unwrap();
        assert!(command.contains("--protocol"));
        assert_eq!(
            command.contains("--recall"),
            matches!(name.as_str(), "SessionStart" | "UserPromptSubmit")
        );
        assert!(handler.get("args").is_none());
    }
    assert!(
        fs::read_to_string(temp.path().join(".opencode/plugins/semafuse.js"))
            .unwrap()
            .contains("pendingTextParts")
    );
    assert!(
        fs::read_to_string(temp.path().join(".claude/settings.json"))
            .unwrap()
            .contains("--protocol")
    );
    let claude_settings: Value = serde_json::from_str(
        &fs::read_to_string(temp.path().join(".claude/settings.json")).unwrap(),
    )
    .unwrap();
    assert!(claude_settings["hooks"]["SessionStart"].is_array());
    assert!(claude_settings["hooks"]["PreToolUse"].is_array());
    assert!(claude_settings["hooks"]["PreCompact"].is_array());
    assert!(claude_settings["hooks"]["PostCompact"].is_array());
    for (name, entries) in claude_settings["hooks"].as_object().unwrap() {
        let args = entries[0]["hooks"][0]["args"].as_array().unwrap();
        assert_eq!(
            args.iter().any(|arg| arg == "--recall"),
            matches!(name.as_str(), "SessionStart" | "UserPromptSubmit")
        );
    }

    bin()
        .args(["hooks", "install", "--agent", "codex", "--target"])
        .arg(temp.path())
        .arg("--binary")
        .arg("/usr/local/bin/semafuse")
        .assert()
        .failure()
        .stderr(predicate::str::contains("refusing to overwrite"));
}

#[test]
fn config_file_and_environment_are_layered() {
    let temp = TempDir::new().unwrap();
    let config = temp.path().join("semafuse.toml");
    fs::write(
        &config,
        "[strategies]\nenabled = [\"history\"]\nrecall = \"history\"\n",
    )
    .unwrap();

    bin()
        .args(["strategy", "list", "--db"])
        .arg(temp.path().join("config.db"))
        .arg("--json")
        .env("SEMAFUSE_CONFIG", &config)
        .assert()
        .success()
        .stdout(predicate::str::contains("\"name\": \"history\""));

    bin()
        .args(["strategy", "list", "--db"])
        .arg(temp.path().join("invalid.db"))
        .arg("--json")
        .env("SEMAFUSE_CONFIG", &config)
        .env("SEMAFUSE__STRATEGIES__RECALL", "missing")
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "recall strategy missing is not enabled",
        ));
}

#[test]
fn commands_default_to_human_output_and_offer_parseable_json() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("output-contract.db");

    bin()
        .args(["init", "--db"])
        .arg(&db)
        .assert()
        .success()
        .stdout(predicate::str::contains("Initialized database at"));
    let initialized = bin()
        .args(["--json", "init", "--db"])
        .arg(&db)
        .output()
        .unwrap();
    let initialized: Value = serde_json::from_slice(&initialized.stdout).unwrap();
    assert_eq!(initialized["ok"], true);
    assert_eq!(initialized["command"], "init");

    let import = bin()
        .args([
            "import",
            "--agent",
            "codex",
            "--source",
            "tests/fixtures/codex-session.jsonl",
            "--db",
        ])
        .arg(&db)
        .arg("--json")
        .output()
        .unwrap();
    let import: Value = serde_json::from_slice(&import.stdout).unwrap();
    assert_eq!(import["ok"], true);
    assert_eq!(import["command"], "import");
    assert_eq!(import["inserted"], 3);

    for args in [
        vec!["search", "sqlx", "--db"],
        vec!["session", "codex-session-1", "--db"],
        vec!["doctor", "--db"],
    ] {
        let output = bin().args(args).arg(&db).arg("--json").output().unwrap();
        assert!(output.status.success());
        assert_eq!(
            serde_json::from_slice::<Value>(&output.stdout).unwrap()["ok"],
            true
        );
    }

    let hook = bin()
        .args(["hook", "--agent", "claude-code", "--event", "Stop", "--db"])
        .arg(&db)
        .arg("--json")
        .write_stdin(
            serde_json::json!({
                "session_id":"json-hook", "last_assistant_message":"done"
            })
            .to_string(),
        )
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(
        serde_json::from_slice::<Value>(&hook).unwrap()["recorded"],
        true
    );

    let printed = bin()
        .args(["hooks", "print", "--agent", "codex", "--json"])
        .output()
        .unwrap();
    assert!(printed.status.success());
    assert!(serde_json::from_slice::<Value>(&printed.stdout).unwrap()["artifacts"].is_array());

    let install_target = temp.path().join("integrations");
    let installed = bin()
        .args(["hooks", "install", "--agent", "opencode", "--target"])
        .arg(&install_target)
        .arg("--json")
        .output()
        .unwrap();
    assert!(installed.status.success());
    assert!(serde_json::from_slice::<Value>(&installed.stdout).unwrap()["installed"].is_array());

    let replayed = bin()
        .args(["spool", "replay", "--db"])
        .arg(&db)
        .arg("--spool-dir")
        .arg(temp.path().join("empty-spool"))
        .arg("--json")
        .output()
        .unwrap();
    assert!(replayed.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&replayed.stdout).unwrap()["replayed"],
        0
    );

    let strategies = bin()
        .args(["strategy", "status", "--db"])
        .arg(&db)
        .arg("--json")
        .output()
        .unwrap();
    assert!(strategies.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&strategies.stdout).unwrap()["strategies"][0]["name"],
        "history"
    );

    let error = bin()
        .args(["search", "anything", "--mode", "not-a-mode", "--db"])
        .arg(&db)
        .arg("--json")
        .output()
        .unwrap();
    assert!(!error.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&error.stderr).unwrap()["ok"],
        false
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&error.stderr).unwrap()["command"],
        "search"
    );

    assert!(
        fs::read_to_string(install_target.join(".opencode/plugins/semafuse.js"))
            .unwrap()
            .contains("--protocol")
    );
}

#[test]
fn history_attachments_stay_with_their_source_event_without_duplicates() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("claude-images.jsonl");
    let db = temp.path().join("images.db");
    let blob_dir = temp.path().join("blobs");
    fs::write(
        &source,
        concat!(
            "{\"uuid\":\"with-image\",\"sessionId\":\"image-history\",\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"inspect this image\"},{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"aGVsbG8=\"}}]}}\n",
            "{\"uuid\":\"without-image\",\"sessionId\":\"image-history\",\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":\"done\"}}\n"
        ),
    )
    .unwrap();

    bin()
        .env("SEMAFUSE_BLOB_DIR", &blob_dir)
        .args(["import", "--agent", "claude-code", "--source"])
        .arg(&source)
        .arg("--db")
        .arg(&db)
        .assert()
        .success();

    let session = session_json(&db, "image-history");
    let events = session["events"].as_array().unwrap();
    let with_image = events
        .iter()
        .find(|event| event["source"]["native_id"] == "with-image")
        .unwrap();
    let without_image = events
        .iter()
        .find(|event| event["source"]["native_id"] == "without-image")
        .unwrap();
    assert_eq!(event_attachments(with_image).len(), 1);
    assert!(event_attachments(without_image).is_empty());
}

#[test]
fn tool_attachments_are_isolated_from_message_and_result_events() {
    let temp = TempDir::new().unwrap();
    let claude_source = temp.path().join("claude-tool.jsonl");
    let opencode_source = temp.path().join("opencode-tool-images.json");
    let db = temp.path().join("tool-images.db");
    let blob_dir = temp.path().join("blobs");
    fs::write(
        &claude_source,
        "{\"uuid\":\"claude-message\",\"sessionId\":\"claude-tool-images\",\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"tool returned an image\"},{\"type\":\"tool_result\",\"tool_use_id\":\"tool-image\",\"content\":[{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"aGVsbG8=\"}}]}]}}\n",
    )
    .unwrap();
    fs::write(
        &opencode_source,
        serde_json::json!({
            "info":{"id":"opencode-tool-images"},
            "messages":[{"info":{"id":"message-1","sessionID":"opencode-tool-images","role":"assistant"},"parts":[{
                "type":"tool", "callID":"call-1", "tool":"vision", "state":{
                    "status":"completed",
                    "input":{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW5wdXQ="}},
                    "output":{"type":"image","source":{"type":"base64","media_type":"image/png","data":"b3V0cHV0"}}
                }
            }]}]
        })
        .to_string(),
    )
    .unwrap();

    for (agent, source) in [
        ("claude-code", &claude_source),
        ("opencode", &opencode_source),
    ] {
        bin()
            .env("SEMAFUSE_BLOB_DIR", &blob_dir)
            .args(["import", "--agent", agent, "--source"])
            .arg(source)
            .arg("--db")
            .arg(&db)
            .assert()
            .success();
    }

    let claude = session_json(&db, "claude-tool-images");
    let claude_events = claude["events"].as_array().unwrap();
    let message = claude_events
        .iter()
        .find(|event| event_type(event) == "message" && event["data"]["role"] == "user")
        .unwrap();
    let result = claude_events
        .iter()
        .find(|event| event_type(event) == "tool_result")
        .unwrap();
    assert!(event_attachments(message).is_empty());
    assert_eq!(event_attachments(result).len(), 1);

    let opencode = session_json(&db, "opencode-tool-images");
    let opencode_events = opencode["events"].as_array().unwrap();
    let call = opencode_events
        .iter()
        .find(|event| event_type(event) == "tool_call")
        .unwrap();
    let result = opencode_events
        .iter()
        .find(|event| event_type(event) == "tool_result")
        .unwrap();
    assert_eq!(event_attachments(call).len(), 1);
    assert_eq!(event_attachments(result).len(), 1);
    assert_ne!(
        event_attachments(call)[0]["digest"],
        event_attachments(result)[0]["digest"]
    );
}

#[test]
fn codex_image_only_messages_preserve_local_and_inline_images() {
    let temp = TempDir::new().unwrap();
    let image = temp.path().join("screen.png");
    let source = temp.path().join("codex-images.json");
    let db = temp.path().join("codex-images.db");
    fs::write(&image, b"local-image").unwrap();
    fs::write(
        &source,
        serde_json::json!({"thread":{
            "id":"codex-image-only", "turns":[{"id":"turn-1","items":[{
                "id":"message-1", "type":"userMessage", "content":[
                    {"type":"localImage","path":image},
                    {"type":"image","url":"data:image/png;base64,aW5saW5l"}
                ]
            }]}]
        }})
        .to_string(),
    )
    .unwrap();

    bin()
        .env("SEMAFUSE_BLOB_DIR", temp.path().join("blobs"))
        .args(["import", "--agent", "codex", "--source"])
        .arg(&source)
        .arg("--db")
        .arg(&db)
        .assert()
        .success();
    let session = session_json(&db, "codex-image-only");
    assert_eq!(session["count"], 1);
    assert_eq!(event_attachments(&session["events"][0]).len(), 2);
}

#[test]
fn dry_run_preserves_image_only_event_counts_without_writing_blobs() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("codex-images-dry-run.json");
    let blob_dir = temp.path().join("dry-run-blobs");
    fs::write(
        &source,
        serde_json::json!({"thread":{
            "id":"codex-image-dry-run", "turns":[{"id":"turn-1","items":[{
                "id":"message-1", "type":"userMessage", "content":[
                    {"type":"image","url":"data:image/png;base64,aW5saW5l"}
                ]
            }]}]
        }})
        .to_string(),
    )
    .unwrap();

    let output = bin()
        .env("SEMAFUSE_BLOB_DIR", &blob_dir)
        .args(["import", "--agent", "codex", "--source"])
        .arg(&source)
        .arg("--dry-run")
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success());
    let output: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(output["events"], 1);
    assert_eq!(output["diagnostics"], 0);
    assert!(!blob_dir.exists());
}

#[test]
fn blob_initialization_failure_spools_and_replay_keeps_the_payload() {
    let temp = TempDir::new().unwrap();
    let config = temp.path().join("invalid.toml");
    let spool = temp.path().join("spool");
    let db = temp.path().join("memory.db");
    fs::write(&config, "[blob]\nscheme = \"not-a-real-opendal-service\"\n").unwrap();
    let payload = serde_json::json!({
        "session_id":"blob-failure", "prompt_id":"prompt-1", "prompt":"keep this",
        "content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}}]
    });

    let output = bin()
        .env("SEMAFUSE_CONFIG", &config)
        .args([
            "hook",
            "--agent",
            "claude-code",
            "--event",
            "UserPromptSubmit",
            "--db",
        ])
        .arg(&db)
        .arg("--spool-dir")
        .arg(&spool)
        .arg("--json")
        .write_stdin(payload.to_string())
        .output()
        .unwrap();
    assert!(output.status.success());
    let output: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(output["recorded"], false);
    assert_eq!(output["spooled"], true);

    bin()
        .env("SEMAFUSE_CONFIG", &config)
        .args(["spool", "replay", "--db"])
        .arg(&db)
        .arg("--spool-dir")
        .arg(&spool)
        .assert()
        .failure();
    assert!(spool.join("hooks.jsonl").is_file());

    bin()
        .env_remove("SEMAFUSE_CONFIG")
        .env("SEMAFUSE_BLOB_DIR", temp.path().join("blobs"))
        .args(["spool", "replay", "--db"])
        .arg(&db)
        .arg("--spool-dir")
        .arg(&spool)
        .assert()
        .success();
    assert_eq!(session_json(&db, "blob-failure")["count"], 1);
}

#[test]
fn sqlite_database_path_accepts_url_characters() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("history?2026.db");
    bin().args(["init", "--db"]).arg(&db).assert().success();
    assert!(db.is_file());
}
