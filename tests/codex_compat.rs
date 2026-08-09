use assert_cmd::Command;
use serde_json::{Value, json};
use std::fs;
use tempfile::TempDir;

fn bin() -> Command {
    let mut command = Command::cargo_bin("semafuse").unwrap();
    command.env(
        "SEMAFUSE_DATA_DIR",
        std::env::temp_dir().join(format!("semafuse-codex-compat-{}", std::process::id())),
    );
    command
}

#[test]
fn generated_codex_hooks_cover_session_end_and_windows_commands() {
    let output = bin()
        .args([
            "hooks",
            "print",
            "--agent",
            "codex",
            "--binary",
            r"C:\Program Files & Tools\Semafuse\semafuse.exe",
            "--db",
            r"C:\O'Brien Memory\semafuse.db",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output.stderr);
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    let content = result["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["path"] == ".codex/hooks.json")
        .unwrap()["content"]
        .as_str()
        .unwrap();
    let config: Value = serde_json::from_str(content).unwrap();
    let hooks = config["hooks"].as_object().unwrap();

    for event in [
        "SessionStart",
        "SessionEnd",
        "UserPromptSubmit",
        "PermissionRequest",
        "PreToolUse",
        "PostToolUse",
        "SubagentStart",
        "SubagentStop",
        "Stop",
        "PreCompact",
        "PostCompact",
    ] {
        let handler = &hooks[event][0]["hooks"][0];
        let windows = handler["commandWindows"].as_str().unwrap();
        assert!(
            windows.starts_with(r#"& 'C:\Program Files & Tools\Semafuse\semafuse.exe' 'hook'"#),
            "{windows}"
        );
        assert!(
            windows.contains(r#"'--db' 'C:\O''Brien Memory\semafuse.db'"#),
            "{windows}"
        );
        assert_eq!(
            windows.contains("'--recall'"),
            matches!(event, "SessionStart" | "UserPromptSubmit")
        );
    }
    assert_eq!(hooks["SessionEnd"][0]["hooks"][0]["timeout"], 3);
}

#[test]
fn processor_depth_marker_stops_nested_codex_hooks_before_ingest() {
    let temp = TempDir::new().unwrap();
    let database = temp.path().join("must-not-be-created.db");
    let output = bin()
        .args(["hook", "--agent", "codex", "--event", "auto", "--db"])
        .arg(&database)
        .arg("--protocol")
        .env("SEMAFUSE_PROCESSOR_DEPTH", "1")
        .write_stdin("deliberately not JSON")
        .output()
        .unwrap();

    assert!(output.status.success(), "{:?}", output.stderr);
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
    assert!(!database.exists());
}

#[cfg(windows)]
#[test]
fn generated_windows_hook_runs_in_powershell() {
    use std::{io::Write, process::Stdio};

    let temp = TempDir::new().unwrap();
    let db = temp.path().join("PowerShell Memory.db");
    let binary = std::path::Path::new(env!("CARGO_BIN_EXE_semafuse"));
    let output = bin()
        .args(["hooks", "print", "--agent", "codex", "--binary"])
        .arg(binary)
        .arg("--db")
        .arg(&db)
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output.stderr);
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    let content = result["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["path"] == ".codex/hooks.json")
        .unwrap()["content"]
        .as_str()
        .unwrap();
    let config: Value = serde_json::from_str(content).unwrap();
    let command = config["hooks"]["SessionStart"][0]["hooks"][0]["commandWindows"]
        .as_str()
        .unwrap();

    let mut child = std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", command])
        .env("SEMAFUSE_BLOB_DIR", temp.path().join("blobs"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(
            json!({
                "session_id":"powershell-hook", "hook_event_name":"SessionStart",
                "cwd":temp.path()
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn codex_current_lifecycle_hooks_map_without_falling_back_to_other() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("lifecycle.db");
    let payloads = [
        json!({
            "session_id":"codex-lifecycle", "turn_id":"turn-1",
            "hook_event_name":"PermissionRequest", "tool_name":"Bash",
            "tool_input":{"command":"cargo test","description":"run tests"}
        }),
        json!({
            "session_id":"codex-lifecycle", "turn_id":"turn-1",
            "hook_event_name":"SubagentStart", "agent_id":"agent-1",
            "agent_type":"reviewer"
        }),
        json!({
            "session_id":"codex-lifecycle", "turn_id":"turn-1",
            "hook_event_name":"SubagentStop", "agent_id":"agent-1",
            "agent_type":"reviewer", "last_assistant_message":"review complete"
        }),
    ];
    for payload in payloads {
        bin()
            .args(["hook", "--agent", "codex", "--event", "auto", "--db"])
            .arg(&db)
            .env("SEMAFUSE_BLOB_DIR", temp.path().join("blobs"))
            .write_stdin(payload.to_string())
            .assert()
            .success();
    }

    let output = bin()
        .args(["session", "codex-lifecycle", "--db"])
        .arg(&db)
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output.stderr);
    let session: Value = serde_json::from_slice(&output.stdout).unwrap();
    let events = session["events"].as_array().unwrap();
    let by_native_event = |name: &str| {
        events
            .iter()
            .find(|event| event["source"]["native_event"] == name)
            .unwrap()
    };
    let permission = by_native_event("PermissionRequest");
    assert_eq!(permission["data"]["type"], "tool_call");
    assert_eq!(permission["data"]["tool"], "Bash");
    assert_eq!(permission["data"]["arguments"]["command"], "cargo test");
    assert_eq!(
        by_native_event("SubagentStart")["data"]["type"],
        "session_started"
    );
    let subagent_stop = events
        .iter()
        .filter(|event| event["source"]["native_event"] == "SubagentStop")
        .collect::<Vec<_>>();
    assert_eq!(subagent_stop.len(), 2);
    let message = subagent_stop
        .iter()
        .find(|event| event["data"]["type"] == "message")
        .unwrap();
    assert_eq!(message["data"]["role"], "assistant");
    assert_eq!(message["data"]["content"][0]["text"], "review complete");
    let ended = subagent_stop
        .iter()
        .find(|event| event["data"]["type"] == "session_ended")
        .unwrap();
    assert_eq!(ended["data"]["reason"], "reviewer");
}

#[test]
fn codex_post_tool_use_requires_an_explicit_success_signal() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("post-tool-use.db");
    let payloads = [
        json!({
            "session_id":"codex-tool-outcomes", "turn_id":"turn-1",
            "hook_event_name":"PostToolUse", "tool_use_id":"failed-call",
            "tool_name":"Bash", "tool_response":{
                "exitCode":17, "error":"command exited with status 17"
            }
        }),
        json!({
            "session_id":"codex-tool-outcomes", "turn_id":"turn-1",
            "hook_event_name":"PostToolUse", "tool_use_id":"successful-call",
            "tool_name":"Read", "tool_response":{
                "status":"completed", "content":"file contents"
            }
        }),
        json!({
            "session_id":"codex-tool-outcomes", "turn_id":"turn-1",
            "hook_event_name":"PostToolUse", "tool_use_id":"opaque-call",
            "tool_name":"custom", "tool_response":"opaque native output"
        }),
    ];
    for payload in payloads {
        bin()
            .args(["hook", "--agent", "codex", "--event", "auto", "--db"])
            .arg(&db)
            .env("SEMAFUSE_BLOB_DIR", temp.path().join("blobs"))
            .write_stdin(payload.to_string())
            .assert()
            .success();
    }

    let output = bin()
        .args(["session", "codex-tool-outcomes", "--db"])
        .arg(&db)
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output.stderr);
    let session: Value = serde_json::from_slice(&output.stdout).unwrap();
    let events = session["events"].as_array().unwrap();
    let by_id = |id: &str| {
        events
            .iter()
            .find(|event| event["source"]["native_id"] == id)
            .unwrap()
    };
    let failed = by_id("failed-call");
    assert_eq!(failed["data"]["status"], "failed");
    assert_eq!(
        failed["data"]["error"]["message"],
        "command exited with status 17"
    );
    assert_eq!(by_id("successful-call")["data"]["status"], "succeeded");
    assert_eq!(by_id("opaque-call")["data"]["status"], "unknown");
}

#[test]
fn codex_app_server_history_preserves_tool_status_and_errors() {
    let temp = TempDir::new().unwrap();
    let source = temp.path().join("app-server-tools.json");
    let db = temp.path().join("app-server-tools.db");
    fs::write(
        &source,
        json!({
            "id":19,
            "result":{"thread":{
                "id":"codex-app-tool-outcomes", "turns":[{"id":"turn-1","items":[
                    {
                        "type":"commandExecution", "id":"failed-command",
                        "command":"cargo test", "aggregatedOutput":"tests failed",
                        "status":"failed"
                    },
                    {
                        "type":"mcpToolCall", "id":"failed-mcp", "server":"github",
                        "tool":"issues.get", "arguments":{"id":42}, "status":"failed",
                        "error":{"message":"upstream unavailable","code":503}
                    },
                    {
                        "type":"fileChange", "id":"declined-file",
                        "changes":[{"kind":"update","diff":"@@ -1 +1 @@"}],
                        "status":"declined"
                    },
                    {
                        "type":"collabAgentToolCall", "id":"running-collab",
                        "tool":"spawn_agent", "status":"inProgress"
                    },
                    {
                        "type":"dynamicToolCall", "id":"completed-dynamic",
                        "tool":"lookup", "arguments":{"id":"ABC-123"},
                        "status":"completed"
                    }
                ]}]
            }}
        })
        .to_string(),
    )
    .unwrap();

    bin()
        .args(["import", "--agent", "codex", "--source"])
        .arg(&source)
        .arg("--db")
        .arg(&db)
        .env("SEMAFUSE_BLOB_DIR", temp.path().join("blobs"))
        .assert()
        .success();

    let output = bin()
        .args(["session", "codex-app-tool-outcomes", "--db"])
        .arg(&db)
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output.stderr);
    let session: Value = serde_json::from_slice(&output.stdout).unwrap();
    let events = session["events"].as_array().unwrap();
    let by_id = |id: &str| {
        events
            .iter()
            .find(|event| event["source"]["native_id"] == id)
            .unwrap()
    };

    let command = by_id("failed-command");
    assert_eq!(command["data"]["status"], "failed");
    assert_eq!(command["data"]["error"]["message"], "tests failed");
    let mcp = by_id("failed-mcp");
    assert_eq!(mcp["data"]["status"], "failed");
    assert_eq!(mcp["data"]["error"]["message"], "upstream unavailable");
    assert_eq!(mcp["data"]["error"]["code"], "503");
    assert_eq!(by_id("declined-file")["data"]["status"], "rejected");
    assert_eq!(
        by_id("declined-file")["data"]["error"]["message"],
        "tool was rejected"
    );
    assert_eq!(by_id("running-collab")["data"]["status"], "running");
    assert_eq!(by_id("completed-dynamic")["data"]["status"], "succeeded");
}

#[test]
fn codex_session_end_spools_before_the_native_deadline_when_sqlite_is_busy() {
    use sqlx::{Connection, sqlite::SqliteConnectOptions};
    use std::time::{Duration, Instant};

    let temp = TempDir::new().unwrap();
    let db = temp.path().join("busy.db");
    let spool = temp.path().join("spool");
    bin().args(["init", "--db"]).arg(&db).assert().success();

    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let options = SqliteConnectOptions::new().filename(&db);
        let mut connection = sqlx::SqliteConnection::connect_with(&options)
            .await
            .unwrap();
        sqlx::query("BEGIN EXCLUSIVE")
            .execute(&mut connection)
            .await
            .unwrap();

        let started = Instant::now();
        let output = bin()
            .args(["hook", "--agent", "codex", "--event", "auto", "--db"])
            .arg(&db)
            .arg("--spool-dir")
            .arg(&spool)
            .arg("--protocol")
            .env("SEMAFUSE_BLOB_DIR", temp.path().join("blobs"))
            .write_stdin(
                json!({
                    "session_id":"deadline-session", "hook_event_name":"SessionEnd",
                    "reason":"other"
                })
                .to_string(),
            )
            .output()
            .unwrap();
        let elapsed = started.elapsed();
        sqlx::query("ROLLBACK")
            .execute(&mut connection)
            .await
            .unwrap();
        // Windows SQLite locking is byte-range based; close the connection so
        // replay observes the database after the simulated contention ends.
        drop(connection);

        assert!(output.status.success(), "{:?}", output.stderr);
        assert!(
            elapsed < Duration::from_secs(3),
            "SessionEnd fallback took {elapsed:?}"
        );
        let queued = fs::read_dir(&spool)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "jsonl")
            })
            .map(|entry| fs::read_to_string(entry.path()).unwrap())
            .collect::<String>();
        assert!(queued.contains("deadline-session"));
        assert!(queued.contains("SessionEnd"));

        bin()
            .args(["spool", "replay", "--db"])
            .arg(&db)
            .arg("--spool-dir")
            .arg(&spool)
            .env("SEMAFUSE_BLOB_DIR", temp.path().join("blobs"))
            .assert()
            .success();
        let still_queued = fs::read_dir(&spool)
            .unwrap()
            .filter_map(Result::ok)
            .any(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "jsonl")
            });
        assert!(!still_queued);
    });
}

#[test]
fn successful_codex_session_end_clears_its_write_ahead_spool() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("session-end.db");
    let spool = temp.path().join("spool");
    bin()
        .args(["hook", "--agent", "codex", "--event", "auto", "--db"])
        .arg(&db)
        .arg("--spool-dir")
        .arg(&spool)
        .arg("--protocol")
        .env("SEMAFUSE_BLOB_DIR", temp.path().join("blobs"))
        .write_stdin(
            json!({
                "session_id":"completed-session", "hook_event_name":"SessionEnd",
                "reason":"other"
            })
            .to_string(),
        )
        .assert()
        .success();

    let queued = fs::read_dir(&spool)
        .unwrap()
        .filter_map(Result::ok)
        .any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "jsonl")
        });
    assert!(!queued);
    let output = bin()
        .args(["session", "completed-session", "--db"])
        .arg(&db)
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output.stderr);
    let session: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(session["events"][0]["data"]["type"], "session_ended");
}

#[test]
fn codex_compaction_keeps_trigger_without_inventing_a_summary() {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("compaction.db");
    let payload = json!({
        "session_id":"codex-compaction",
        "turn_id":"turn-1",
        "hook_event_name":"PostCompact",
        "trigger":"auto",
        "compact_summary":"not part of the Codex hook protocol",
        "custom_instructions":"also not part of the Codex hook protocol",
    });
    bin()
        .args(["hook", "--agent", "codex", "--event", "auto", "--db"])
        .arg(&db)
        .env("SEMAFUSE_BLOB_DIR", temp.path().join("blobs"))
        .write_stdin(payload.to_string())
        .assert()
        .success();

    let output = bin()
        .args(["session", "codex-compaction", "--db"])
        .arg(&db)
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output.stderr);
    let session: Value = serde_json::from_slice(&output.stdout).unwrap();
    let data = &session["events"][0]["data"];
    assert_eq!(data["type"], "compaction");
    assert!(data["summary"].is_null());
    assert_eq!(data["details"], json!({"trigger":"auto"}));
}

#[cfg(unix)]
#[test]
fn codex_app_server_lists_every_source_kind() {
    use std::os::unix::fs::PermissionsExt;

    let temp = TempDir::new().unwrap();
    let fake_codex = temp.path().join("codex");
    let capture = temp.path().join("requests.jsonl");
    fs::write(
        &fake_codex,
        r#"#!/bin/sh
next_list_id=1
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$SEMAFUSE_CAPTURE"
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"id":0,"result":{}}\n'
      ;;
    *'"method":"thread/list"'*)
      printf '{"id":%s,"result":{"data":[],"nextCursor":null}}\n' "$next_list_id"
      next_list_id=$((next_list_id + 1))
      ;;
  esac
done
"#,
    )
    .unwrap();
    let mut permissions = fs::metadata(&fake_codex).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_codex, permissions).unwrap();

    bin()
        .args(["import", "--agent", "codex", "--dry-run", "--json"])
        .env("PATH", temp.path())
        .env("SEMAFUSE_CAPTURE", &capture)
        .assert()
        .success();

    let requests = fs::read_to_string(capture)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|request| request["method"] == "thread/list")
        .collect::<Vec<_>>();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["params"]["archived"], false);
    assert_eq!(requests[1]["params"]["archived"], true);
    for request in requests {
        assert_eq!(
            request["params"]["sourceKinds"],
            json!([
                "cli",
                "vscode",
                "exec",
                "appServer",
                "subAgent",
                "subAgentReview",
                "subAgentCompact",
                "subAgentThreadSpawn",
                "subAgentOther",
                "unknown",
            ])
        );
    }
}

#[cfg(unix)]
#[test]
fn codex_app_server_reports_and_skips_thread_read_errors() {
    use std::os::unix::fs::PermissionsExt;

    let temp = TempDir::new().unwrap();
    let fake_codex = temp.path().join("codex");
    fs::write(
        &fake_codex,
        r#"#!/bin/sh
next_list_id=1
next_read_id=3
while IFS= read -r line; do
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"id":0,"result":{}}\n'
      ;;
    *'"method":"thread/list"'*)
      if [ "$next_list_id" -eq 1 ]; then
        printf '{"id":1,"result":{"data":[{"id":"missing-thread"},{"id":"good-thread"}],"nextCursor":null}}\n'
      else
        printf '{"id":2,"result":{"data":[],"nextCursor":null}}\n'
      fi
      next_list_id=$((next_list_id + 1))
      ;;
    *'"method":"thread/read"'*)
      if [ "$next_read_id" -eq 3 ]; then
        printf '{"id":3,"error":{"code":-32000,"message":"rollout missing"}}\n'
      else
        printf '{"id":4,"result":{"thread":{"id":"good-thread","turns":[{"id":"turn-1","items":[{"type":"userMessage","id":"message-1","content":[{"type":"text","text":"retained history"}]}]}]}}}\n'
      fi
      next_read_id=$((next_read_id + 1))
      ;;
  esac
done
"#,
    )
    .unwrap();
    let mut permissions = fs::metadata(&fake_codex).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_codex, permissions).unwrap();

    let output = bin()
        .args(["import", "--agent", "codex", "--dry-run", "--json"])
        .env("PATH", temp.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["ok"], false);
    assert_eq!(result["source_errors"], 1);
    assert_eq!(result["files"], 1);
    assert_eq!(result["records"], 1);
    assert_eq!(result["events"], 1);
    let error = result["errors"][0].as_str().unwrap();
    assert!(error.contains("codex-app-server:missing-thread"), "{error}");
    assert!(error.contains("thread/read"), "{error}");
    assert!(error.contains("rollout missing"), "{error}");
}
