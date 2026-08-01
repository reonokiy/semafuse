use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    process::{Command, Stdio},
};
use tempfile::TempDir;

fn exchange(version: &str, requests: &[Value], invalid_config: bool) -> Vec<Value> {
    let temp = TempDir::new().unwrap();
    let db = temp.path().join("memory.db");
    let config = temp.path().join("config.toml");
    if invalid_config {
        fs::write(
            &config,
            "[strategies]\nenabled = [\"history\"]\nrecall = \"missing\"\n",
        )
        .unwrap();
    }

    let executable = assert_cmd::cargo::cargo_bin("semafuse");
    let mut child = Command::new(executable)
        .args(["mcp", "--db"])
        .arg(&db)
        .env("SEMAFUSE_CONFIG", &config)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    let initialize = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": version,
            "capabilities": {},
            "clientInfo": {"name": "semafuse-test", "version": "1.0.0"}
        }
    });
    let initialized = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });
    let mut input = format!("{initialize}\n{initialized}\n");
    for request in requests {
        input.push_str(&format!("{request}\n"));
    }
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    drop(child.stdin.take());

    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "MCP server failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn response(responses: &[Value], id: i64) -> &Value {
    responses
        .iter()
        .find(|response| response["id"] == id)
        .unwrap()
}

#[test]
fn current_protocol_exposes_structured_tools_and_standard_errors() {
    let responses = exchange(
        "2026-07-28",
        &[
            json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
            json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"semafuse_recent","arguments":{}}}),
            json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"semafuse_search","arguments":{"query":"anything","mode":"invalid"}}}),
            json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"semafuse_recall","arguments":{"query":"anything"}}}),
            json!({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"not_a_tool","arguments":{}}}),
        ],
        true,
    );

    assert_eq!(responses.len(), 6);
    let initialize = response(&responses, 1);
    let list = response(&responses, 2);
    let recent = response(&responses, 3);
    let invalid_mode = response(&responses, 4);
    let recall_error = response(&responses, 5);
    let unknown_tool = response(&responses, 6);
    assert_eq!(initialize["result"]["protocolVersion"], "2026-07-28");
    assert_eq!(initialize["result"]["serverInfo"]["name"], "semafuse");

    let tools = list["result"]["tools"].as_array().unwrap();
    let names = tools
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        names,
        BTreeSet::from([
            "semafuse_recall",
            "semafuse_recent",
            "semafuse_search",
            "semafuse_session",
        ])
    );
    for (name, required) in [
        ("semafuse_search", "query"),
        ("semafuse_recall", "query"),
        ("semafuse_session", "session_id"),
    ] {
        let tool = tools.iter().find(|tool| tool["name"] == name).unwrap();
        assert!(
            tool["inputSchema"]["required"]
                .as_array()
                .unwrap()
                .iter()
                .any(|field| field == required)
        );
    }

    assert_eq!(recent["result"]["resultType"], "complete");
    assert_eq!(recent["result"]["isError"], false);
    assert_eq!(recent["result"]["structuredContent"]["results"], json!([]));
    assert_eq!(recent["result"]["content"][0]["text"], "[]");

    assert_eq!(invalid_mode["error"]["code"], -32602);
    assert_eq!(recall_error["result"]["isError"], true);
    assert!(
        recall_error["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("recall strategy missing is not enabled")
    );
    assert_eq!(unknown_tool["error"]["code"], -32602);
}

#[test]
fn legacy_protocol_is_negotiated_without_new_result_discriminator() {
    let responses = exchange(
        "2025-06-18",
        &[
            json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"semafuse_recent","arguments":{}}}),
            json!({"jsonrpc":"2.0","id":3,"method":"not/a/method","params":{}}),
        ],
        false,
    );

    assert_eq!(responses.len(), 3);
    let initialize = response(&responses, 1);
    let recent = response(&responses, 2);
    let unknown_method = response(&responses, 3);
    assert_eq!(initialize["result"]["protocolVersion"], "2025-06-18");
    assert!(recent["result"].get("resultType").is_none());
    assert_eq!(recent["result"]["structuredContent"]["results"], json!([]));
    assert_eq!(unknown_method["error"]["code"], -32601);
}
