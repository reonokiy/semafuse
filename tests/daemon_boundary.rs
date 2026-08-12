#![cfg(unix)]

use assert_cmd::Command;
use predicates::str::contains;
use std::{
    fs::{self, Permissions},
    os::unix::fs::PermissionsExt,
};

#[test]
fn daemon_status_never_starts_a_server_process() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let fake_bin = directory.path().join("bin");
    fs::create_dir(&fake_bin).expect("fake bin directory");
    let marker = directory.path().join("started");
    let fake_daemon = fake_bin.join("semafuse");
    fs::write(
        &fake_daemon,
        format!("#!/bin/sh\nprintf started > '{}'\n", marker.display()),
    )
    .expect("fake daemon");
    fs::set_permissions(&fake_daemon, Permissions::from_mode(0o700))
        .expect("fake daemon permissions");

    let mut command = Command::cargo_bin("semafuse").expect("semafuse binary");
    command
        .env("PATH", &fake_bin)
        .args([
            "daemon",
            "status",
            "--socket",
            directory
                .path()
                .join("missing.sock")
                .to_str()
                .expect("UTF-8 temporary path"),
        ])
        .assert()
        .failure()
        .stderr(contains("Semafuse daemon status probe failed"));

    assert!(
        !marker.exists(),
        "daemon status must never spawn a server process"
    );
}

#[cfg(feature = "daemon")]
#[test]
fn daemon_serve_loads_the_requested_config() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let missing = directory.path().join("missing.toml");
    let mut command = Command::cargo_bin("semafuse").expect("semafuse binary");

    command
        .args([
            "daemon",
            "serve",
            "--config",
            missing.to_str().expect("UTF-8 temporary path"),
        ])
        .assert()
        .failure()
        .stderr(contains(format!(
            "failed to read config {}",
            missing.display()
        )));
}
