use anyhow::{Context, Result, bail, ensure};
use reqwest::{Client, StatusCode};
use serde_json::Value;
use std::time::Duration;

pub(super) fn default_timeout_secs() -> u64 {
    120
}

pub(super) fn default_retries() -> usize {
    3
}

pub(super) fn validate_transport(
    label: &str,
    base_url: &str,
    api_key_env: Option<&str>,
    timeout_secs: u64,
    max_retries: usize,
) -> Result<()> {
    let url = reqwest::Url::parse(base_url)
        .with_context(|| format!("{label}.base_url must be an absolute URL"))?;
    ensure!(
        matches!(url.scheme(), "http" | "https"),
        "{label}.base_url must use http or https"
    );
    ensure!(timeout_secs > 0, "{label}.timeout_secs must be positive");
    ensure!(max_retries <= 10, "{label}.max_retries must not exceed 10");
    if let Some(name) = api_key_env {
        ensure!(
            !name.trim().is_empty()
                && name
                    .chars()
                    .all(|character| character == '_' || character.is_ascii_alphanumeric()),
            "{label}.api_key_env must be an environment variable name"
        );
        if url.scheme() == "http" {
            let host = url.host_str().unwrap_or_default();
            let loopback = host.eq_ignore_ascii_case("localhost")
                || host
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|address| address.is_loopback());
            ensure!(
                loopback,
                "{label}.base_url cannot send bearer credentials over cleartext HTTP to a non-loopback host"
            );
        }
    }
    Ok(())
}

pub(super) async fn post_json(
    base_url: &str,
    path: &str,
    api_key_env: Option<&str>,
    timeout_secs: u64,
    max_retries: usize,
    body: &Value,
) -> Result<Value> {
    let client = Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .context("build inference HTTP client")?;
    let url = format!("{}/{}", base_url.trim_end_matches('/'), path);
    let token = api_key_env
        .map(|name| {
            std::env::var(name).with_context(|| {
                format!("inference credential environment variable {name} is unset")
            })
        })
        .transpose()?;

    for attempt in 0..=max_retries {
        let mut request = client.post(&url).json(body);
        if let Some(token) = &token {
            request = request.bearer_auth(token);
        }
        match request.send().await {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    return response
                        .json::<Value>()
                        .await
                        .context("decode inference JSON response");
                }
                let retryable = status == StatusCode::TOO_MANY_REQUESTS
                    || status == StatusCode::REQUEST_TIMEOUT
                    || status.is_server_error();
                let detail = response.text().await.unwrap_or_default();
                if retryable && attempt < max_retries {
                    retry_delay(attempt).await;
                    continue;
                }
                bail!(
                    "inference endpoint returned HTTP {status}: {}",
                    truncate_error(&detail)
                );
            }
            Err(error) if (error.is_timeout() || error.is_connect()) && attempt < max_retries => {
                retry_delay(attempt).await;
            }
            Err(error) => return Err(error).context("send inference HTTP request"),
        }
    }
    unreachable!("retry loop always returns on its last attempt")
}

async fn retry_delay(attempt: usize) {
    let exponent = attempt.min(6) as u32;
    tokio::time::sleep(Duration::from_millis(250 * 2_u64.pow(exponent))).await;
}

pub(super) fn truncate_error(value: &str) -> String {
    value.chars().take(2_000).collect()
}
