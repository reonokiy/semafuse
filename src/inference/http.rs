use anyhow::{Context, Result, bail, ensure};
use reqwest::{Client, Response, StatusCode};
use serde_json::Value;
use std::{sync::OnceLock, time::Duration};

static INFERENCE_HTTP_CLIENT: OnceLock<Client> = OnceLock::new();
const MAX_JSON_RESPONSE_BYTES: usize = 128 * 1024 * 1024;
const MAX_ERROR_RESPONSE_BYTES: usize = 64 * 1024;

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
    ensure!(
        url.host_str().is_some(),
        "{label}.base_url must include a host"
    );
    ensure!(
        url.username().is_empty() && url.password().is_none(),
        "{label}.base_url must not include URL credentials"
    );
    ensure!(
        url.query().is_none() && url.fragment().is_none(),
        "{label}.base_url must not include a query or fragment"
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
    let client = INFERENCE_HTTP_CLIENT.get_or_init(Client::new);
    let url = format!("{}/{}", base_url.trim_end_matches('/'), path);
    let token = api_key_env
        .map(|name| {
            std::env::var(name).with_context(|| {
                format!("inference credential environment variable {name} is unset")
            })
        })
        .transpose()?;

    for attempt in 0..=max_retries {
        let mut request = client
            .post(&url)
            .timeout(Duration::from_secs(timeout_secs))
            .json(body);
        if let Some(token) = &token {
            request = request.bearer_auth(token);
        }
        match request.send().await {
            Ok(mut response) => {
                let status = response.status();
                if status.is_success() {
                    let body =
                        read_response_bytes(&mut response, MAX_JSON_RESPONSE_BYTES, "JSON").await?;
                    return serde_json::from_slice(&body).context("decode inference JSON response");
                }
                let retryable = status == StatusCode::TOO_MANY_REQUESTS
                    || status == StatusCode::REQUEST_TIMEOUT
                    || status.is_server_error();
                let detail = read_response_bytes(&mut response, MAX_ERROR_RESPONSE_BYTES, "error")
                    .await
                    .map(|body| String::from_utf8_lossy(&body).into_owned())
                    .unwrap_or_else(|error| format!("<response body unavailable: {error}>"));
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

async fn read_response_bytes(response: &mut Response, limit: usize, kind: &str) -> Result<Vec<u8>> {
    if let Some(length) = response.content_length() {
        ensure!(
            length <= limit as u64,
            "inference {kind} response body exceeds the {limit}-byte limit"
        );
    }
    let capacity = response
        .content_length()
        .and_then(|length| usize::try_from(length).ok())
        .unwrap_or_default()
        .min(limit);
    let mut body = Vec::with_capacity(capacity);
    while let Some(chunk) = response
        .chunk()
        .await
        .with_context(|| format!("read inference {kind} response body"))?
    {
        ensure!(
            chunk.len() <= limit.saturating_sub(body.len()),
            "inference {kind} response body exceeds the {limit}-byte limit"
        );
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

async fn retry_delay(attempt: usize) {
    let exponent = attempt.min(6) as u32;
    tokio::time::sleep(Duration::from_millis(250 * 2_u64.pow(exponent))).await;
}

pub(super) fn truncate_error(value: &str) -> String {
    value.chars().take(2_000).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_rejects_url_secrets() {
        let error = validate_transport(
            "answerer",
            "https://user:secret@inference.example/v1",
            Some("SEMAFUSE_TEST_KEY"),
            1,
            0,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("must not include URL credentials"),
            "{error}"
        );

        for base_url in [
            "https://inference.example/v1?api_key=secret",
            "https://inference.example/v1#secret",
        ] {
            let error = validate_transport("answerer", base_url, None, 1, 0)
                .unwrap_err()
                .to_string();
            assert!(error.contains("query or fragment"), "{error}");
        }
    }

    #[tokio::test]
    async fn oversized_success_response_is_rejected_from_its_header() {
        use tokio::{
            io::{AsyncReadExt, AsyncWriteExt},
            net::TcpListener,
        };

        let listener = match TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => listener,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(error) => panic!("bind mock inference server: {error}"),
        };
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4_096];
            while !request.windows(4).any(|part| part == b"\r\n\r\n") {
                let count = stream.read(&mut buffer).await.unwrap();
                assert_ne!(count, 0, "client closed before sending request headers");
                request.extend_from_slice(&buffer[..count]);
            }
            let header = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                MAX_JSON_RESPONSE_BYTES + 1
            );
            stream.write_all(header.as_bytes()).await.unwrap();
        });

        let error = post_json(
            &format!("http://{address}"),
            "v1/test",
            None,
            2,
            0,
            &serde_json::json!({}),
        )
        .await
        .unwrap_err()
        .to_string();
        server.await.unwrap();
        assert!(error.contains("JSON response body exceeds"), "{error}");
    }
}
