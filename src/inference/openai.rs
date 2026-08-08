use super::SemafusedEndpointConfig;
use super::http::{default_retries, default_timeout_secs, post_json, validate_transport};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OpenAiEndpoint {
    pub(crate) base_url: String,
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) api_key_env: Option<String>,
    #[serde(default = "default_timeout_secs")]
    timeout_secs: u64,
    #[serde(default = "default_retries")]
    max_retries: usize,
    /// Provider-specific request fields such as OpenRouter routing policy.
    #[serde(default)]
    extra_body: Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum EmbeddingBackend {
    OpenaiCompatible { endpoint: OpenAiEndpoint },
    Semafused { endpoint: SemafusedEndpointConfig },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GenerationOptions {
    pub(crate) temperature: f64,
    pub(crate) top_p: f64,
    pub(crate) max_tokens: usize,
    pub(crate) enable_thinking: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ChatCompletion {
    pub(crate) content: String,
    pub(crate) model: Option<String>,
    pub(crate) prompt_tokens: Option<u64>,
    pub(crate) completion_tokens: Option<u64>,
}

impl OpenAiEndpoint {
    pub(crate) fn validate(&self, label: &str) -> Result<()> {
        validate_transport(
            label,
            &self.base_url,
            self.api_key_env.as_deref(),
            self.timeout_secs,
            self.max_retries,
        )?;
        ensure!(
            !self.model.trim().is_empty(),
            "{label}.model cannot be empty"
        );
        Ok(())
    }

    pub(crate) async fn embed(&self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let mut body = self.extra_body.clone();
        ensure!(
            !body.contains_key("model") && !body.contains_key("input"),
            "embedding extra_body cannot override model or input"
        );
        body.insert("model".into(), json!(self.model));
        body.insert("input".into(), json!(input));
        let value = post_json(
            &self.base_url,
            "embeddings",
            self.api_key_env.as_deref(),
            self.timeout_secs,
            self.max_retries,
            &Value::Object(body),
        )
        .await
        .context("embedding request failed")?;
        let data = value
            .get("data")
            .and_then(Value::as_array)
            .context("embedding response is missing data array")?;
        ensure!(
            data.len() == input.len(),
            "embedding response returned {} vectors for {} inputs",
            data.len(),
            input.len()
        );

        let mut ordered = vec![None; input.len()];
        let mut dimensions = None;
        for item in data {
            let index = item
                .get("index")
                .and_then(Value::as_u64)
                .context("embedding item is missing a non-negative index")?
                as usize;
            ensure!(
                index < input.len(),
                "embedding index {index} is out of range"
            );
            ensure!(
                ordered[index].is_none(),
                "duplicate embedding index {index}"
            );
            let values = item
                .get("embedding")
                .and_then(Value::as_array)
                .context("embedding item is missing embedding array")?
                .iter()
                .map(|value| {
                    let number = value
                        .as_f64()
                        .context("embedding contains a non-numeric value")?;
                    ensure!(number.is_finite(), "embedding contains a non-finite value");
                    ensure!(
                        number >= f64::from(f32::MIN) && number <= f64::from(f32::MAX),
                        "embedding value does not fit f32"
                    );
                    Ok(number as f32)
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(!values.is_empty(), "embedding vector cannot be empty");
            if let Some(expected) = dimensions {
                ensure!(
                    values.len() == expected,
                    "embedding dimensions changed within one response"
                );
            } else {
                dimensions = Some(values.len());
            }
            ordered[index] = Some(values);
        }
        ordered
            .into_iter()
            .enumerate()
            .map(|(index, value)| value.with_context(|| format!("missing embedding index {index}")))
            .collect()
    }

    pub(crate) async fn chat_json(
        &self,
        system: &str,
        user: &str,
        options: GenerationOptions,
    ) -> Result<String> {
        ensure!(
            options.temperature.is_finite() && options.temperature >= 0.0,
            "temperature must be finite and non-negative"
        );
        ensure!(
            options.top_p.is_finite() && options.top_p > 0.0 && options.top_p <= 1.0,
            "top_p must be in (0, 1]"
        );
        ensure!(options.max_tokens > 0, "max_tokens must be positive");
        Ok(self.chat(system, user, options).await?.content)
    }

    pub(crate) async fn chat(
        &self,
        system: &str,
        user: &str,
        options: GenerationOptions,
    ) -> Result<ChatCompletion> {
        self.chat_messages(
            json!([
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]),
            options,
        )
        .await
    }

    pub(crate) async fn chat_user_only(
        &self,
        user: &str,
        options: GenerationOptions,
    ) -> Result<ChatCompletion> {
        self.chat_messages(json!([{"role": "user", "content": user}]), options)
            .await
    }

    async fn chat_messages(
        &self,
        messages: Value,
        options: GenerationOptions,
    ) -> Result<ChatCompletion> {
        ensure!(
            options.temperature.is_finite() && options.temperature >= 0.0,
            "temperature must be finite and non-negative"
        );
        ensure!(
            options.top_p.is_finite() && options.top_p > 0.0 && options.top_p <= 1.0,
            "top_p must be in (0, 1]"
        );
        ensure!(options.max_tokens > 0, "max_tokens must be positive");
        let mut body = self.extra_body.clone();
        for field in [
            "model",
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "chat_template_kwargs",
        ] {
            ensure!(
                !body.contains_key(field),
                "chat extra_body cannot override {field}"
            );
        }
        body.extend(Map::from_iter([
            ("model".into(), json!(self.model)),
            ("messages".into(), messages),
            ("temperature".into(), json!(options.temperature)),
            ("top_p".into(), json!(options.top_p)),
            ("max_tokens".into(), json!(options.max_tokens)),
            (
                "chat_template_kwargs".into(),
                json!({"enable_thinking": options.enable_thinking}),
            ),
        ]));
        let value = post_json(
            &self.base_url,
            "chat/completions",
            self.api_key_env.as_deref(),
            self.timeout_secs,
            self.max_retries,
            &Value::Object(body),
        )
        .await
        .context("chat completion request failed")?;
        let content = value
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .context("chat completion response is missing string content")?;
        ensure!(
            !content.trim().is_empty(),
            "chat completion returned empty content"
        );
        Ok(ChatCompletion {
            content: content.to_owned(),
            model: value
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_owned),
            prompt_tokens: value
                .pointer("/usage/prompt_tokens")
                .and_then(Value::as_u64),
            completion_tokens: value
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64),
        })
    }
}

impl EmbeddingBackend {
    pub(crate) fn validate(&self, label: &str) -> Result<()> {
        match self {
            Self::OpenaiCompatible { endpoint } => endpoint.validate(label),
            Self::Semafused { endpoint } => endpoint.validate(label),
        }
    }

    pub(crate) fn backend_name(&self) -> &'static str {
        match self {
            Self::OpenaiCompatible { .. } => "openai_compatible",
            Self::Semafused { .. } => "semafused",
        }
    }

    pub(crate) fn model_identity(&self) -> &str {
        match self {
            Self::OpenaiCompatible { endpoint } => &endpoint.model,
            Self::Semafused { endpoint } => endpoint.model_fingerprint(),
        }
    }

    pub(crate) async fn embed(&self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::OpenaiCompatible { endpoint } => endpoint.embed(input).await,
            Self::Semafused { endpoint } => endpoint.embed(input).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_credential_names_without_reading_credentials() {
        let endpoint = OpenAiEndpoint {
            base_url: "http://127.0.0.1:8000/v1".into(),
            model: "model-revision".into(),
            api_key_env: Some("SEMAFUSE_TEST_KEY".into()),
            timeout_secs: 1,
            max_retries: 0,
            extra_body: Map::new(),
        };
        endpoint.validate("embedding").unwrap();
    }

    #[test]
    fn rejects_credential_values_in_the_environment_name_field() {
        let endpoint = OpenAiEndpoint {
            base_url: "http://localhost/v1".into(),
            model: "model".into(),
            api_key_env: Some("secret key".into()),
            timeout_secs: 1,
            max_retries: 0,
            extra_body: Map::new(),
        };
        assert!(endpoint.validate("embedding").is_err());
    }

    #[test]
    fn rejects_cleartext_credentials_to_remote_hosts() {
        let endpoint = OpenAiEndpoint {
            base_url: "http://inference.example/v1".into(),
            model: "model".into(),
            api_key_env: Some("SEMAFUSE_TEST_KEY".into()),
            timeout_secs: 1,
            max_retries: 0,
            extra_body: Map::new(),
        };
        assert!(endpoint.validate("embedding").is_err());
    }
}
