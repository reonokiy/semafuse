use super::JsonContract;
use super::SemafusedEndpointConfig;
use super::http::{default_retries, default_timeout_secs, post_json, validate_transport};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OpenAiEndpoint {
    pub(crate) base_url: String,
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) api_key_env: Option<String>,
    #[serde(default)]
    thinking_control: ThinkingControl,
    #[serde(default = "default_timeout_secs")]
    timeout_secs: u64,
    #[serde(default = "default_retries")]
    max_retries: usize,
    /// Provider-specific request fields such as OpenRouter routing policy.
    #[serde(default)]
    extra_body: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ThinkingControl {
    /// Prefer a provider's standardized `reasoning` field when configured;
    /// otherwise pass the Hugging Face/vLLM chat-template option.
    #[default]
    ProviderParameters,
    /// Append Qwen3's documented `/no_think` soft switch when thinking is
    /// disabled. This covers gateways that do not propagate the upstream
    /// chat-template option.
    QwenSoftSwitch,
    /// Send no model-specific thinking parameter. This is for checkpoints that
    /// are intrinsically non-thinking (for example an `*-instruct` release).
    NoControl,
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
    pub(crate) provider: Option<String>,
    pub(crate) finish_reason: Option<String>,
    pub(crate) prompt_tokens: Option<u64>,
    pub(crate) completion_tokens: Option<u64>,
    pub(crate) reasoning_tokens: Option<u64>,
    pub(crate) reasoning_present: bool,
    /// Provider-reported request cost, in the provider's billing units.
    /// OpenRouter exposes this as credits in `usage.cost`.
    pub(crate) cost: Option<f64>,
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
        if self.thinking_control == ThinkingControl::QwenSoftSwitch {
            ensure!(
                self.model.to_ascii_lowercase().contains("qwen3"),
                "{label}.thinking_control=qwen_soft_switch requires a Qwen3 model"
            );
        }
        Ok(())
    }

    pub(crate) fn validate_generation(
        &self,
        label: &str,
        options: &GenerationOptions,
    ) -> Result<()> {
        self.validate(label)?;
        for field in [
            "model",
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "chat_template_kwargs",
        ] {
            ensure!(
                !self.extra_body.contains_key(field),
                "{label}.extra_body cannot override {field}"
            );
        }
        self.validate_thinking_control(label, options.enable_thinking)?;
        Ok(())
    }

    fn validate_thinking_control(
        &self,
        label: &str,
        enable_thinking: bool,
    ) -> Result<Option<bool>> {
        let standardized_reasoning = self
            .extra_body
            .get("reasoning")
            .map(reasoning_enabled)
            .transpose()?;
        match self.thinking_control {
            ThinkingControl::ProviderParameters | ThinkingControl::QwenSoftSwitch => {
                if let Some(reasoning_enabled) = standardized_reasoning {
                    ensure!(
                        reasoning_enabled == enable_thinking,
                        "{label}.reasoning extra_body disagrees with generation.enable_thinking"
                    );
                }
            }
            ThinkingControl::NoControl => {
                ensure!(
                    !enable_thinking,
                    "{label}.thinking_control=no_control requires enable_thinking=false"
                );
                ensure!(
                    standardized_reasoning.is_none(),
                    "{label}.thinking_control=no_control cannot be combined with reasoning extra_body"
                );
            }
        }
        Ok(standardized_reasoning)
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

    /// Request a provider-enforced JSON response. Domain-specific validation
    /// remains with the workflow that owns the selected contract.
    pub(crate) async fn complete_json_contract(
        &self,
        system: &str,
        user: &str,
        options: GenerationOptions,
        contract: JsonContract,
    ) -> Result<ChatCompletion> {
        self.chat_messages(
            json!([
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]),
            options,
            Some(contract.openai_response_format()?),
        )
        .await
    }

    #[cfg(feature = "benchmarks")]
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
            None,
        )
        .await
    }

    #[cfg(feature = "benchmarks")]
    pub(crate) async fn chat_user_only(
        &self,
        user: &str,
        options: GenerationOptions,
    ) -> Result<ChatCompletion> {
        self.chat_messages(json!([{"role": "user", "content": user}]), options, None)
            .await
    }

    async fn chat_messages(
        &self,
        mut messages: Value,
        options: GenerationOptions,
        response_format: Option<Value>,
    ) -> Result<ChatCompletion> {
        let structured_response = response_format.is_some();
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
        let standardized_reasoning =
            self.validate_thinking_control("chat endpoint", options.enable_thinking)?;
        match self.thinking_control {
            ThinkingControl::ProviderParameters => {}
            ThinkingControl::QwenSoftSwitch => {
                if !options.enable_thinking {
                    append_qwen_no_think(&mut messages)?;
                }
            }
            ThinkingControl::NoControl => {}
        }
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
        ]));
        if self.thinking_control == ThinkingControl::ProviderParameters
            && standardized_reasoning.is_none()
        {
            body.insert(
                "chat_template_kwargs".into(),
                json!({"enable_thinking": options.enable_thinking}),
            );
        }
        if let Some(response_format) = response_format {
            ensure!(
                !body.contains_key("response_format"),
                "chat extra_body cannot override response_format for a structured processor request"
            );
            body.insert("response_format".into(), response_format);
        }
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
        parse_chat_completion(
            &value,
            structured_response,
            self.thinking_control,
            options.enable_thinking,
        )
    }
}

fn parse_chat_completion(
    value: &Value,
    structured_response: bool,
    thinking_control: ThinkingControl,
    enable_thinking: bool,
) -> Result<ChatCompletion> {
    let reasoning_tokens = value
        .pointer("/usage/completion_tokens_details/reasoning_tokens")
        .and_then(Value::as_u64);
    let reasoning_present = value
        .pointer("/choices/0/message/reasoning")
        .is_some_and(reasoning_has_content);
    if thinking_control == ThinkingControl::QwenSoftSwitch && !enable_thinking {
        ensure!(
            reasoning_tokens.is_some(),
            "Qwen /no_think response is missing reasoning-token telemetry"
        );
        ensure!(
            !reasoning_present && reasoning_tokens.unwrap_or(0) <= 1,
            "Qwen /no_think was not honored: provider returned substantive reasoning"
        );
    }
    if thinking_control == ThinkingControl::NoControl {
        ensure!(
            !reasoning_present && reasoning_tokens.unwrap_or(0) == 0,
            "inherently non-thinking endpoint returned reasoning"
        );
    }
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str);
    let content = match (content, structured_response) {
        (Some(content), _) => content,
        (None, true) => "",
        (None, false) => {
            return Err(anyhow::anyhow!(
                "chat completion response is missing string content"
            ));
        }
    };
    if !structured_response {
        ensure!(
            !content.trim().is_empty(),
            "chat completion returned empty content"
        );
    }
    Ok(ChatCompletion {
        content: content.to_owned(),
        model: value
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned),
        provider: value
            .get("provider")
            .and_then(Value::as_str)
            .map(str::to_owned),
        finish_reason: value
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .map(str::to_owned),
        prompt_tokens: value
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64),
        completion_tokens: value
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64),
        reasoning_tokens,
        reasoning_present,
        cost: value.pointer("/usage/cost").and_then(Value::as_f64),
    })
}

fn append_qwen_no_think(messages: &mut Value) -> Result<()> {
    let messages = messages
        .as_array_mut()
        .context("chat messages must be an array")?;
    let user = messages
        .iter_mut()
        .rev()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .context("qwen_soft_switch requires a user message")?;
    let content = user
        .get("content")
        .and_then(Value::as_str)
        .context("qwen_soft_switch requires string user content")?
        .trim_end()
        .to_owned();
    if content.lines().next_back() == Some("/no_think") {
        return Ok(());
    }
    user["content"] = json!(format!("{content}\n\n/no_think"));
    Ok(())
}

fn reasoning_has_content(reasoning: &Value) -> bool {
    match reasoning {
        Value::Null => false,
        Value::String(text) => !text.trim().is_empty(),
        _ => true,
    }
}

fn reasoning_enabled(value: &Value) -> Result<bool> {
    let reasoning = value
        .as_object()
        .context("reasoning extra_body must be an object")?;
    let enabled = reasoning
        .get("enabled")
        .map(|value| {
            value
                .as_bool()
                .context("reasoning.enabled must be a boolean")
        })
        .transpose()?;
    let effort_enabled = reasoning
        .get("effort")
        .map(|value| {
            value
                .as_str()
                .context("reasoning.effort must be a string")
                .map(|effort| effort != "none")
        })
        .transpose()?;
    if let (Some(enabled), Some(effort_enabled)) = (enabled, effort_enabled) {
        ensure!(
            enabled == effort_enabled,
            "reasoning.enabled and reasoning.effort disagree"
        );
    }
    enabled
        .or(effort_enabled)
        .context("reasoning extra_body must set enabled or effort")
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
    fn standardized_reasoning_control_is_unambiguous() {
        assert!(reasoning_enabled(&json!({"enabled": true})).unwrap());
        assert!(!reasoning_enabled(&json!({"effort": "none"})).unwrap());
        assert!(reasoning_enabled(&json!({"effort": "low"})).unwrap());
        assert!(reasoning_enabled(&json!({"enabled": true, "effort": "none"})).is_err());
    }

    #[test]
    fn qwen_soft_switch_targets_the_last_user_message_without_duplication() {
        let mut messages = json!([
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "new"}
        ]);
        append_qwen_no_think(&mut messages).unwrap();
        append_qwen_no_think(&mut messages).unwrap();
        assert_eq!(messages[0]["content"], "old");
        assert_eq!(messages[2]["content"], "new\n\n/no_think");
    }

    #[test]
    fn whitespace_reasoning_is_not_substantive() {
        assert!(!reasoning_has_content(&Value::Null));
        assert!(!reasoning_has_content(&json!("\n\n")));
        assert!(reasoning_has_content(&json!("actual reasoning")));
    }

    #[test]
    fn structured_length_response_preserves_usage_when_content_is_null() {
        let response = json!({
            "model": "qwen3",
            "provider": "provider",
            "choices": [{"finish_reason": "length", "message": {"content": null}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 3072,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "cost": 0.001
            }
        });
        let completion =
            parse_chat_completion(&response, true, ThinkingControl::ProviderParameters, false)
                .unwrap();
        assert!(completion.content.is_empty());
        assert_eq!(completion.finish_reason.as_deref(), Some("length"));
        assert_eq!(completion.completion_tokens, Some(3072));
        assert_eq!(completion.cost, Some(0.001));
        assert!(
            parse_chat_completion(&response, false, ThinkingControl::ProviderParameters, false)
                .is_err()
        );
    }

    #[test]
    fn no_control_rejects_response_reasoning() {
        let response = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": "OK", "reasoning": "hidden reasoning"}
            }],
            "usage": {
                "completion_tokens_details": {"reasoning_tokens": 1}
            }
        });
        assert!(
            parse_chat_completion(&response, false, ThinkingControl::NoControl, false).is_err()
        );
    }

    #[test]
    fn no_control_conflicts_fail_during_generation_validation() {
        let disabled = GenerationOptions {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 32,
            enable_thinking: false,
        };
        let mut endpoint: OpenAiEndpoint = serde_json::from_value(json!({
            "base_url": "https://inference.example/v1",
            "model": "non-thinking-instruct",
            "api_key_env": "SEMAFUSE_TEST_KEY",
            "thinking_control": "no_control"
        }))
        .unwrap();
        endpoint.validate_generation("answerer", &disabled).unwrap();

        let enabled = GenerationOptions {
            enable_thinking: true,
            ..disabled
        };
        assert!(endpoint.validate_generation("answerer", &enabled).is_err());

        endpoint
            .extra_body
            .insert("reasoning".into(), json!({"enabled": false}));
        assert!(endpoint.validate_generation("answerer", &disabled).is_err());

        endpoint.extra_body.remove("reasoning");
        endpoint.extra_body.insert(
            "chat_template_kwargs".into(),
            json!({"enable_thinking": false}),
        );
        assert!(endpoint.validate_generation("answerer", &disabled).is_err());
    }

    #[cfg(feature = "benchmarks")]
    #[tokio::test]
    async fn structured_no_control_sends_schema_without_thinking_fields() {
        use tokio::{
            io::{AsyncReadExt, AsyncWriteExt},
            net::TcpListener,
            sync::oneshot,
        };

        let listener = match TcpListener::bind("127.0.0.1:0").await {
            Ok(listener) => listener,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(error) => panic!("bind mock chat server: {error}"),
        };
        let address = listener.local_addr().unwrap();
        let (body_sender, body_receiver) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4_096];
            let header_end = loop {
                let count = stream.read(&mut buffer).await.unwrap();
                assert!(count > 0, "mock request ended before its headers");
                request.extend_from_slice(&buffer[..count]);
                if let Some(position) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                    break position + 4;
                }
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap();
            while request.len() < header_end + content_length {
                let count = stream.read(&mut buffer).await.unwrap();
                assert!(count > 0, "mock request ended before its body");
                request.extend_from_slice(&buffer[..count]);
            }
            let body: Value =
                serde_json::from_slice(&request[header_end..header_end + content_length]).unwrap();
            body_sender.send(body).unwrap();

            let response = serde_json::to_vec(&json!({
                "model": "non-thinking-instruct",
                "provider": "MockProvider",
                "choices": [{"finish_reason": "stop", "message": {"content": r#"{"answer":"OK"}"#}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "cost": 0.000001}
            }))
            .unwrap();
            let header = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                response.len()
            );
            stream.write_all(header.as_bytes()).await.unwrap();
            stream.write_all(&response).await.unwrap();
        });

        let endpoint: OpenAiEndpoint = serde_json::from_value(json!({
            "base_url": format!("http://{address}/v1"),
            "model": "non-thinking-instruct",
            "thinking_control": "no_control",
            "timeout_secs": 5,
            "max_retries": 0
        }))
        .unwrap();
        fn answer_schema() -> Value {
            json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["answer"],
                "properties": {"answer": {"type": "string", "minLength": 1}}
            })
        }
        let completion = endpoint
            .complete_json_contract(
                "system instruction",
                "original prompt",
                GenerationOptions {
                    temperature: 0.0,
                    top_p: 1.0,
                    max_tokens: 32,
                    enable_thinking: false,
                },
                JsonContract::direct_object(
                    "test_answer",
                    "Return the required answer object.",
                    answer_schema,
                ),
            )
            .await
            .unwrap();
        let body = body_receiver.await.unwrap();
        server.await.unwrap();

        assert!(body.get("reasoning").is_none());
        assert!(body.get("chat_template_kwargs").is_none());
        assert_eq!(body["messages"][0]["content"], "system instruction");
        assert_eq!(body["messages"][1]["content"], "original prompt");
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            "test_answer"
        );
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["required"],
            json!(["answer"])
        );
        assert_eq!(completion.content, r#"{"answer":"OK"}"#);
        assert_eq!(completion.finish_reason.as_deref(), Some("stop"));
        assert_eq!(completion.reasoning_tokens, None);
        assert!(!completion.reasoning_present);
    }

    #[test]
    fn validates_credential_names_without_reading_credentials() {
        let endpoint = OpenAiEndpoint {
            base_url: "http://127.0.0.1:8000/v1".into(),
            model: "model-revision".into(),
            api_key_env: Some("SEMAFUSE_TEST_KEY".into()),
            thinking_control: ThinkingControl::default(),
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
            thinking_control: ThinkingControl::default(),
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
            thinking_control: ThinkingControl::default(),
            timeout_secs: 1,
            max_retries: 0,
            extra_body: Map::new(),
        };
        assert!(endpoint.validate("embedding").is_err());
    }
}
