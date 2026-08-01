use super::{
    codex::run_codex_processor,
    openai::{GenerationOptions, OpenAiEndpoint},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

const DEFAULT_CODEX_TIMEOUT_SECS: u64 = 600;
const DEFAULT_CODEX_MAX_OUTPUT_BYTES: usize = 1_048_576;
const MAX_CODEX_TIMEOUT_SECS: u64 = 3_600;
const MAX_CODEX_OUTPUT_BYTES: usize = 16 * 1_048_576;

/// A query-time LazyMem processor that consumes the Appendix G.1 prompt and
/// must return its strict positional JSON decision array.
///
/// The tagged configuration deliberately exposes only the supported remote
/// API and Codex CLI transports. The Codex transport owns its argument vector;
/// neither the prompt nor configuration is interpolated into a shell command.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "backend", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum PromptProcessorBackend {
    OpenaiCompatible {
        endpoint: OpenAiEndpoint,
    },
    CodexCli {
        model: String,
        #[serde(default = "default_codex_command")]
        command: String,
        #[serde(default = "default_codex_timeout_secs")]
        timeout_secs: u64,
        #[serde(default = "default_codex_max_output_bytes")]
        max_output_bytes: usize,
    },
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProcessorBackendIdentity<'a> {
    pub(crate) backend: &'static str,
    pub(crate) transport: &'static str,
    pub(crate) model: &'a str,
}

fn default_codex_command() -> String {
    "codex".into()
}

fn default_codex_timeout_secs() -> u64 {
    DEFAULT_CODEX_TIMEOUT_SECS
}

fn default_codex_max_output_bytes() -> usize {
    DEFAULT_CODEX_MAX_OUTPUT_BYTES
}

impl PromptProcessorBackend {
    pub(crate) fn validate(&self, label: &str) -> Result<()> {
        match self {
            Self::OpenaiCompatible { endpoint } => endpoint.validate(label),
            Self::CodexCli {
                model,
                command,
                timeout_secs,
                max_output_bytes,
            } => {
                ensure!(
                    !model.trim().is_empty() && !model.contains('\0') && model.len() <= 512,
                    "{label}.model must be a non-empty model identifier no longer than 512 bytes"
                );
                ensure!(
                    !command.trim().is_empty() && !command.contains('\0') && command.len() <= 4_096,
                    "{label}.command must be a direct executable name or path"
                );
                ensure!(
                    (1..=MAX_CODEX_TIMEOUT_SECS).contains(timeout_secs),
                    "{label}.timeout_secs must be between 1 and {MAX_CODEX_TIMEOUT_SECS}"
                );
                ensure!(
                    (1..=MAX_CODEX_OUTPUT_BYTES).contains(max_output_bytes),
                    "{label}.max_output_bytes must be between 1 and {MAX_CODEX_OUTPUT_BYTES}"
                );
                Ok(())
            }
        }
    }

    pub(crate) fn identity(&self) -> ProcessorBackendIdentity<'_> {
        match self {
            Self::OpenaiCompatible { endpoint } => ProcessorBackendIdentity {
                backend: "openai_compatible",
                transport: "openai-chat-completions-v1",
                model: &endpoint.model,
            },
            Self::CodexCli { model, .. } => ProcessorBackendIdentity {
                backend: "codex_cli",
                transport: "codex-exec-json-envelope-v1",
                model,
            },
        }
    }

    pub(crate) fn default_max_concurrency(&self) -> usize {
        match self {
            Self::OpenaiCompatible { .. } => 64,
            Self::CodexCli { .. } => 1,
        }
    }

    pub(crate) fn max_concurrency_limit(&self) -> usize {
        match self {
            Self::OpenaiCompatible { .. } => 256,
            Self::CodexCli { .. } => 4,
        }
    }

    pub(crate) async fn chat_json(
        &self,
        system: &str,
        user: &str,
        options: GenerationOptions,
        expected_decisions: usize,
    ) -> Result<String> {
        ensure!(
            expected_decisions > 0,
            "processor request must expect at least one decision"
        );
        match self {
            Self::OpenaiCompatible { endpoint } => endpoint
                .chat_json(system, user, options)
                .await
                .context("OpenAI-compatible processor failed"),
            Self::CodexCli {
                model,
                command,
                timeout_secs,
                max_output_bytes,
            } => {
                run_codex_processor(
                    command,
                    model,
                    system,
                    user,
                    expected_decisions,
                    *timeout_secs,
                    *max_output_bytes,
                )
                .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, to_value};

    #[test]
    fn prompt_processor_backends_are_strictly_tagged_and_secret_free() {
        let api: PromptProcessorBackend = serde_json::from_value(json!({
            "backend": "openai_compatible",
            "endpoint": {
                "base_url": "https://inference.example/v1",
                "model": "qwen3-32b",
                "api_key_env": "SEMAFUSE_TEST_KEY"
            }
        }))
        .unwrap();
        assert_eq!(api.identity().backend, "openai_compatible");
        assert_eq!(api.identity().model, "qwen3-32b");
        assert_eq!(
            to_value(api.identity()).unwrap(),
            json!({
                "backend": "openai_compatible",
                "transport": "openai-chat-completions-v1",
                "model": "qwen3-32b"
            })
        );

        let codex: PromptProcessorBackend = serde_json::from_value(json!({
            "backend": "codex_cli",
            "model": "gpt-5.2-codex"
        }))
        .unwrap();
        codex.validate("processor").unwrap();
        assert_eq!(codex.default_max_concurrency(), 1);

        for invalid in [
            json!({"backend": "codex_cli", "model": "gpt-5.2-codex", "args": []}),
            json!({"backend": "openai_compatible", "endpoint": {"base_url": "https://inference.example/v1", "model": "qwen3-32b", "token": "secret"}}),
            json!({"backend": "unknown", "model": "model"}),
        ] {
            assert!(
                serde_json::from_value::<PromptProcessorBackend>(invalid).is_err(),
                "accepted an unknown processor field or backend"
            );
        }
    }
}
