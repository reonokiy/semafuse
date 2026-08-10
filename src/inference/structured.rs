use super::{MAX_PROCESSOR_COMPRESSED_CONTENT_CHARS, MAX_PROCESSOR_REASON_CHARS};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};

/// Provider-enforced JSON shapes used by prompt-driven inference workflows.
///
/// Transport schemas deliberately stop at structural constraints. Callers
/// remain responsible for domain validation (for example, checking that an
/// evidence index refers to an item in the current retrieval result).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JsonContract {
    WindowDecisions { count: usize },
    EvidenceGap,
}

impl JsonContract {
    pub(super) fn validate(self) -> Result<()> {
        if let Self::WindowDecisions { count } = self {
            ensure!(
                count > 0,
                "processor request must expect at least one decision"
            );
        }
        Ok(())
    }

    pub(super) fn openai_response_format(self) -> Result<Value> {
        self.validate()?;
        Ok(json!({
            "type": "json_schema",
            "json_schema": {
                "name": self.name(),
                "strict": true,
                "schema": self.payload_schema()
            }
        }))
    }

    /// Codex currently requires an object at the output-schema root. Window
    /// decisions therefore retain their established transport envelope, while
    /// the already-object-shaped evidence-gap contract can be emitted directly.
    pub(super) fn codex_output_schema(self) -> Result<Value> {
        self.validate()?;
        Ok(match self {
            Self::WindowDecisions { .. } => json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["decisions"],
                "properties": {
                    "decisions": self.payload_schema()
                }
            }),
            Self::EvidenceGap => self.payload_schema(),
        })
    }

    pub(super) fn codex_instruction(self) -> &'static str {
        match self {
            Self::WindowDecisions { .. } => {
                "Return the required JSON decision array as the value of one top-level object field named `decisions`. Do not add any other top-level fields. Semafuse removes this transport envelope before local validation."
            }
            Self::EvidenceGap => {
                "Return the required evidence-gap JSON object directly. Include exactly `is_sufficient`, `evidence_indices`, `gaps`, `new_query`, and `confidence_score`; do not add any other fields."
            }
        }
    }

    pub(super) fn decode_codex_response(self, response: &str) -> Result<String> {
        match self {
            Self::WindowDecisions { count } => {
                let envelope: CodexWindowDecisionsEnvelope = serde_json::from_str(response)
                    .context(
                        "Codex processor response does not match its structured-output envelope",
                    )?;
                ensure!(
                    envelope.decisions.len() == count,
                    "Codex processor returned {} decisions; expected {count}",
                    envelope.decisions.len()
                );
                serde_json::to_string(&envelope.decisions)
                    .context("unwrap Codex processor decisions for local validation")
            }
            Self::EvidenceGap => {
                let value: Value = serde_json::from_str(response)
                    .context("Codex processor response is not a JSON evidence-gap object")?;
                serde_json::to_string(&value).context("normalize Codex evidence-gap response")
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::WindowDecisions { .. } => "lazymem_window_decisions",
            Self::EvidenceGap => "evidence_gap",
        }
    }

    fn payload_schema(self) -> Value {
        match self {
            Self::WindowDecisions { count } => window_decisions_schema(count),
            Self::EvidenceGap => evidence_gap_schema(),
        }
    }
}

fn window_decisions_schema(count: usize) -> Value {
    json!({
        "type": "array",
        "minItems": count,
        "maxItems": count,
        "items": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["op", "compressed_content", "reason"],
                    "properties": {
                        "op": {"type": "string", "const": "KEEP"},
                        "compressed_content": {"type": "string", "minLength": 1, "maxLength": MAX_PROCESSOR_COMPRESSED_CONTENT_CHARS},
                        "reason": {"type": "string", "minLength": 1, "maxLength": MAX_PROCESSOR_REASON_CHARS}
                    }
                },
                {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["op", "compressed_content", "reason"],
                    "properties": {
                        "op": {"type": "string", "const": "DROP"},
                        "compressed_content": {"type": "string", "const": ""},
                        "reason": {"type": "string", "minLength": 1, "maxLength": MAX_PROCESSOR_REASON_CHARS}
                    }
                }
            ]
        }
    })
}

fn evidence_gap_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": [
            "is_sufficient",
            "evidence_indices",
            "gaps",
            "new_query",
            "confidence_score"
        ],
        "properties": {
            "is_sufficient": {"type": "boolean"},
            "evidence_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0}
            },
            "gaps": {
                "type": "array",
                "items": {"type": "string"}
            },
            "new_query": {"type": "string"},
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
        }
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CodexWindowDecisionsEnvelope {
    decisions: Vec<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_contract_requires_the_exact_decision_count() {
        let contract = JsonContract::WindowDecisions { count: 8 };
        let format = contract.openai_response_format().unwrap();
        let schema = &format["json_schema"]["schema"];
        assert_eq!(format["json_schema"]["strict"], true);
        assert_eq!(schema["type"], "array");
        assert_eq!(schema["minItems"], 8);
        assert_eq!(schema["maxItems"], 8);
        let variants = schema["items"]["oneOf"].as_array().unwrap();
        assert_eq!(variants.len(), 2);
        assert!(variants.iter().all(|variant| {
            variant["required"] == json!(["op", "compressed_content", "reason"])
                && variant["additionalProperties"] == false
                && variant["properties"]["reason"]["maxLength"] == MAX_PROCESSOR_REASON_CHARS
        }));
        assert_eq!(variants[0]["properties"]["op"]["const"], "KEEP");
        assert_eq!(
            variants[0]["properties"]["compressed_content"]["minLength"],
            1
        );
        assert_eq!(
            variants[0]["properties"]["compressed_content"]["maxLength"],
            MAX_PROCESSOR_COMPRESSED_CONTENT_CHARS
        );
        assert_eq!(variants[1]["properties"]["op"]["const"], "DROP");
        assert_eq!(variants[1]["properties"]["compressed_content"]["const"], "");
    }

    #[test]
    fn evidence_gap_contract_is_strict_for_both_transports() {
        let contract = JsonContract::EvidenceGap;
        let openai = contract.openai_response_format().unwrap();
        let codex = contract.codex_output_schema().unwrap();
        let schema = &openai["json_schema"]["schema"];
        assert_eq!(openai["json_schema"]["name"], "evidence_gap");
        assert_eq!(openai["json_schema"]["strict"], true);
        assert_eq!(schema, &codex);
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["required"],
            json!([
                "is_sufficient",
                "evidence_indices",
                "gaps",
                "new_query",
                "confidence_score"
            ])
        );
        assert_eq!(
            schema["properties"]["evidence_indices"]["items"]["minimum"],
            0
        );
        assert_eq!(schema["properties"]["confidence_score"]["minimum"], 0);
        assert_eq!(schema["properties"]["confidence_score"]["maximum"], 1);
    }

    #[test]
    fn codex_window_contract_keeps_its_compatibility_envelope() {
        let contract = JsonContract::WindowDecisions { count: 3 };
        let schema = contract.codex_output_schema().unwrap();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"]["decisions"]["minItems"], 3);
        assert_eq!(schema["properties"]["decisions"]["maxItems"], 3);

        let response = contract
            .decode_codex_response(
                r#"{"decisions":[{"op":"DROP","compressed_content":"","reason":"irrelevant"},{"op":"DROP","compressed_content":"","reason":"irrelevant"},{"op":"DROP","compressed_content":"","reason":"irrelevant"}]}"#,
            )
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&response).unwrap()[0]["op"],
            "DROP"
        );
    }

    #[test]
    fn zero_window_decisions_is_not_a_valid_contract() {
        let error = JsonContract::WindowDecisions { count: 0 }
            .openai_response_format()
            .unwrap_err();
        assert!(error.to_string().contains("at least one decision"));
    }
}
