use std::{fs, path::Path, sync::Arc};

use crate::daemon_ipc::protocol::ModelIdentity;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{
    Config as XlmRobertaConfig, XLMRobertaForSequenceClassification,
};
use fastembed::{Qwen3Config, Qwen3Model, Qwen3TextEmbedding};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokenizers::{PaddingDirection, PaddingStrategy, Tokenizer, TruncationParams};

use crate::daemon::{
    config::{EmbeddingModelConfig, ModelsConfig, RerankerModelConfig},
    models::{EmbeddingEngine, ModelEngineError, ModelEngines, RerankerEngine},
};

const MODEL_WEIGHTS_FILE: &str = "model.safetensors";
const CONFIG_FILE: &str = "config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";

pub(super) fn load_engines(config: &ModelsConfig) -> Result<ModelEngines, String> {
    let embedding = config
        .embedding
        .as_ref()
        .map(QwenEmbeddingEngine::load)
        .transpose()?
        .map(|engine| Arc::new(engine) as Arc<dyn EmbeddingEngine>);
    let reranker = config
        .reranker
        .as_ref()
        .map(BgeRerankerEngine::load)
        .transpose()?
        .map(|engine| Arc::new(engine) as Arc<dyn RerankerEngine>);
    Ok(ModelEngines {
        embedding,
        reranker,
    })
}

struct QwenEmbeddingEngine {
    identity: ModelIdentity,
    model: Qwen3TextEmbedding,
    batch_size: usize,
}

impl QwenEmbeddingEngine {
    fn load(config: &EmbeddingModelConfig) -> Result<Self, String> {
        let config_bytes = read_artifact(&config.model_dir, CONFIG_FILE)?;
        let config_value: Value = serde_json::from_slice(&config_bytes)
            .map_err(|error| format!("invalid Qwen config.json: {error}"))?;
        require_architecture(&config_value, "Qwen3ForCausalLM")?;
        require_model_type(&config_value, "qwen3")?;
        let qwen_config: Qwen3Config = serde_json::from_slice(&config_bytes)
            .map_err(|error| format!("config is not a Qwen3 embedding config: {error}"))?;
        if config.max_length > qwen_config.max_position_embeddings {
            return Err(format!(
                "configured Qwen max_length {} exceeds model max_position_embeddings {}",
                config.max_length, qwen_config.max_position_embeddings
            ));
        }

        let tokenizer_bytes = read_artifact(&config.model_dir, TOKENIZER_FILE)?;
        let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|error| format!("invalid Qwen tokenizer.json: {error}"))?;
        let mut padding = tokenizer.get_padding().cloned().unwrap_or_default();
        padding.strategy = PaddingStrategy::BatchLongest;
        padding.direction = PaddingDirection::Left;
        let _ = tokenizer.with_padding(Some(padding));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_length,
                ..Default::default()
            }))
            .map_err(|error| format!("invalid Qwen tokenizer truncation: {error}"))?;

        let weights = read_artifact(&config.model_dir, MODEL_WEIGHTS_FILE)?;
        let fingerprint = fingerprint(
            "qwen3-fastembed-candle-v1",
            config.max_length,
            [&config_bytes, &tokenizer_bytes, &weights],
        );
        let model_name = model_name(&config_value, "Qwen/Qwen3-Embedding-0.6B");
        let dimensions = qwen_config.hidden_size;
        let device = Device::Cpu;
        let builder = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)
            .map_err(|error| format!("failed to load Qwen safetensors: {error}"))?;
        let model = Qwen3Model::new(qwen_config, builder)
            .map_err(|error| format!("failed to construct Qwen3 embedding model: {error}"))?;
        let engine = Self {
            identity: ModelIdentity {
                fingerprint,
                model: model_name,
                dimensions: Some(dimensions),
            },
            model: Qwen3TextEmbedding::new(model, tokenizer),
            batch_size: config.batch_size,
        };
        engine.run_golden_gate()?;
        Ok(engine)
    }

    fn run_golden_gate(&self) -> Result<(), String> {
        let task = "Given a web search query, retrieve relevant passages that answer the query";
        let texts = vec![
            format!("Instruct: {task}\nQuery:What is the capital of China?"),
            format!("Instruct: {task}\nQuery:Explain gravity"),
            "The capital of China is Beijing.".to_owned(),
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.".to_owned(),
        ];
        let vectors = self
            .embed(&texts)
            .map_err(|error| format!("Qwen startup golden inference failed: {error}"))?;
        if vectors.len() != texts.len() {
            return Err(format!(
                "Qwen startup golden gate returned {} vectors for {} inputs",
                vectors.len(),
                texts.len()
            ));
        }
        let dimensions = vectors[0].len();
        if dimensions == 0
            || vectors.iter().any(|vector| {
                vector.len() != dimensions || vector.iter().any(|value| !value.is_finite())
            })
        {
            return Err(
                "Qwen startup golden gate returned empty, inconsistent, or non-finite vectors"
                    .to_owned(),
            );
        }
        let actual = [
            cosine(&vectors[0], &vectors[2])?,
            cosine(&vectors[0], &vectors[3])?,
            cosine(&vectors[1], &vectors[2])?,
            cosine(&vectors[1], &vectors[3])?,
        ];
        let expected = [0.7646_f32, 0.1414, 0.1355, 0.6000];
        if actual.iter().any(|actual| !actual.is_finite())
            || actual
                .iter()
                .zip(expected)
                .any(|(actual, expected)| (actual - expected).abs() > 0.05)
        {
            return Err(format!(
                "Qwen startup golden gate failed: cosine values were {actual:?}, expected {expected:?} (tolerance 0.05)"
            ));
        }
        Ok(())
    }
}

impl EmbeddingEngine for QwenEmbeddingEngine {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn embed(&self, input: &[String]) -> Result<Vec<Vec<f32>>, ModelEngineError> {
        let mut output = Vec::with_capacity(input.len());
        for batch in input.chunks(self.batch_size) {
            let vectors = self
                .model
                .embed(batch)
                .map_err(|error| ModelEngineError::Inference(error.to_string()))?;
            output.extend(vectors);
        }
        Ok(output)
    }
}

struct BgeRerankerEngine {
    identity: ModelIdentity,
    model: XLMRobertaForSequenceClassification,
    tokenizer: Tokenizer,
    device: Device,
    batch_size: usize,
}

impl BgeRerankerEngine {
    fn load(config: &RerankerModelConfig) -> Result<Self, String> {
        let config_bytes = read_artifact(&config.model_dir, CONFIG_FILE)?;
        let config_value: Value = serde_json::from_slice(&config_bytes)
            .map_err(|error| format!("invalid BGE config.json: {error}"))?;
        require_architecture(&config_value, "XLMRobertaForSequenceClassification")?;
        require_model_type(&config_value, "xlm-roberta")?;
        let labels = config_value
            .get("num_labels")
            .and_then(Value::as_u64)
            .or_else(|| {
                config_value
                    .get("id2label")
                    .and_then(Value::as_object)
                    .map(|labels| labels.len() as u64)
            })
            .unwrap_or(1);
        if labels != 1 {
            return Err(format!(
                "BGE reranker must have exactly one output label, found {labels}"
            ));
        }
        let xlm_config: XlmRobertaConfig = serde_json::from_slice(&config_bytes)
            .map_err(|error| format!("config is not an XLM-R sequence classifier: {error}"))?;
        if config.max_length > xlm_config.max_position_embeddings {
            return Err(format!(
                "configured BGE max_length {} exceeds model max_position_embeddings {}",
                config.max_length, xlm_config.max_position_embeddings
            ));
        }

        let tokenizer_bytes = read_artifact(&config.model_dir, TOKENIZER_FILE)?;
        let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|error| format!("invalid BGE tokenizer.json: {error}"))?;
        let mut padding = tokenizer.get_padding().cloned().unwrap_or_default();
        padding.strategy = PaddingStrategy::BatchLongest;
        padding.direction = PaddingDirection::Right;
        let _ = tokenizer.with_padding(Some(padding));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_length,
                ..Default::default()
            }))
            .map_err(|error| format!("invalid BGE tokenizer truncation: {error}"))?;

        let weights = read_artifact(&config.model_dir, MODEL_WEIGHTS_FILE)?;
        let fingerprint = fingerprint(
            "bge-xlm-roberta-candle-sigmoid-v1",
            config.max_length,
            [&config_bytes, &tokenizer_bytes, &weights],
        );
        let device = Device::Cpu;
        let builder = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)
            .map_err(|error| format!("failed to load BGE safetensors: {error}"))?;
        let model = XLMRobertaForSequenceClassification::new(1, &xlm_config, builder)
            .map_err(|error| format!("failed to construct BGE reranker: {error}"))?;
        let engine = Self {
            identity: ModelIdentity {
                fingerprint,
                model: model_name(&config_value, "BAAI/bge-reranker-v2-m3"),
                dimensions: None,
            },
            model,
            tokenizer,
            device,
            batch_size: config.batch_size,
        };
        engine.run_golden_gate()?;
        Ok(engine)
    }

    fn run_golden_gate(&self) -> Result<(), String> {
        let documents = vec![
            "hi".to_owned(),
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.".to_owned(),
        ];
        let logits = self
            .rerank("what is panda?", &documents)
            .map_err(|error| format!("BGE startup golden inference failed: {error}"))?;
        if logits.len() != documents.len() {
            return Err(format!(
                "BGE startup golden gate returned {} logits for {} documents",
                logits.len(),
                documents.len()
            ));
        }
        let expected = [-8.1875_f64, 5.26171875];
        if logits.iter().any(|actual| !actual.is_finite())
            || logits
                .iter()
                .zip(expected)
                .any(|(actual, expected)| (actual - expected).abs() > 0.25)
        {
            return Err(format!(
                "BGE startup golden gate failed: raw logits were {logits:?}, expected {expected:?} (tolerance 0.25)"
            ));
        }
        Ok(())
    }
}

impl RerankerEngine for BgeRerankerEngine {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>, ModelEngineError> {
        let mut output = Vec::with_capacity(documents.len());
        for batch in documents.chunks(self.batch_size) {
            let pairs: Vec<_> = batch
                .iter()
                .map(|document| (query, document.as_str()))
                .collect();
            let encodings = self
                .tokenizer
                .encode_batch(pairs, true)
                .map_err(|error| ModelEngineError::InvalidInput(error.to_string()))?;
            let sequence_length = encodings
                .first()
                .ok_or_else(|| ModelEngineError::InvalidInput("empty tokenizer batch".to_owned()))?
                .len();
            let batch_size = encodings.len();
            let mut input_ids = Vec::with_capacity(batch_size * sequence_length);
            let mut attention_mask = Vec::with_capacity(batch_size * sequence_length);
            let mut token_type_ids = Vec::with_capacity(batch_size * sequence_length);
            for encoding in encodings {
                input_ids.extend_from_slice(encoding.get_ids());
                attention_mask.extend_from_slice(encoding.get_attention_mask());
                token_type_ids.extend_from_slice(encoding.get_type_ids());
            }
            let input_ids =
                Tensor::from_vec(input_ids, (batch_size, sequence_length), &self.device)
                    .map_err(candle_inference_error)?;
            let attention_mask =
                Tensor::from_vec(attention_mask, (batch_size, sequence_length), &self.device)
                    .map_err(candle_inference_error)?;
            let token_type_ids =
                Tensor::from_vec(token_type_ids, (batch_size, sequence_length), &self.device)
                    .map_err(candle_inference_error)?;
            let logits = self
                .model
                .forward(&input_ids, &attention_mask, &token_type_ids)
                .and_then(|tensor| tensor.to_dtype(DType::F32))
                .and_then(|tensor| tensor.to_vec2::<f32>())
                .map_err(candle_inference_error)?;
            for row in logits {
                let logit = row.first().ok_or_else(|| {
                    ModelEngineError::Inference("reranker returned an empty logit row".to_owned())
                })?;
                output.push(f64::from(*logit));
            }
        }
        Ok(output)
    }
}

fn read_artifact(model_dir: &Path, name: &str) -> Result<Vec<u8>, String> {
    let path = model_dir.join(name);
    fs::read(&path).map_err(|error| format!("failed to read {}: {error}", path.display()))
}

fn require_architecture(config: &Value, expected: &str) -> Result<(), String> {
    let matches = config
        .get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|architectures| {
            architectures
                .iter()
                .any(|architecture| architecture.as_str() == Some(expected))
        });
    if !matches {
        return Err(format!(
            "model config must declare architecture {expected:?}"
        ));
    }
    Ok(())
}

fn require_model_type(config: &Value, expected: &str) -> Result<(), String> {
    let actual = config.get("model_type").and_then(Value::as_str);
    if actual != Some(expected) {
        return Err(format!(
            "model config has model_type {actual:?}; expected {expected:?}"
        ));
    }
    Ok(())
}

fn model_name(config: &Value, fallback: &str) -> String {
    config
        .get("_name_or_path")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .unwrap_or(fallback)
        .to_owned()
}

fn fingerprint<const N: usize>(
    implementation: &str,
    max_length: usize,
    parts: [&Vec<u8>; N],
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"semafused-model-fingerprint-v1\0");
    digest.update(implementation.as_bytes());
    digest.update([0]);
    digest.update(max_length.to_be_bytes());
    for part in parts {
        digest.update((part.len() as u64).to_be_bytes());
        digest.update(part);
    }
    format!("sha256:{}", hex::encode(digest.finalize()))
}

fn candle_inference_error(error: candle_core::Error) -> ModelEngineError {
    ModelEngineError::Inference(error.to_string())
}

fn cosine(left: &[f32], right: &[f32]) -> Result<f32, String> {
    if left.is_empty() || right.is_empty() {
        return Err("cannot compute cosine similarity for an empty vector".to_owned());
    }
    if left.len() != right.len() {
        return Err(format!(
            "cannot compute cosine similarity for dimensions {} and {}",
            left.len(),
            right.len()
        ));
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return Err("cannot compute cosine similarity with non-finite values".to_owned());
    }
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !dot.is_finite()
        || !left_norm.is_finite()
        || !right_norm.is_finite()
        || left_norm == 0.0
        || right_norm == 0.0
    {
        return Err("cannot compute cosine similarity with zero or non-finite norm".to_owned());
    }
    let similarity = dot / (left_norm * right_norm);
    if !similarity.is_finite() {
        return Err("cosine similarity was non-finite".to_owned());
    }
    Ok(similarity)
}
