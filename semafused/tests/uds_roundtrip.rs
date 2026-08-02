use std::{fs, os::unix::fs::PermissionsExt, sync::Arc, time::Duration};

use semafuse::{
    daemon_ipc::protocol::ModelIdentity,
    daemon_ipc::{DaemonClient, ModelServiceClient},
};
use semafused::{
    Config, DaemonError, Server, ServiceRouter,
    models::{EmbeddingEngine, ModelEngineError, ModelEngines, RerankerEngine},
    services::ModelsService,
};
use tempfile::TempDir;
use tokio::sync::oneshot;

struct FakeEmbedding {
    identity: ModelIdentity,
}

impl EmbeddingEngine for FakeEmbedding {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn embed(&self, input: &[String]) -> Result<Vec<Vec<f32>>, ModelEngineError> {
        Ok(input
            .iter()
            .map(|text| vec![text.len() as f32, 1.0])
            .collect())
    }
}

struct FakeReranker {
    identity: ModelIdentity,
}

impl RerankerEngine for FakeReranker {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn rerank(&self, _query: &str, documents: &[String]) -> Result<Vec<f64>, ModelEngineError> {
        Ok(documents
            .iter()
            .map(|document| if document == "relevant" { 2.0 } else { -2.0 })
            .collect())
    }
}

#[tokio::test]
async fn shared_clients_round_trip_over_the_private_socket() {
    let temp = TempDir::new().unwrap();
    let runtime = temp.path().join("runtime");
    fs::create_dir(&runtime).unwrap();
    fs::set_permissions(&runtime, fs::Permissions::from_mode(0o700)).unwrap();
    let socket = runtime.join("semafused.sock");

    let embedding_fingerprint = "sha256:test-embedding";
    let reranker_fingerprint = "sha256:test-reranker";
    let engines = ModelEngines {
        embedding: Some(Arc::new(FakeEmbedding {
            identity: ModelIdentity {
                fingerprint: embedding_fingerprint.to_owned(),
                model: "fake-embedding".to_owned(),
                dimensions: Some(2),
            },
        })),
        reranker: Some(Arc::new(FakeReranker {
            identity: ModelIdentity {
                fingerprint: reranker_fingerprint.to_owned(),
                model: "fake-reranker".to_owned(),
                dimensions: None,
            },
        })),
    };
    let mut router = ServiceRouter::new("test-daemon");
    router
        .register(Arc::new(ModelsService::new(engines, 1).unwrap()))
        .unwrap();
    let mut config = Config::default();
    config.daemon.socket_path = socket.clone();
    let server = Server::with_router(config, router).await.unwrap();

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server_task = tokio::spawn(server.run_until(async move {
        shutdown_rx
            .await
            .map_err(|_| DaemonError::Config("test shutdown sender dropped".to_owned()))
    }));

    let status = DaemonClient::status_at(
        socket.clone(),
        Duration::from_secs(1),
        Duration::from_secs(5),
    )
    .await
    .unwrap();
    let models = status.models().unwrap().unwrap();
    assert_eq!(models.embedding.unwrap().fingerprint, embedding_fingerprint);
    assert_eq!(models.reranker.unwrap().fingerprint, reranker_fingerprint);

    let embedding = ModelServiceClient::new(
        socket.clone(),
        embedding_fingerprint,
        Duration::from_secs(1),
        Duration::from_secs(5),
    )
    .embed(&["a".to_owned(), "hello".to_owned()])
    .await
    .unwrap();
    assert_eq!(embedding, vec![vec![1.0, 1.0], vec![5.0, 1.0]]);

    let scores = ModelServiceClient::new(
        socket.clone(),
        reranker_fingerprint,
        Duration::from_secs(1),
        Duration::from_secs(5),
    )
    .rerank("query", &["irrelevant".to_owned(), "relevant".to_owned()])
    .await
    .unwrap();
    assert!(scores[0] < 0.5);
    assert!(scores[1] > 0.5);

    shutdown_tx.send(()).unwrap();
    server_task.await.unwrap().unwrap();
    assert!(!socket.exists());
}
