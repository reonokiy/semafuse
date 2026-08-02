use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::Arc,
};

use async_trait::async_trait;
use semafuse::daemon_ipc::protocol::{
    DaemonRequest, DaemonResponse, DaemonState, DaemonStatus, ErrorCode, ModelRequest,
    ProtocolError, RequestEnvelope, ResponseEnvelope, ServiceResponse, ServiceState,
};
use serde_json::Value;

use crate::error::{DaemonError, Result};

#[derive(Debug)]
pub enum ServiceRequest {
    Models(ModelRequest),
}

#[async_trait]
pub trait Service: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn state(&self) -> ServiceState;
    fn status(&self) -> Value;

    async fn call(&self, request: ServiceRequest) -> Result<ServiceResponse, ProtocolError>;
}

/// Protocol router and service registry. The daemon status endpoint is built
/// in; all workload services are registered independently.
pub struct ServiceRouter {
    daemon_version: String,
    services: BTreeMap<String, Arc<dyn Service>>,
}

impl ServiceRouter {
    pub fn new(daemon_version: impl Into<String>) -> Self {
        Self {
            daemon_version: daemon_version.into(),
            services: BTreeMap::new(),
        }
    }

    pub fn register(&mut self, service: Arc<dyn Service>) -> Result<()> {
        let name = service.name();
        if name.is_empty()
            || name.len() > 64
            || !name.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
            })
            || name == "daemon"
        {
            return Err(DaemonError::ServiceRegistration(format!(
                "service name {name:?} must contain 1..=64 ASCII lowercase letters, digits, '_' or '-', and must not be reserved"
            )));
        }

        match self.services.entry(name.to_owned()) {
            Entry::Vacant(entry) => {
                entry.insert(service);
                Ok(())
            }
            Entry::Occupied(_) => Err(DaemonError::ServiceRegistration(format!(
                "duplicate service {name:?}"
            ))),
        }
    }

    pub fn status(&self) -> DaemonStatus {
        let mut state = DaemonState::Ready;
        let services = self
            .services
            .iter()
            .map(|(name, service)| {
                if service.state() != ServiceState::Ready {
                    state = DaemonState::Degraded;
                }
                (name.clone(), service.status())
            })
            .collect();

        DaemonStatus {
            state,
            daemon_version: self.daemon_version.clone(),
            services,
        }
    }

    pub async fn route(&self, request: RequestEnvelope) -> ResponseEnvelope {
        let result = match request {
            RequestEnvelope::Daemon(DaemonRequest::Status {}) => {
                Ok(ServiceResponse::Daemon(DaemonResponse::Status {
                    status: self.status(),
                }))
            }
            RequestEnvelope::Models(request) => match self.services.get("models") {
                Some(service) => service.call(ServiceRequest::Models(request)).await,
                None => Err(protocol_error(
                    ErrorCode::NotReady,
                    "models service is not configured",
                    false,
                )),
            },
        };

        match result {
            Ok(response) => ResponseEnvelope::Ok { response },
            Err(error) => ResponseEnvelope::Error { error },
        }
    }
}

pub(crate) fn protocol_error(
    code: ErrorCode,
    message: impl Into<String>,
    retryable: bool,
) -> ProtocolError {
    use semafuse::daemon_ipc::protocol::MAX_ERROR_MESSAGE_BYTES;

    let mut message = message.into();
    if message.len() > MAX_ERROR_MESSAGE_BYTES {
        let mut boundary = MAX_ERROR_MESSAGE_BYTES;
        while boundary > 0 && !message.is_char_boundary(boundary) {
            boundary -= 1;
        }
        message.truncate(boundary);
    }
    ProtocolError {
        code,
        message,
        retryable,
        retry_after_ms: retryable.then_some(10),
    }
}

#[cfg(test)]
mod tests {
    use semafuse::daemon_ipc::protocol::{ModelRequest, ResponseEnvelope};

    use super::*;

    struct StubService {
        name: &'static str,
        marker: &'static str,
    }

    #[async_trait]
    impl Service for StubService {
        fn name(&self) -> &'static str {
            self.name
        }

        fn state(&self) -> ServiceState {
            ServiceState::Ready
        }

        fn status(&self) -> Value {
            serde_json::json!({"marker": self.marker})
        }

        async fn call(&self, _request: ServiceRequest) -> Result<ServiceResponse, ProtocolError> {
            unreachable!("stub service must not receive requests")
        }
    }

    fn stub(name: &'static str, marker: &'static str) -> Arc<dyn Service> {
        Arc::new(StubService { name, marker })
    }

    #[tokio::test]
    async fn daemon_status_does_not_require_a_workload_service() {
        let router = ServiceRouter::new("test-version");
        let response = router
            .route(RequestEnvelope::Daemon(DaemonRequest::Status {}))
            .await;
        let ResponseEnvelope::Ok {
            response: ServiceResponse::Daemon(DaemonResponse::Status { status }),
        } = response
        else {
            panic!("unexpected response");
        };
        assert_eq!(status.state, DaemonState::Ready);
        assert_eq!(status.daemon_version, "test-version");
        assert!(status.services.is_empty());
    }

    #[tokio::test]
    async fn unconfigured_models_service_is_not_ready() {
        let response = ServiceRouter::new("test")
            .route(RequestEnvelope::Models(ModelRequest::Embed {
                expected_model_fingerprint: "sha256:test".to_owned(),
                input: vec!["hello".to_owned()],
            }))
            .await;
        let ResponseEnvelope::Error { error } = response else {
            panic!("expected error");
        };
        assert_eq!(error.code, ErrorCode::NotReady);
    }

    #[test]
    fn service_names_use_a_small_ascii_namespace() {
        for name in [
            "",
            "daemon",
            "Models",
            "model.service",
            "model/service",
            "模型",
        ] {
            let mut router = ServiceRouter::new("test");
            assert!(router.register(stub(name, "invalid")).is_err(), "{name:?}");
        }

        let long_name = Box::leak("a".repeat(65).into_boxed_str());
        let mut router = ServiceRouter::new("test");
        assert!(router.register(stub(long_name, "invalid")).is_err());

        let mut router = ServiceRouter::new("test");
        router.register(stub("model-service_2", "valid")).unwrap();
    }

    #[test]
    fn duplicate_registration_does_not_replace_the_original_service() {
        let mut router = ServiceRouter::new("test");
        router.register(stub("models", "first")).unwrap();

        let error = router.register(stub("models", "second")).unwrap_err();
        assert!(matches!(error, DaemonError::ServiceRegistration(_)));
        assert_eq!(router.status().services["models"]["marker"], "first");
    }
}
