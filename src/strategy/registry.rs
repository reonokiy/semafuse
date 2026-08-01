use super::{MemoryStrategy, history, lazymem};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::collections::HashMap;

pub(super) type StrategyFactory = fn(Option<&Value>) -> Result<Box<dyn MemoryStrategy>>;

/// One independently configured strategy module registered with the runtime.
#[derive(Clone, Copy)]
pub(super) struct StrategyRegistration {
    pub(super) name: &'static str,
    pub(super) factory: StrategyFactory,
}

impl StrategyRegistration {
    pub(super) const fn new(name: &'static str, factory: StrategyFactory) -> Self {
        Self { name, factory }
    }
}

const REGISTRY: &[StrategyRegistration] = &[history::REGISTRATION, lazymem::REGISTRATION];

fn registration(name: &str) -> Result<&'static StrategyRegistration> {
    REGISTRY
        .iter()
        .find(|registration| registration.name == name)
        .ok_or_else(|| anyhow::anyhow!("unknown memory strategy: {name}"))
}

pub(super) fn build(name: &str, settings: Option<&Value>) -> Result<Box<dyn MemoryStrategy>> {
    let registration = registration(name)?;
    let strategy = (registration.factory)(settings)
        .with_context(|| format!("invalid settings for memory strategy {name}"))?;
    ensure!(
        strategy.name() == registration.name,
        "memory strategy factory {} returned {}",
        registration.name,
        strategy.name()
    );
    Ok(strategy)
}

pub(super) fn validate_settings(settings: &HashMap<String, Value>) -> Result<()> {
    let mut names = settings.keys().map(String::as_str).collect::<Vec<_>>();
    names.sort_unstable();
    for name in names {
        build(name, settings.get(name))?;
    }
    Ok(())
}
