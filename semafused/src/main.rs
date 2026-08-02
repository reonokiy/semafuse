use std::{path::PathBuf, process::ExitCode};

use clap::Parser;
use semafused::{Config, Server};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "semafused",
    version,
    about = "Semafuse foreground local service daemon"
)]
struct Args {
    /// Strict TOML configuration file.
    #[arg(long, value_name = "PATH")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let args = Args::parse();
    match run(args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            error!(%error, "semafused stopped with an error");
            ExitCode::FAILURE
        }
    }
}

async fn run(args: Args) -> semafused::Result<()> {
    let config = Config::load(&args.config)?;
    let server = Server::from_config(config).await?;
    info!(socket = %server.socket_path().display(), "starting semafused");
    server.run_until_shutdown_signal().await
}
