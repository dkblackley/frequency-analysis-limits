use thiserror::Error;

#[derive(Error, Debug)]
pub enum LAMAError {
    #[error("Failed to write to file: {0}")]
    Io(#[from] std::io::Error),

    #[error("Bincode encoding error: {0}")]
    Bincode(#[from] bincode::Error),

    #[error("Progress bar failed")]
    ProgressBar(#[from] indicatif::style::TemplateError),

    #[error("Serde Json Error: {0}")]
    SerdeJson(#[from] serde_json::Error),
}
