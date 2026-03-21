use thiserror::Error;

// Define your custom error types

#[derive(Error, Debug)]
pub enum DataLoadingError {
    // We use {0} to print the underlying std::io::Error message
    // The #[from] macro enables the automatic ? conversion
    #[error("Failed to read file: {0}")]
    Io(#[from] std::io::Error),


    #[error("Failed to read file {file_path}: {error}")]
    Parsing {
        file_path: String,
        error: String,
    },
}

#[derive(Error, Debug)]
pub enum DataProcessingError {

    #[error("Failed to write to file: {0}")]
    Io(#[from] std::io::Error),

    #[error("Bincode encoding error: {0}")]
    Bincode(#[from] bincode::Error),

    #[error("Progress bar failed")]
    ProgressBar(#[from] indicatif::style::TemplateError),
}

// // This is how to pass complex values, otherwise (if just a string) use {0}, {1}, etc. Needs
// // to use map_err()
// #[error("Error reading file {file_path}")]
// Syntax {
//     file_path: String,
// },