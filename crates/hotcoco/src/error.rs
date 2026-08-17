use std::io;

use crate::convert::ConvertError;

/// Unified error type for hotcoco operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// I/O error (file open, read, write).
    #[error(transparent)]
    Io(#[from] io::Error),

    /// JSON serialization or deserialization error (serde_json paths: saving,
    /// report emission).
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// JSON parse error from the simd-json loader — the main dataset-loading
    /// path (`COCO::new`, `COCO::load_res`).
    #[error("invalid JSON: {0}")]
    JsonParse(#[from] simd_json::Error),

    /// Format conversion error (COCO ↔ YOLO).
    #[error(transparent)]
    Convert(#[from] ConvertError),

    /// Any other error with a human-readable message.
    #[error("{0}")]
    Other(String),
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Other(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Other(s.to_string())
    }
}

/// Convenience alias used throughout hotcoco.
pub type Result<T> = std::result::Result<T, Error>;
