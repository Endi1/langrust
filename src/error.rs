use std::error::Error;
use std::fmt;

/// Unified error type for all providers.
///
/// Implemented by hand instead of via `thiserror` to keep the dependency
/// tree minimal; the shape is the same.
#[derive(Debug)]
pub enum LlmError {
    /// The provider returned a non-success HTTP status.
    Http {
        provider: &'static str,
        status: u16,
        body: String,
    },
    /// A request or response body could not be (de)serialized.
    Parse(serde_json::Error),
    /// The HTTP request itself failed (connection, TLS, timeout, ...).
    Transport(reqwest::Error),
    /// Authentication failed (e.g. obtaining a gcloud access token).
    Auth(String),
    /// Anything else (e.g. a well-formed response missing required fields).
    Other(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::Http {
                provider,
                status,
                body,
            } => write!(f, "{} request failed with status {}: {}", provider, status, body),
            LlmError::Parse(e) => write!(f, "failed to parse provider payload: {}", e),
            LlmError::Transport(e) => write!(f, "transport error: {}", e),
            LlmError::Auth(msg) => write!(f, "authentication error: {}", msg),
            LlmError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl Error for LlmError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            LlmError::Parse(e) => Some(e),
            LlmError::Transport(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for LlmError {
    fn from(e: serde_json::Error) -> Self {
        LlmError::Parse(e)
    }
}

impl From<reqwest::Error> for LlmError {
    fn from(e: reqwest::Error) -> Self {
        LlmError::Transport(e)
    }
}

impl From<String> for LlmError {
    fn from(msg: String) -> Self {
        LlmError::Other(msg)
    }
}

impl From<&str> for LlmError {
    fn from(msg: &str) -> Self {
        LlmError::Other(msg.to_string())
    }
}
