pub mod error;
pub mod http;
pub mod provider;
pub mod providers;
pub mod request;
pub mod types;

#[cfg(test)]
mod tests;

pub use error::LlmError;
pub use provider::{Action, LlmClient, ProviderAdapter, Transport};
pub use providers::claude::{ClaudeApiModel, ClaudeModel};
pub use providers::gemini::{GeminiApiModel, GeminiModel, GeminiVertexModel};
pub use providers::openai::{OpenAiApiModel, OpenAiModel};
pub use request::{Model, ModelRequest, ModelRequestBuilder};
pub use types::{
    Completion, FunctionCall, Message, MessageType, Role, Settings, StreamEvent, StreamResult,
    Tool, ToolParameters, Usage,
};
