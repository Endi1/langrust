mod adapter;
mod types;

#[cfg(test)]
mod tests;

use async_trait::async_trait;

use crate::error::LlmError;
use crate::provider::{Action, LlmClient, Transport};

pub use adapter::ClaudeAdapter;
pub use types::ClaudeModel;

/// Anthropic Messages API client (API-key auth).
pub type ClaudeApiModel = LlmClient<ClaudeAdapter, AnthropicTransport>;

impl ClaudeApiModel {
    pub fn new(api_key: impl Into<String>, model: ClaudeModel) -> Self {
        Self::with_client(api_key, model, reqwest::Client::new())
    }

    pub fn with_client(
        api_key: impl Into<String>,
        model: ClaudeModel,
        client: reqwest::Client,
    ) -> Self {
        LlmClient::from_parts(
            AnthropicTransport {
                api_key: api_key.into(),
                client,
            },
            model.to_string(),
        )
    }
}

pub struct AnthropicTransport {
    pub api_key: String,
    pub client: reqwest::Client,
}

#[async_trait]
impl Transport for AnthropicTransport {
    async fn send(
        &self,
        _model: &str,
        _action: Action,
        body: serde_json::Value,
    ) -> Result<reqwest::Response, LlmError> {
        Ok(self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", self.api_key.clone())
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?)
    }
}
