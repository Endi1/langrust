mod adapter;
mod types;

#[cfg(test)]
mod tests;

use async_trait::async_trait;

use crate::provider::{Action, BoxError, LlmClient, Transport};

pub use adapter::OpenAiAdapter;
pub use types::OpenAiModel;

pub type OpenAiApiModel = LlmClient<OpenAiAdapter, OpenAiTransport>;

impl OpenAiApiModel {
    pub fn new(api_key: impl Into<String>, model: OpenAiModel) -> Self {
        Self::with_client(api_key, model, reqwest::Client::new())
    }

    pub fn with_client(
        api_key: impl Into<String>,
        model: OpenAiModel,
        client: reqwest::Client,
    ) -> Self {
        LlmClient::from_parts(
            OpenAiTransport {
                api_key: api_key.into(),
                client,
            },
            model.to_string(),
        )
    }
}

pub struct OpenAiTransport {
    pub api_key: String,
    pub client: reqwest::Client,
}

#[async_trait]
impl Transport for OpenAiTransport {
    async fn send(
        &self,
        _model: &str,
        _action: Action,
        body: serde_json::Value,
    ) -> Result<reqwest::Response, BoxError> {
        Ok(self
            .client
            .post("https://api.openai.com/v1/responses")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?)
    }
}
