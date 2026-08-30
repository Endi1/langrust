mod adapter;
mod gcloud_helpers;
mod types;

#[cfg(test)]
mod tests;

use async_trait::async_trait;

use crate::error::LlmError;
use crate::provider::{Action, LlmClient, Transport};
use gcloud_helpers::get_access_token;

pub use adapter::GeminiAdapter;
pub use types::GeminiModel;

pub type GeminiApiModel = LlmClient<GeminiAdapter, GeminiApiTransport>;

pub type GeminiVertexModel = LlmClient<GeminiAdapter, VertexTransport>;

impl GeminiApiModel {
    pub fn new(api_key: impl Into<String>, model: GeminiModel) -> Self {
        Self::with_client(api_key, model, reqwest::Client::new())
    }

    pub fn with_client(
        api_key: impl Into<String>,
        model: GeminiModel,
        client: reqwest::Client,
    ) -> Self {
        LlmClient::from_parts(
            GeminiApiTransport {
                api_key: api_key.into(),
                client,
            },
            model.to_string(),
        )
    }
}

impl GeminiVertexModel {
    pub fn new(project_name: impl Into<String>, model: GeminiModel) -> Self {
        Self::with_client(project_name, model, reqwest::Client::new())
    }

    pub fn with_client(
        project_name: impl Into<String>,
        model: GeminiModel,
        client: reqwest::Client,
    ) -> Self {
        LlmClient::from_parts(
            VertexTransport {
                project_name: project_name.into(),
                client,
            },
            model.to_string(),
        )
    }
}

fn method(action: Action) -> &'static str {
    match action {
        Action::Generate => "generateContent",
        Action::Stream => "streamGenerateContent?alt=sse",
    }
}

pub struct GeminiApiTransport {
    pub api_key: String,
    pub client: reqwest::Client,
}

#[async_trait]
impl Transport for GeminiApiTransport {
    async fn send(
        &self,
        model: &str,
        action: Action,
        body: serde_json::Value,
    ) -> Result<reqwest::Response, LlmError> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:{}",
            model,
            method(action)
        );
        Ok(self
            .client
            .post(url)
            .header("x-goog-api-key", self.api_key.clone())
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?)
    }
}

pub struct VertexTransport {
    pub project_name: String,
    pub client: reqwest::Client,
}

#[async_trait]
impl Transport for VertexTransport {
    async fn send(
        &self,
        model: &str,
        action: Action,
        body: serde_json::Value,
    ) -> Result<reqwest::Response, LlmError> {
        let url = format!(
            "https://aiplatform.googleapis.com/v1/projects/{}/locations/global/publishers/google/models/{}:{}",
            self.project_name,
            model,
            method(action)
        );
        let access_token = get_access_token().await.map_err(LlmError::Auth)?;
        Ok(self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", access_token))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?)
    }
}
