use std::error::Error;

use async_trait::async_trait;
use reqwest::RequestBuilder;

use crate::{
    claude::{
        base::ClaudeClient,
        types::{ClaudeModel, ClaudeRequest},
    },
    client::{Completion, Model, ModelRequest, StreamResult},
};

pub struct ClaudeApiModel {
    pub api_key: String,
    pub client: reqwest::Client,
    pub model: ClaudeModel,
}

#[async_trait]
impl Model for ClaudeApiModel {
    async fn completion(
        &self,
        request: ModelRequest,
    ) -> Result<Completion, Box<dyn Error + Send + Sync>> {
        self.generate_content(request).await
    }

    async fn stream_completion(
        &self,
        request: ModelRequest,
    ) -> Result<StreamResult, Box<dyn Error + Send + Sync>> {
        self.stream_generate_content(request).await
    }

    fn model_name(&self) -> String {
        self.model.to_string()
    }
}

impl ClaudeClient for ClaudeApiModel {
    fn get_endpoint(&self) -> String {
        "https://api.anthropic.com/v1/messages".to_string()
    }

    async fn build_request(
        &self,
        endpoint: &String,
        request_body: &ClaudeRequest,
    ) -> Result<RequestBuilder, Box<dyn Error + Send + Sync>> {
        Ok(self
            .client
            .post(endpoint)
            .header("x-api-key", self.api_key.clone())
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(request_body))
    }
}
