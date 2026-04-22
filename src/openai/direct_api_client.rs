use std::error::Error;

use async_trait::async_trait;
use reqwest::RequestBuilder;

use crate::{
    client::{Completion, Model, ModelRequest, StreamResult},
    openai::{
        base::OpenAiClient,
        types::{OpenAiModel, OpenAiRequest},
    },
};

pub struct OpenAiApiModel {
    pub api_key: String,
    pub client: reqwest::Client,
    pub model: OpenAiModel,
}

#[async_trait]
impl Model for OpenAiApiModel {
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
}

impl OpenAiClient for OpenAiApiModel {
    fn model(&self) -> String {
        self.model.to_string()
    }

    fn get_endpoint(&self) -> String {
        "https://api.openai.com/v1/chat/completions".to_string()
    }

    async fn build_request(
        &self,
        endpoint: &String,
        request_body: &OpenAiRequest,
    ) -> Result<RequestBuilder, Box<dyn Error + Send + Sync>> {
        Ok(self
            .client
            .post(endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request_body))
    }
}
