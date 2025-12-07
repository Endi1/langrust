use std::error::Error;

use crate::{
    client::{Completion, Model, ModelRequest},
    gemini::{base::GeminiClient, types::GeminiRequest},
};
use async_trait::async_trait;
use reqwest::RequestBuilder;

pub struct GeminiApiModel {
    pub api_key: String,
    pub client: reqwest::Client,
    pub model: String, // TODO Replace this with a type
}

#[async_trait]
impl Model for GeminiApiModel {
    async fn completion(
        &self,
        request: ModelRequest,
    ) -> Result<Completion, Box<dyn Error + Send + Sync>> {
        let response = self.generate_content(request).await?;
        return Ok(Completion {
            completion: response.completion,
            completion_tokens: response.completion_tokens,
            prompt_tokens: response.prompt_tokens,
        });
    }
}

impl GeminiClient for GeminiApiModel {
    fn model(&self) -> &String {
        &self.model
    }
    fn get_endpoint(&self, model: &String, method: String) -> String {
        return format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:{}",
            model, method
        );
    }

    async fn build_request(
        &self,
        endpoint: &String,
        request_body: &GeminiRequest,
    ) -> Result<RequestBuilder, Box<dyn Error + Send + Sync>> {
        return Ok(self
            .client
            .post(endpoint.clone())
            .header("x-goog-api-key", self.api_key.clone())
            .header("Content-Type", "application/json")
            .json(request_body));
    }
}
