use std::error::Error;

use crate::{
    client::{Completion, Model, ModelRequest},
    gemini::{
        base::GeminiClient,
        gcloud_helpers::get_access_token,
        types::{GeminiModel, GeminiRequest},
    },
};
use async_trait::async_trait;
use reqwest::RequestBuilder;

pub struct GeminiVertexModel {
    pub region: String,
    pub project_name: String,
    pub client: reqwest::Client,
    pub model: GeminiModel,
}

#[async_trait]
impl Model for GeminiVertexModel {
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

impl GeminiClient for GeminiVertexModel {
    fn model(&self) -> String {
        self.model.to_string()
    }

    fn get_endpoint(&self, model: &String, method: String) -> String {
        return format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{model}:{method}",
            self.region, self.project_name, self.region
        );
    }

    async fn build_request(
        &self,
        endpoint: &String,
        request_body: &GeminiRequest,
    ) -> Result<RequestBuilder, Box<dyn Error + Send + Sync>> {
        let access_token = get_access_token().await?;
        return Ok(self
            .client
            .post(endpoint)
            .header("Authorization", format!("Bearer {}", access_token))
            .header("Content-Type", "application/json")
            .json(request_body));
    }
}
