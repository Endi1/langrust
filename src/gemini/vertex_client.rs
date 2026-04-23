use std::error::Error;

use crate::{
    client::{Completion, Model, ModelRequest, StreamResult},
    gemini::{
        base::GeminiClient,
        gcloud_helpers::get_access_token,
        types::{GeminiModel, GeminiRequest},
    },
};
use async_trait::async_trait;
use reqwest::RequestBuilder;

pub struct GeminiVertexModel {
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
        Ok(response)
    }

    async fn stream_completion(
        &self,
        request: ModelRequest,
    ) -> Result<StreamResult, Box<dyn Error + Send + Sync>> {
        self.stream_generate_content(request).await
    }
}

impl GeminiClient for GeminiVertexModel {
    fn model(&self) -> String {
        self.model.to_string()
    }

    fn get_endpoint(&self, model: &String, method: String) -> String {
        return format!(
            "https://aiplatform.googleapis.com/v1/projects/{}/locations/global/publishers/google/models/{model}:{method}",
            self.project_name
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
