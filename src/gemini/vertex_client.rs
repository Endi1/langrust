use std::error::Error;

use crate::{
    client::{ChatMessage, Client, CompletionWrapper, LLMCallSettings},
    gemini::{
        base::GeminiClient,
        gcloud_helpers::get_access_token,
        types::{GeminiCompletion, GeminiRequest},
    },
};
use async_trait::async_trait;
use reqwest::RequestBuilder;

pub struct GeminiVertexClient {
    pub region: String,
    pub project_name: String,
    pub client: reqwest::Client,
}

#[async_trait]
impl Client for GeminiVertexClient {
    async fn chat_completion(
        &self,
        system_message: &Option<String>,
        messages: &Vec<ChatMessage>,
        llm_call_settings: &LLMCallSettings,
    ) -> Result<Box<dyn CompletionWrapper>, Box<dyn Error + Send + Sync>> {
        let response = self
            .generate_content(system_message, messages, llm_call_settings)
            .await?;
        return Ok(Box::new(GeminiCompletion {
            content: response.content,
            completion_tokens: response.completion_tokens,
            prompt_tokens: response.prompt_tokens,
        }));
    }
}

impl GeminiClient for GeminiVertexClient {
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
