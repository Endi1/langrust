use std::error::Error;

use crate::{
    client::{ChatMessage, Client, CompletionWrapper, LLMCallSettings, Role},
    gemini::{
        base::GeminiClient,
        gcloud_helpers::get_access_token,
        types::{GeminiCompletion, GeminiRequest},
    },
};
use async_trait::async_trait;
use reqwest::RequestBuilder;

pub struct GeminiApiClient {
    pub api_key: String,
    pub client: reqwest::Client,
}

#[async_trait]
impl Client for GeminiApiClient {
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

impl GeminiClient for GeminiApiClient {
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
