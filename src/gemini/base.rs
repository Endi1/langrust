use futures::TryFutureExt;
use std::error::Error;

use reqwest::RequestBuilder;

use crate::{
    client::{ChatMessage, Settings, Role},
    gemini::types::{
        Content, GeminiCompletion, GeminiRequest, GeminiResponse, GenerationConfig, Part,
        SystemInstructionContent, ThinkingConfig,
    },
};

pub trait GeminiClient {
    fn create_request_body(
        &self,
        system_message: &Option<String>,
        messages: &Vec<ChatMessage>,
        llm_call_settings: &Settings,
    ) -> GeminiRequest {
        let thinking_config = if !llm_call_settings.model.contains("1.5")
            && !llm_call_settings.model.contains("2.0")
        {
            Some(ThinkingConfig {
                thinking_budget: llm_call_settings.thinking_budget.unwrap_or_default(),
            })
        } else {
            None
        };

        let generation_config = GenerationConfig {
            max_output_tokens: llm_call_settings.max_tokens.unwrap_or_default(),
            temperature: llm_call_settings.temperature,
            thinking_config,
        };

        let contents: Vec<Content> = messages
            .iter()
            .map(|message| Content {
                parts: Vec::from([Part {
                    text: message.content.clone(),
                }]),
                role: message.role.clone().unwrap_or_else(|| Role::User),
            })
            .collect();

        let system_instruction = system_message.clone().map(|m| SystemInstructionContent {
            parts: vec![Part { text: m }],
        });

        GeminiRequest {
            system_instruction,
            contents,
            generation_config,
        }
    }

    async fn generate_content(
        &self,
        system_message: &Option<String>,
        messages: &Vec<ChatMessage>,
        llm_call_settings: &Settings,
    ) -> Result<GeminiCompletion, Box<dyn Error + Send + Sync>> {
        let endpoint = self.get_endpoint(&llm_call_settings.model, String::from("generateContent"));
        let request_body = self.create_request_body(system_message, messages, llm_call_settings);
        let response = self
            .build_request(&endpoint, &request_body)
            .await?
            .send()
            .map_err(|e| e.to_string())
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().map_err(|e| e.to_string()).await?;
            return Err(format!(
                "Gemini request failed with status {}: {}",
                status, error_text
            )
            .into());
        }

        let response_body: GeminiResponse = response.json().map_err(|e| e.to_string()).await?;
        return Ok(GeminiCompletion {
            content: response_body.get_text(),
            prompt_tokens: response_body
                .usage_metadata
                .as_ref()
                .and_then(|m| m.candidates_token_count),
            completion_tokens: response_body
                .usage_metadata
                .as_ref()
                .and_then(|m| m.candidates_token_count),
        });
    }

    fn get_endpoint(&self, model: &String, method: String) -> String;
    async fn build_request(
        &self,
        endpoint: &String,
        request_body: &GeminiRequest,
    ) -> Result<RequestBuilder, Box<dyn Error + Send + Sync>>;
}
