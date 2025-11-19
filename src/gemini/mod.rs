use std::error::Error;

use crate::{
    client::{ChatMessage, Client, CompletionWrapper, LLMCallSettings, Role},
    gemini::gcloud_helpers::get_access_token,
};
use async_trait::async_trait;
use futures::TryFutureExt;
use serde::{Deserialize, Serialize};

mod gcloud_helpers;

pub struct VertexGeminiClient {
    region: String,
    project_name: String,
    client: reqwest::Client,
}

pub struct GeminiClient {
    api_key: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct ThinkingConfig {
    #[serde(rename = "thinkingBudget")]
    thinking_budget: i16,
}

#[derive(Serialize)]
struct GenerationConfig {
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: i16,
    temperature: i16,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    thinking_config: Option<ThinkingConfig>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
    role: Role,
}

#[derive(Serialize)]
struct SystemInstructionContent {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct GeminiRequest {
    system_instruction: Option<SystemInstructionContent>,
    contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    generation_config: GenerationConfig, // TODO implement safetySettings
                                         // TODO implement image stuff
}

#[derive(Debug, Deserialize)]
pub struct GeminiResponse {
    pub candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<UsageMetadata>,
}

impl GeminiResponse {
    fn get_text(&self) -> Option<String> {
        if self.candidates.is_empty() {
            return None;
        }

        let mut response_text = String::new();
        let candidate = &self.candidates[0];
        for part in &candidate.content.parts {
            response_text.push_str(&part.text);
        }
        return Some(response_text);
    }
}

#[derive(Debug, Deserialize)]
pub struct Candidate {
    pub content: ResponseContent,
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
    pub index: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseContent {
    pub parts: Vec<ResponsePart>,
    pub role: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsePart {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: Option<i32>,
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<i32>,
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: Option<i32>,
}

#[derive(Debug)]
struct GeminiCompletion {
    content: Option<String>,
    prompt_tokens: Option<i32>,
    completion_tokens: Option<i32>,
}

impl CompletionWrapper for GeminiCompletion {
    fn completion(&self) -> Option<String> {
        self.content.clone()
    }
    fn prompt_tokens(&self) -> Option<i32> {
        self.prompt_tokens
    }
    fn completion_tokens(&self) -> Option<i32> {
        self.completion_tokens
    }
}

#[async_trait]
impl Client for VertexGeminiClient {
    async fn chat_completion(
        &self,
        system_message: &Option<String>,
        messages: &Vec<ChatMessage>,
        llm_call_settings: &LLMCallSettings,
    ) -> Result<Box<dyn CompletionWrapper>, Box<dyn Error>> {
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

#[async_trait]
impl Client for GeminiClient {
    async fn chat_completion(
        &self,
        system_message: &Option<String>,
        messages: &Vec<ChatMessage>,
        llm_call_settings: &LLMCallSettings,
    ) -> Result<Box<dyn CompletionWrapper>, Box<dyn Error>> {
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

impl VertexGeminiClient {
    async fn generate_content(
        &self,
        system_message: &Option<String>,
        messages: &Vec<ChatMessage>,
        llm_call_settings: &LLMCallSettings,
    ) -> Result<GeminiCompletion, String> {
        let endpoint = self.get_endpoint(&llm_call_settings.model, String::from("generateContent"));
        let access_token = get_access_token().await?;
        let request_body = self.create_request_body(system_message, messages, llm_call_settings);
        let response = self
            .client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {}", access_token))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .map_err(|e| e.to_string())
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().map_err(|e| e.to_string()).await?;
            return Err(format!(
                "Gemini request failed with status {}: {}",
                status, error_text
            ));
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

    fn get_endpoint(&self, model: &String, method: String) -> String {
        return format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/us-central1/publishers/google/models/{model}:{method}",
            self.region, self.project_name
        );
    }

    fn create_request_body(
        &self,
        system_message: &Option<String>,
        messages: &Vec<ChatMessage>,
        llm_call_settings: &LLMCallSettings,
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
}

impl GeminiClient {
    async fn generate_content(
        &self,
        system_message: &Option<String>,
        messages: &Vec<ChatMessage>,
        llm_call_settings: &LLMCallSettings,
    ) -> Result<GeminiCompletion, String> {
        let endpoint = self.get_endpoint(&llm_call_settings.model, String::from("generateContent"));
        let request_body = self.create_request_body(system_message, messages, llm_call_settings);
        let response = self
            .client
            .post(&endpoint)
            .header("x-goog-api-key", self.api_key.clone())
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .map_err(|e| e.to_string())
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().map_err(|e| e.to_string()).await?;
            return Err(format!(
                "Gemini request failed with status {}: {}",
                status, error_text
            ));
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

    fn get_endpoint(&self, model: &String, method: String) -> String {
        return format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:{}",
            model, method
        );
    }

    fn create_request_body(
        &self,
        system_message: &Option<String>,
        messages: &Vec<ChatMessage>,
        llm_call_settings: &LLMCallSettings,
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
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;

    #[tokio::test]
    async fn test_generate_content_vertex() {
        let gemini_client = VertexGeminiClient {
            region: env::var("VERTEX_REGION").unwrap(),
            project_name: env::var("VERTEX_PROJECT").unwrap(),
            client: reqwest::Client::new(),
        };
        let messages = [ChatMessage {
            content: "hello how are you?".to_string(),
            role: Some(Role::User),
        }]
        .to_vec();
        let call_settings = LLMCallSettings {
            model: "gemini-2.5-flash".to_string(),
            max_tokens: Some(8000),
            timeout: None,
            temperature: 0,
            thinking_budget: Some(0),
        };
        let response = gemini_client
            .complete(&None, &messages, &call_settings)
            .await;
        assert!(response.is_ok());

        let completion = response.expect("No completion found").completion();
        assert!(completion.is_some());
    }

    #[tokio::test]
    async fn test_generate_content_direct() {
        let gemini_client = GeminiClient {
            client: reqwest::Client::new(),
            api_key: env::var("GEMINI_KEY").unwrap(),
        };
        let messages = [ChatMessage {
            content: "hello how are you?".to_string(),
            role: Some(Role::User),
        }]
        .to_vec();
        let call_settings = LLMCallSettings {
            model: "gemini-2.5-flash".to_string(),
            max_tokens: Some(8000),
            timeout: None,
            temperature: 0,
            thinking_budget: Some(0),
        };
        let system_message = Some("you are a helpful assistant".to_string());
        let response = gemini_client
            .complete(&system_message, &messages, &call_settings)
            .await;
        println!("{:?}", response);
        assert!(response.is_ok());

        let completion = response.expect("No completion found").completion();
        assert!(completion.is_some());
    }
}
