use crate::client::Role;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeminiModel {
    Gemini25Flash,
}

impl GeminiModel {
    pub fn to_string(&self) -> String {
        match self {
            GeminiModel::Gemini25Flash => "gemini-2.5-flash".to_string(),
        }
    }
}

#[derive(Serialize)]
pub struct ThinkingConfig {
    #[serde(rename = "thinkingBudget")]
    pub thinking_budget: i16,
}

#[derive(Serialize)]
pub struct GenerationConfig {
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i16>,
    pub temperature: i16,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

#[derive(Serialize)]
pub struct Part {
    pub text: String,
}

#[derive(Serialize)]
pub struct Content {
    pub parts: Vec<Part>,
    pub role: Role,
}

#[derive(Serialize)]
pub struct SystemInstructionContent {
    pub parts: Vec<Part>,
}

#[derive(Serialize)]
pub struct GeminiRequest {
    pub system_instruction: Option<SystemInstructionContent>,
    pub contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    pub generation_config: GenerationConfig, // TODO implement safetySettings
                                             // TODO implement image stuff
}

#[derive(Debug, Deserialize)]
pub struct GeminiResponse {
    pub candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<UsageMetadata>,
}

impl GeminiResponse {
    pub fn get_text(&self) -> Option<String> {
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

    pub fn get_prompt_tokens(&self) -> Option<i32> {
        self.usage_metadata
            .as_ref()
            .and_then(|m| m.prompt_token_count)
    }

    pub fn get_completion_tokens(&self) -> Option<i32> {
        self.usage_metadata
            .as_ref()
            .and_then(|m| m.candidates_token_count)
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
pub struct GeminiCompletion {
    pub content: String,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
}
