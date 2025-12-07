use std::collections::HashMap;

use crate::client::{Role, Tool};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
pub struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<Tool>,
}

#[derive(Serialize)]
pub struct GeminiRequest {
    pub system_instruction: Option<SystemInstructionContent>,
    pub contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    pub generation_config: GenerationConfig, // TODO implement safetySettings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiResponse {
    pub candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<UsageMetadata>,
}

impl GeminiResponse {
    pub fn get_function(&self) -> Option<GeminiFunction> {
        if self.candidates.is_empty() {
            return None;
        }

        let candidate = &self.candidates[0];
        let mut function_call: Option<GeminiFunction> = None;
        for part in &candidate.content.parts {
            if part.function_call.is_some() {
                function_call = part.function_call.clone();
            }
        }
        return function_call;
    }

    pub fn get_text(&self) -> Option<String> {
        if self.candidates.is_empty() {
            return None;
        }

        let mut response_text = String::new();
        let candidate = &self.candidates[0];
        for part in &candidate.content.parts {
            match &part.text {
                None => continue,
                Some(t) => response_text.push_str(&t),
            }
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

#[derive(Debug, Deserialize, Clone)]
pub struct GeminiFunction {
    pub name: String,
    pub args: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsePart {
    pub text: Option<String>,
    #[serde(rename = "functionCall")]
    pub function_call: Option<GeminiFunction>,
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
