use std::collections::HashMap;

use crate::client::{CompletionWrapper, Role};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct ThinkingConfig {
    #[serde(rename = "thinkingBudget")]
    pub thinking_budget: i16,
}

#[derive(Serialize)]
pub struct GenerationConfig {
    #[serde(rename = "maxOutputTokens")]
    pub max_output_tokens: i16,
    pub temperature: i16,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

#[derive(Serialize)]
pub struct Part {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none", rename = "functionCall")]
    pub function_call: Option<FunctionCall>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FunctionCall {
    pub id: String,
    pub name: String,
    pub args: HashMap<String, serde_json::Value>,
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
pub struct Request {
    pub system_instruction: Option<SystemInstructionContent>,
    pub contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    pub generation_config: GenerationConfig,
    // TODO implement safetySettings
    // TODO implement image stuff
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "toolConfig")]
    pub tool_config: Option<ToolConfig>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    pub function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    pub mode: Mode,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Mode {
    /// Default model behavior, model decides to predict either a function call or a natural language response.
    Auto,
    /// Model is constrained to always predicting a function call only.
    Any,
    /// Model will not predict any function call.
    None,
    /// Model decides to predict either a function call or a natural language response, but will validate
    /// function calls with constrained decoding.
    Validated,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: Parameters,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Parameters {
    #[serde(rename = "type")]
    pub param_type: String,
    pub properties: HashMap<String, Property>,
    pub required: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Property {
    #[serde(rename = "type")]
    pub _type: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Items>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Items {
    #[serde(rename = "type")]
    pub item_type: String,
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
    pub content: Option<String>,
    pub prompt_tokens: Option<i32>,
    pub completion_tokens: Option<i32>,
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
