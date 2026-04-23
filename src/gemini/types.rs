use std::collections::HashMap;

use crate::client::{Role, Tool};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeminiModel {
    Gemini25Flash,
    Gemini31Pro,
    Gemini3Flash,
    Gemini31FlashLite,
}

impl GeminiModel {
    pub fn to_string(&self) -> String {
        match self {
            GeminiModel::Gemini25Flash => "gemini-2.5-flash".to_string(),
            GeminiModel::Gemini31Pro => "gemini-3.1-pro-preview".to_string(),
            GeminiModel::Gemini3Flash => "gemini-3-flash-preview".to_string(),
            GeminiModel::Gemini31FlashLite => "gemini-3.1-flash-lite-preview".to_string(),
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
#[serde(untagged)]
pub enum Part {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCallPart,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponsePart,
    },
}

#[derive(Serialize)]
pub struct FunctionCallPart {
    pub name: String,
    pub args: HashMap<String, Value>,
}

#[derive(Serialize)]
pub struct FunctionResponsePart {
    pub name: String,
    pub response: Value,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeminiToolParameters {
    #[serde(rename = "type")]
    pub _type: String,
    pub properties: HashMap<String, Value>, // TODO Eventually improve the typing here
    pub required: Vec<String>,
}

#[derive(Serialize)]
pub struct GeminiTool {
    name: String,
    description: String,
    parameters: Option<GeminiToolParameters>,
}

/// Convert a JSON Schema type string to Gemini API type string.
fn to_gemini_type(json_schema_type: &str) -> String {
    match json_schema_type {
        "string" => "STRING".to_string(),
        "integer" => "INTEGER".to_string(),
        "number" => "NUMBER".to_string(),
        "boolean" => "BOOLEAN".to_string(),
        "array" => "ARRAY".to_string(),
        "object" => "OBJECT".to_string(),
        other => other.to_uppercase(),
    }
}

/// Convert a JSON Schema property value to a Gemini-compatible schema value.
/// Handles: type arrays like ["integer", "null"] → { type: "INTEGER", nullable: true },
/// removes unsupported fields (format, minimum, maximum, $schema, title, etc.),
/// and recursively converts nested objects/arrays.
fn convert_property_to_gemini(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut result = serde_json::Map::new();
            let mut nullable = false;
            let mut resolved_type: Option<String> = None;

            // Handle "type" field — could be a string or an array like ["integer", "null"]
            if let Some(type_val) = map.get("type") {
                match type_val {
                    Value::Array(types) => {
                        // e.g. ["integer", "null"]
                        let non_null: Vec<&str> = types
                            .iter()
                            .filter_map(|t| t.as_str())
                            .filter(|t| *t != "null")
                            .collect();
                        if types.iter().any(|t| t.as_str() == Some("null")) {
                            nullable = true;
                        }
                        if let Some(first) = non_null.first() {
                            resolved_type = Some(to_gemini_type(first));
                        }
                    }
                    Value::String(s) => {
                        resolved_type = Some(to_gemini_type(s));
                    }
                    _ => {}
                }
            }

            if let Some(t) = resolved_type {
                result.insert("type".to_string(), Value::String(t));
            }

            if nullable {
                result.insert("nullable".to_string(), Value::Bool(true));
            }

            // Copy over description if present
            if let Some(desc) = map.get("description") {
                result.insert("description".to_string(), desc.clone());
            }

            // Copy over enum if present
            if let Some(enum_val) = map.get("enum") {
                result.insert("enum".to_string(), enum_val.clone());
            }

            // Recursively convert properties for nested objects
            if let Some(Value::Object(props)) = map.get("properties") {
                let converted: serde_json::Map<String, Value> = props
                    .iter()
                    .map(|(k, v)| (k.clone(), convert_property_to_gemini(v)))
                    .collect();
                result.insert("properties".to_string(), Value::Object(converted));
            }

            // Copy over required for nested objects
            if let Some(req) = map.get("required") {
                result.insert("required".to_string(), req.clone());
            }

            // Handle items for arrays
            if let Some(items) = map.get("items") {
                result.insert("items".to_string(), convert_property_to_gemini(items));
            }

            Value::Object(result)
        }
        other => other.clone(),
    }
}

impl GeminiTool {
    pub fn from_tool(tool: &Tool) -> GeminiTool {
        GeminiTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone().map(|p| {
                let converted_properties: HashMap<String, Value> = p
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), convert_property_to_gemini(v)))
                    .collect();
                GeminiToolParameters {
                    _type: to_gemini_type(&p._type),
                    properties: converted_properties,
                    required: p.required,
                }
            }),
        }
    }
}

#[derive(Serialize)]
pub struct GeminiTools {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<GeminiTool>,
}

#[derive(Serialize)]
pub struct GeminiRequest {
    pub system_instruction: Option<SystemInstructionContent>,
    pub contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    pub generation_config: GenerationConfig, // TODO implement safetySettings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTools>>,
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

    pub fn get_total_tokens(&self) -> Option<i32> {
        self.usage_metadata
            .as_ref()
            .and_then(|m| m.total_token_count)
    }
}

#[derive(Debug, Deserialize)]
pub struct Candidate {
    #[serde(default)]
    pub content: ResponseContent,
    #[serde(rename = "finishReason")]
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
    #[allow(dead_code)]
    pub index: Option<i32>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ResponseContent {
    #[serde(default)]
    pub parts: Vec<ResponsePart>,
    #[allow(dead_code)]
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
