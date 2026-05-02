use std::collections::HashMap;

use crate::client::Tool;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAiModel {
    Gpt5_4,
    Gpt5_4Mini,
    Gpt5_4Nano,
    Gpt5_5,
}

impl OpenAiModel {
    pub fn to_string(&self) -> String {
        match self {
            OpenAiModel::Gpt5_4 => "gpt-5.4".to_string(),
            OpenAiModel::Gpt5_4Mini => "gpt-5.4-mini".to_string(),
            OpenAiModel::Gpt5_4Nano => "gpt-5.4-nano".to_string(),
            OpenAiModel::Gpt5_5 => "gpt-5.5".to_string(),
        }
    }
}

/// Deterministic synthesis of a `call_id` from a function name.
///
/// NOTE: Same known limitation as in the Claude client — two calls to the
/// same tool in the same assistant turn would collide. Resolving this
/// properly requires an `id` field on the common `FunctionCall` type.
pub fn synth_call_id(name: &str) -> String {
    format!("call_{}", name)
}

// ---------------- Request types (Responses API) ----------------

/// Input items for the Responses API.
/// Uses internal tagging via `type` field.
#[derive(Serialize)]
#[serde(tag = "type")]
pub enum OpenAiInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: String,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

/// Tool definition for the Responses API (internally tagged, flat structure).
#[derive(Serialize)]
pub struct OpenAiTool {
    #[serde(rename = "type")]
    pub kind: &'static str, // "function"
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub strict: bool,
}

impl OpenAiTool {
    pub fn from_tool(tool: &Tool) -> OpenAiTool {
        let parameters = match &tool.parameters {
            Some(p) => {
                let mut map = serde_json::Map::new();
                map.insert("type".to_string(), Value::String(p._type.clone()));
                map.insert(
                    "properties".to_string(),
                    Value::Object(p.properties.clone().into_iter().collect()),
                );
                map.insert(
                    "required".to_string(),
                    Value::Array(
                        p.required
                            .iter()
                            .map(|s| Value::String(s.clone()))
                            .collect(),
                    ),
                );
                Value::Object(map)
            }
            None => serde_json::json!({ "type": "object", "properties": {} }),
        };

        OpenAiTool {
            kind: "function",
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters,
            strict: false,
        }
    }
}

/// Request body for the Responses API (`POST /v1/responses`).
#[derive(Serialize)]
pub struct OpenAiRequest {
    pub model: String,
    pub input: Vec<OpenAiInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    pub store: bool,
}

// ---------------- Response types (Responses API, non-streaming) ----------------

#[derive(Debug, Deserialize)]
pub struct OpenAiUsage {
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum OpenAiContentPart {
    #[serde(rename = "output_text")]
    OutputText {
        #[allow(dead_code)]
        text: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum OpenAiOutputItem {
    #[serde(rename = "message")]
    Message {
        #[serde(default)]
        #[allow(dead_code)]
        content: Vec<OpenAiContentPart>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(default)]
        #[allow(dead_code)]
        call_id: Option<String>,
        name: String,
        arguments: String,
    },
}

#[derive(Debug, Deserialize)]
pub struct OpenAiResponse {
    pub output: Vec<OpenAiOutputItem>,
    #[serde(default)]
    pub output_text: Option<String>,
    pub usage: Option<OpenAiUsage>,
}

impl OpenAiResponse {
    pub fn get_text(&self) -> String {
        self.output_text.clone().unwrap_or_default()
    }

    pub fn get_function(&self) -> Option<(String, HashMap<String, Value>)> {
        for item in &self.output {
            if let OpenAiOutputItem::FunctionCall { name, arguments, .. } = item {
                let args: HashMap<String, Value> = if arguments.is_empty() {
                    HashMap::new()
                } else {
                    serde_json::from_str(arguments).unwrap_or_default()
                };
                return Some((name.clone(), args));
            }
        }
        None
    }
}

// ---------------- Streaming event types (Responses API) ----------------

/// A single streamed item from `response.output_item.done`.
#[derive(Debug, Deserialize)]
pub struct StreamOutputItem {
    #[serde(rename = "type")]
    pub item_type: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Generic shape for all SSE data payloads from the Responses API.
/// Different event types populate different fields.
#[derive(Debug, Deserialize)]
pub struct ResponsesStreamEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    /// Text delta (for `response.output_text.delta`)
    #[serde(default)]
    pub delta: Option<String>,
    /// Output item (for `response.output_item.done`)
    #[serde(default)]
    pub item: Option<StreamOutputItem>,
    /// Full response (for `response.completed`)
    #[serde(default)]
    pub response: Option<OpenAiResponse>,
}
