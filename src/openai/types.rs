use std::collections::HashMap;

use crate::client::Tool;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAiModel {
    Gpt5_4,
    Gpt5_4Mini,
    Gpt5_4Nano
}

impl OpenAiModel {
    pub fn to_string(&self) -> String {
        match self {
            OpenAiModel::Gpt5_4 => "gpt-5.4".to_string(),
            OpenAiModel::Gpt5_4Mini => "gpt-5.4-mini".to_string(),
            OpenAiModel::Gpt5_4Nano => "gpt-5.4-nano".to_string()
        }
    }
}

/// Deterministic synthesis of a `tool_call_id` from a function name.
///
/// NOTE: Same known limitation as in the Claude client — two calls to the
/// same tool in the same assistant turn would collide. Resolving this
/// properly requires an `id` field on the common `FunctionCall` type.
pub fn synth_tool_call_id(name: &str) -> String {
    format!("call_{}", name)
}

// ---------------- Request types ----------------

#[derive(Serialize)]
pub struct OpenAiFunctionCall {
    pub name: String,
    /// JSON-encoded arguments string (as required by OpenAI).
    pub arguments: String,
}

#[derive(Serialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str, // "function"
    pub function: OpenAiFunctionCall,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum OpenAiMessage {
    System {
        role: &'static str, // "system"
        content: String,
    },
    User {
        role: &'static str, // "user"
        content: String,
    },
    Assistant {
        role: &'static str, // "assistant"
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(rename = "tool_calls", skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<OpenAiToolCall>>,
    },
    Tool {
        role: &'static str, // "tool"
        #[serde(rename = "tool_call_id")]
        tool_call_id: String,
        content: String,
    },
}

#[derive(Serialize)]
pub struct OpenAiFunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Serialize)]
pub struct OpenAiTool {
    #[serde(rename = "type")]
    pub kind: &'static str, // "function"
    pub function: OpenAiFunctionDef,
}

impl OpenAiTool {
    pub fn from_tool(tool: &Tool) -> OpenAiTool {
        // OpenAI accepts standard JSON Schema directly, like Anthropic.
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
            function: OpenAiFunctionDef {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters,
            },
        }
    }
}

#[derive(Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Serialize)]
pub struct OpenAiRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "stream_options")]
    pub stream_options: Option<StreamOptions>,
}

// ---------------- Response types (non-streaming) ----------------

#[derive(Debug, Deserialize)]
pub struct OpenAiUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiResponseFunctionCall {
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiResponseToolCall {
    #[allow(dead_code)]
    pub id: Option<String>,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub kind: Option<String>,
    pub function: OpenAiResponseFunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiResponseMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, rename = "tool_calls")]
    pub tool_calls: Option<Vec<OpenAiResponseToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiChoice {
    pub message: OpenAiResponseMessage,
    #[allow(dead_code)]
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub index: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiResponse {
    pub choices: Vec<OpenAiChoice>,
    pub usage: Option<OpenAiUsage>,
}

impl OpenAiResponse {
    pub fn get_text(&self) -> String {
        self.choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default()
    }

    pub fn get_function(&self) -> Option<(String, HashMap<String, Value>)> {
        let choice = self.choices.first()?;
        let tool_calls = choice.message.tool_calls.as_ref()?;
        let first = tool_calls.first()?;
        let name = first.function.name.clone()?;
        let args_str = first.function.arguments.clone().unwrap_or_default();
        let args: HashMap<String, Value> = if args_str.is_empty() {
            HashMap::new()
        } else {
            serde_json::from_str(&args_str).unwrap_or_default()
        };
        Some((name, args))
    }
}

// ---------------- Streaming event types ----------------

#[derive(Debug, Deserialize)]
pub struct StreamToolCallFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct StreamToolCallDelta {
    pub index: u32,
    #[serde(default)]
    #[allow(dead_code)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<StreamToolCallFunctionDelta>,
}

#[derive(Debug, Deserialize)]
pub struct StreamChoiceDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, rename = "tool_calls")]
    pub tool_calls: Option<Vec<StreamToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
pub struct StreamChoice {
    pub delta: StreamChoiceDelta,
    #[serde(default)]
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub index: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct StreamChunk {
    #[serde(default)]
    pub choices: Vec<StreamChoice>,
    #[serde(default)]
    pub usage: Option<OpenAiUsage>,
}
