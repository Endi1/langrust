use std::collections::HashMap;

use crate::client::Tool;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Default `max_tokens` used when the caller did not specify one in `Settings`.
/// Anthropic requires this field on every request.
pub const DEFAULT_MAX_TOKENS: i32 = 8192;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaudeModel {
    Sonnet4_5,
    Opus4_6,
    Opus4_7,
}

impl ClaudeModel {
    pub fn to_string(&self) -> String {
        match self {
            ClaudeModel::Sonnet4_5 => "claude-sonnet-4-5".to_string(),
            ClaudeModel::Opus4_6 => "claude-opus-4-6".to_string(),
            ClaudeModel::Opus4_7 => "claude-opus-4-7".to_string(),
        }
    }
}

/// Deterministic synthesis of a `tool_use_id` from a function name.
///
/// NOTE: This is a known limitation. Two calls to the same tool in the same
/// assistant turn would collide. Resolving this properly requires adding an
/// `id` field to the common `FunctionCall` type.
pub fn synth_tool_use_id(name: &str) -> String {
    format!("toolu_{}", name)
}

// ---------------- Request types ----------------

#[derive(Serialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub kind: &'static str, // "enabled"
    pub budget_tokens: i32,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: HashMap<String, Value>,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Serialize)]
pub struct ClaudeMessage {
    pub role: &'static str, // "user" | "assistant"
    pub content: Vec<ContentBlock>,
}

#[derive(Serialize)]
pub struct ClaudeTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

impl ClaudeTool {
    pub fn from_tool(tool: &Tool) -> ClaudeTool {
        // Anthropic accepts standard JSON Schema. If the caller provided
        // `ToolParameters`, serialize them straight through; otherwise emit
        // an empty object schema.
        let input_schema = match &tool.parameters {
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

        ClaudeTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema,
        }
    }
}

#[derive(Serialize)]
pub struct ClaudeRequest {
    pub model: String,
    pub max_tokens: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<ClaudeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ClaudeTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

// ---------------- Response types (non-streaming) ----------------

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        #[allow(dead_code)]
        id: String,
        name: String,
        input: HashMap<String, Value>,
    },
    #[serde(other)]
    Other, // e.g. "thinking" blocks, which we currently ignore
}

#[derive(Debug, Deserialize)]
pub struct ClaudeUsage {
    pub input_tokens: i32,
    pub output_tokens: i32,
}

#[derive(Debug, Deserialize)]
pub struct ClaudeResponse {
    pub content: Vec<ResponseBlock>,
    pub usage: ClaudeUsage,
    #[allow(dead_code)]
    #[serde(default)]
    pub stop_reason: Option<String>,
}

// ---------------- Streaming event types ----------------

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamingEvent {
    MessageStart {
        message: MessageStartMessage,
    },
    ContentBlockStart {
        index: u32,
        content_block: StreamContentBlock,
    },
    ContentBlockDelta {
        index: u32,
        delta: BlockDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        #[allow(dead_code)]
        #[serde(default)]
        delta: Option<serde_json::Value>,
        usage: MessageDeltaUsage,
    },
    MessageStop,
    Ping,
    #[serde(rename = "error")]
    Error {
        error: StreamingError,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub struct MessageStartMessage {
    pub usage: ClaudeUsage,
}

#[derive(Debug, Deserialize)]
pub struct MessageDeltaUsage {
    pub output_tokens: i32,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamContentBlock {
    Text {
        #[allow(dead_code)]
        #[serde(default)]
        text: String,
    },
    ToolUse {
        #[allow(dead_code)]
        id: String,
        name: String,
        // `input` is streamed as `input_json_delta`s; starts empty.
        #[allow(dead_code)]
        #[serde(default)]
        input: serde_json::Value,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub struct StreamingError {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub kind: String,
    pub message: String,
}
