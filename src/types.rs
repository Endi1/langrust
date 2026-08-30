//! Provider-agnostic data types: messages, tools, settings, usage and
//! completion/stream results.

use std::{collections::HashMap, pin::Pin};

use futures::Stream;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub args: HashMap<String, Value>,
}

#[derive(Debug)]
pub struct Completion {
    pub completion: String,
    pub usage: Usage,
    pub function: Option<FunctionCall>,
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Delta(String),
    Usage(Usage),
    FunctionCall(FunctionCall),
    Error(String),
}

pub type StreamResult = Pin<Box<dyn Stream<Item = StreamEvent> + Send>>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "model")]
    Model,
    #[serde(rename = "user")]
    User,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    #[default]
    Text,
    FunctionCall(FunctionCall),
    FunctionResponse {
        name: String,
        response: Option<serde_json::Value>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub content: String,
    pub role: Option<Role>,
    #[serde(default)]
    pub message_type: MessageType,
}

impl Message {
    pub fn user(content: String) -> Message {
        Message {
            content,
            role: Some(Role::User),
            message_type: MessageType::Text,
        }
    }

    pub fn model(content: String) -> Message {
        Message {
            content,
            role: Some(Role::Model),
            message_type: MessageType::Text,
        }
    }

    pub fn function_call(function_call: FunctionCall) -> Message {
        let content = serde_json::to_string(&function_call).unwrap();
        Message {
            content,
            role: Some(Role::Model),
            message_type: MessageType::FunctionCall(function_call),
        }
    }

    pub fn function_result<T: Serialize>(name: String, value: T) -> Message {
        let response = serde_json::to_value(&value).ok();
        let content = serde_json::json!({
            "name": name,
            "response": response,
        })
        .to_string();
        Message {
            content,
            role: Some(Role::User),
            message_type: MessageType::FunctionResponse { name, response },
        }
    }
}

#[derive(Clone)]
pub struct Settings {
    pub max_tokens: Option<f32>,
    pub timeout: Option<i16>,
    pub temperature: Option<f32>,
    pub thinking_budget: Option<i16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolParameters {
    #[serde(rename = "type")]
    pub _type: String,
    #[serde(default)]
    pub properties: HashMap<String, Value>, // TODO Eventually improve the typing here
    #[serde(default)]
    pub required: Vec<String>,
}

impl ToolParameters {
    /// Standard JSON Schema object form:
    /// `{ "type": ..., "properties": ..., "required": ... }`.
    ///
    /// Used verbatim by providers that accept plain JSON Schema (Anthropic,
    /// OpenAI). Gemini applies its own conversion on top (uppercase type
    /// names, `nullable`).
    pub fn to_json_schema(&self) -> Value {
        serde_json::json!({
            "type": self._type,
            "properties": self.properties,
            "required": self.required,
        })
    }

    /// Schema used when a tool declares no parameters: an empty object.
    pub fn empty_json_schema() -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Option<ToolParameters>,
}

impl Tool {
    pub fn new(name: &'static str, description: &'static str) -> Tool {
        Tool {
            name: name.to_string(),
            description: description.to_string(),
            parameters: None,
        }
    }

    pub fn with_parameter<T: JsonSchema>(self) -> Result<Tool, serde_json::Error> {
        let arg_schema = schema_for!(T);
        let json_value = serde_json::to_value(&arg_schema)?;
        let parameters: ToolParameters = serde_json::from_value(json_value)?;
        Ok(Tool {
            name: self.name,
            description: self.description,
            parameters: Some(parameters),
        })
    }

    /// The tool's parameters as standard JSON Schema, or an empty object
    /// schema when the tool takes no parameters.
    pub fn parameters_json_schema(&self) -> Value {
        self.parameters
            .as_ref()
            .map(ToolParameters::to_json_schema)
            .unwrap_or_else(ToolParameters::empty_json_schema)
    }
}
