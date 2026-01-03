use schemars::{JsonSchema, Schema, schema_for, schema_for_value};
use serde_json::{self, Value};
use std::{collections::HashMap, error::Error};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub args: HashMap<String, Value>,
}

#[derive(Debug)]
pub struct Completion {
    pub completion: String,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub function: Option<FunctionCall>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "model")]
    Model,
    #[serde(rename = "user")]
    User,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub content: String,
    pub role: Option<Role>,
}

#[derive(Serialize)]
pub struct FunctionResponse {
    name: String,
    response: Option<serde_json::Value>,
}

impl FunctionResponse {
    fn new(name: String, response: Option<serde_json::Value>) -> FunctionResponse {
        Self {
            name: name,
            response: response,
        }
    }
}

impl Message {
    pub fn user(content: String) -> Message {
        Message {
            content: content,
            role: Some(Role::User),
        }
    }

    pub fn model(content: String) -> Message {
        Message {
            content: content,
            role: Some(Role::Model),
        }
    }

    pub fn function_call(function_call: FunctionCall) -> Message {
        Message {
            content: serde_json::to_string(&function_call).unwrap(),
            role: Some(Role::Model),
        }
    }

    pub fn function_result<T: Serialize>(name: String, value: T) -> Message {
        let function_response = serde_json::to_value(&value).ok();
        Message {
            content: serde_json::to_string(&FunctionResponse::new(name, function_response))
                .unwrap(),
            role: Some(Role::Model),
        }
    }
}

#[async_trait]
pub trait Model {
    async fn completion(
        &self,
        request: ModelRequest,
    ) -> Result<Completion, Box<dyn Error + Send + Sync>>;

    fn new_request(&self) -> ModelRequestBuilder<'_>
    where
        Self: Sized,
    {
        ModelRequestBuilder::new(self as &dyn Model)
    }
}

#[derive(Clone)]
pub struct Settings {
    pub max_tokens: Option<i16>,
    pub timeout: Option<i16>,
    pub temperature: Option<i16>,
    pub thinking_budget: Option<i16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolParameters {
    #[serde(rename = "type")]
    pub _type: String,
    pub properties: HashMap<String, Value>, // TODO Eventually improve the typing here
    pub required: Vec<String>,
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
        match self.parameters {
            None => Ok(Tool {
                name: self.name,
                description: self.description,
                parameters: Some(parameters),
            }),
            Some(_) => Ok(self),
        }
    }
}

#[derive(Clone)]
pub struct ModelRequestBuilder<'a> {
    pub model: &'a dyn Model,
    pub system: Option<String>,
    pub messages: Option<Vec<Message>>,
    pub settings: Option<Settings>,
    pub tools: Option<Vec<Tool>>,
}

unsafe impl<'a> Sync for ModelRequestBuilder<'a> {}
unsafe impl<'a> Send for ModelRequestBuilder<'a> {}

pub struct ModelRequest {
    pub system: Option<String>,
    pub messages: Option<Vec<Message>>,
    pub settings: Option<Settings>,
    pub tools: Option<Vec<Tool>>,
}

impl<'a> ModelRequestBuilder<'a> {
    pub fn new(model: &'a dyn Model) -> Self {
        ModelRequestBuilder {
            model,
            system: None,
            messages: None,
            settings: None,
            tools: None,
        }
    }

    pub fn with_system(&mut self, system: String) -> &mut Self {
        self.system = Some(system);
        return self;
    }

    pub fn with_message(&mut self, message: Message) -> &mut Self {
        match self.messages {
            None => self.messages = Some(vec![message]),
            Some(_) => {
                self.messages.clone().map(|mut ms| ms.push(message));
            }
        }
        return self;
    }

    pub fn with_messages(&mut self, messages: Vec<Message>) -> &mut Self {
        match self.messages {
            None => self.messages = Some(messages),
            Some(_) => {
                self.messages.clone().map(|mut ms| ms.extend(messages));
            }
        }
        return self;
    }

    pub fn with_settings(&mut self, settings: Settings) -> &mut Self {
        self.settings = Some(settings);
        return self;
    }

    pub fn with_tool(&mut self, tool: Tool) -> &mut Self {
        match self.tools {
            None => self.tools = Some(vec![tool]),
            Some(_) => {
                self.tools.clone().map(|mut ts| ts.push(tool));
            }
        }
        return self;
    }

    pub fn with_tools(&mut self, tools: Vec<Tool>) -> &mut Self {
        match self.tools {
            None => self.tools = Some(tools),
            Some(_) => {
                self.tools.clone().map(|mut ts| ts.extend(tools));
            }
        }
        return self;
    }

    pub async fn completion(&self) -> Result<Completion, Box<dyn Error + Send + Sync>> {
        self.model.completion(self.to_model_request()).await
    }

    pub fn to_model_request(&self) -> ModelRequest {
        ModelRequest {
            system: self.system.clone(),
            messages: self.messages.clone(),
            settings: self.settings.clone(),
            tools: self.tools.clone(),
        }
    }
}
