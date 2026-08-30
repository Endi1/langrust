//! The [`Model`] trait and the fluent [`ModelRequestBuilder`].

use async_trait::async_trait;

use crate::error::LlmError;
use crate::types::{Completion, Message, Settings, StreamResult, Tool};

#[async_trait]
pub trait Model: Send + Sync {
    async fn completion(&self, request: ModelRequest) -> Result<Completion, LlmError>;

    async fn stream_completion(&self, request: ModelRequest) -> Result<StreamResult, LlmError>;

    fn new_request(&self) -> ModelRequestBuilder<'_>
    where
        Self: Sized,
    {
        ModelRequestBuilder::new(self as &dyn Model)
    }

    fn model_name(&self) -> String;
}

pub struct ModelRequest {
    pub system: Option<String>,
    pub messages: Option<Vec<Message>>,
    pub settings: Option<Settings>,
    pub tools: Option<Vec<Tool>>,
}

#[derive(Clone)]
pub struct ModelRequestBuilder<'a> {
    pub model: &'a dyn Model,
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
        self
    }

    pub fn with_message(&mut self, message: Message) -> &mut Self {
        self.messages.get_or_insert_with(Vec::new).push(message);
        self
    }

    pub fn with_messages(&mut self, messages: Vec<Message>) -> &mut Self {
        self.messages.get_or_insert_with(Vec::new).extend(messages);
        self
    }

    pub fn with_settings(&mut self, settings: Settings) -> &mut Self {
        self.settings = Some(settings);
        self
    }

    pub fn with_tool(&mut self, tool: Tool) -> &mut Self {
        self.tools.get_or_insert_with(Vec::new).push(tool);
        self
    }

    pub fn with_tools(&mut self, tools: Vec<Tool>) -> &mut Self {
        self.tools.get_or_insert_with(Vec::new).extend(tools);
        self
    }

    pub async fn completion(&self) -> Result<Completion, LlmError> {
        self.model.completion(self.to_model_request()).await
    }

    pub async fn stream(&self) -> Result<StreamResult, LlmError> {
        self.model.stream_completion(self.to_model_request()).await
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
