use std::error::Error;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct Completion {
    pub completion: String,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
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

#[derive(Clone)]
pub struct ModelRequestBuilder<'a> {
    pub model: &'a dyn Model,
    pub system: Option<String>,
    pub messages: Option<Vec<Message>>,
    pub settings: Option<Settings>,
}

unsafe impl<'a> Sync for ModelRequestBuilder<'a> {}
unsafe impl<'a> Send for ModelRequestBuilder<'a> {}

pub struct ModelRequest {
    pub system: Option<String>,
    pub messages: Option<Vec<Message>>,
    pub settings: Option<Settings>,
}

impl<'a> ModelRequestBuilder<'a> {
    pub fn new(model: &'a dyn Model) -> Self {
        ModelRequestBuilder {
            model,
            system: None,
            messages: None,
            settings: None,
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

    pub async fn completion(&self) -> Result<Completion, Box<dyn Error + Send + Sync>> {
        self.model.completion(self.to_model_request()).await
    }

    pub fn to_model_request(&self) -> ModelRequest {
        ModelRequest {
            system: self.system.clone(),
            messages: self.messages.clone(),
            settings: self.settings.clone(),
        }
    }
}
