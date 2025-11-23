use std::{error::Error, fmt};

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

pub trait CompletionWrapper {
    fn completion(&self) -> Option<String>;
    fn prompt_tokens(&self) -> Option<i32>;
    fn completion_tokens(&self) -> Option<i32>;
}

impl fmt::Debug for dyn CompletionWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let completion = self.completion().unwrap_or_else(|| "None".to_string());
        let prompt_tokens = self
            .prompt_tokens()
            .map(|t| t.to_string())
            .unwrap_or_else(|| "None".to_string());
        let completion_tokens = self
            .completion_tokens()
            .map(|t| t.to_string())
            .unwrap_or_else(|| "None".to_string());

        write!(
            f,
            "Completion: {}, Prompt tokens: {}, Completion tokens: {}",
            completion, prompt_tokens, completion_tokens
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "model")]
    Model,
    #[serde(rename = "user")]
    User,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub content: String,
    pub role: Option<Role>,
}

pub struct LLMCallSettings {
    pub model: String,
    pub max_tokens: Option<i16>,
    pub timeout: Option<i16>,
    pub temperature: i16,
    pub thinking_budget: Option<i16>,
}

pub trait StreamWrapper<T>: Stream<Item = Option<String>> + Send + Unpin {
    fn new(
        messages: Vec<ChatMessage>,
        llm_call_settings: LLMCallSettings,
        caller_id: String,
        stream: Box<dyn Stream<Item = Option<T>> + Send + Unpin>,
    ) -> Self
    where
        Self: Sized;

    fn llm_call_settings(&self) -> &LLMCallSettings;
}

#[async_trait]
pub trait Client {
    async fn complete(
        &self,
        system_message: &Option<String>,
        messages: &Vec<ChatMessage>,
        llm_call_settings: &LLMCallSettings,
    ) -> Result<Box<dyn CompletionWrapper>, Box<dyn Error + Send + Sync>> {
        // TODO Add logging/tracing etc

        let response = self
            .chat_completion(system_message, messages, llm_call_settings)
            .await;
        return response;
    }
    /// Abstract method that must be implemented by concrete types
    async fn chat_completion(
        &self,
        system_message: &Option<String>,
        messages: &Vec<ChatMessage>,
        llm_call_settings: &LLMCallSettings,
    ) -> Result<Box<dyn CompletionWrapper>, Box<dyn Error + Send + Sync>>;
}
