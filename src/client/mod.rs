use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

pub trait CompletionWrapper {
    fn completion(&self) -> Option<String>;
    fn prompt_tokens(&self) -> Option<i32>;
    fn completion_tokens(&self) -> Option<i32>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "system")]
    System,
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
        messages: &Vec<ChatMessage>,
        llm_call_settings: &LLMCallSettings,
    ) -> Option<Box<dyn CompletionWrapper>> {
        // TODO Add logging/tracing etc

        let response = self.chat_completion(messages, llm_call_settings).await;

        let Ok(response) = response else {
            return None;
        };

        Some(response)
    }
    /// Abstract method that must be implemented by concrete types
    async fn chat_completion(
        &self,
        messages: &Vec<ChatMessage>,
        llm_call_settings: &LLMCallSettings,
    ) -> Result<Box<dyn CompletionWrapper>, String>;
}
