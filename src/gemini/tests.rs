use std::env;

use crate::{
    client::{ChatMessage, Client, LLMCallSettings, Role},
    gemini::{direct_api_client::GeminiApiClient, vertex_client::GeminiVertexClient},
};

#[tokio::test]
async fn test_generate_content_vertex() {
    let gemini_client = GeminiVertexClient {
        region: env::var("VERTEX_REGION").unwrap(),
        project_name: env::var("VERTEX_PROJECT").unwrap(),
        client: reqwest::Client::new(),
    };
    let messages = [ChatMessage {
        content: "hello how are you?".to_string(),
        role: Some(Role::User),
    }]
    .to_vec();
    let call_settings = LLMCallSettings {
        model: "gemini-2.5-flash".to_string(),
        max_tokens: Some(8000),
        timeout: None,
        temperature: 0,
        thinking_budget: Some(0),
    };
    let response = gemini_client
        .complete(&None, &messages, &call_settings)
        .await;
    assert!(response.is_ok());

    let completion = response.expect("No completion found").completion();
    assert!(completion.is_some());
}

#[tokio::test]
async fn test_generate_content_direct() {
    let gemini_client = GeminiApiClient {
        client: reqwest::Client::new(),
        api_key: env::var("GEMINI_KEY").unwrap(),
    };
    let messages = [ChatMessage {
        content: "hello how are you?".to_string(),
        role: Some(Role::User),
    }]
    .to_vec();
    let call_settings = LLMCallSettings {
        model: "gemini-2.5-flash".to_string(),
        max_tokens: Some(8000),
        timeout: None,
        temperature: 0,
        thinking_budget: Some(0),
    };
    let system_message = Some("you are a helpful assistant".to_string());
    let response = gemini_client
        .complete(&system_message, &messages, &call_settings)
        .await;
    println!("{:?}", response);
    assert!(response.is_ok());

    let completion = response.expect("No completion found").completion();
    assert!(completion.is_some());
}
