use std::env;

use crate::{
    client::{Message, Model, Role, Settings, Tool},
    gemini::{
        direct_api_client::GeminiApiModel, types::GeminiModel, vertex_client::GeminiVertexModel,
    },
};

#[tokio::test]
async fn test_generate_content_vertex() {
    let model = GeminiVertexModel {
        region: env::var("VERTEX_REGION").unwrap(),
        project_name: env::var("VERTEX_PROJECT").unwrap(),
        client: reqwest::Client::new(),
        model: GeminiModel::Gemini25Flash,
    };

    let response = model
        .new_request()
        .with_message(Message {
            content: "hello how are you?".to_string(),
            role: Some(Role::User),
        })
        .completion()
        .await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_generate_content_direct() {
    let model = GeminiApiModel {
        client: reqwest::Client::new(),
        api_key: env::var("GEMINI_KEY").unwrap(),
        model: GeminiModel::Gemini25Flash,
    };
    let response = model
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message {
            content: "hello, how are you?".to_string(),
            role: Some(Role::User),
        })
        .with_settings(Settings {
            max_tokens: Some(8000),
            timeout: None,
            temperature: None,
            thinking_budget: None,
        })
        .completion()
        .await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_gemini_direct_function_call() {
    let model = GeminiApiModel {
        client: reqwest::Client::new(),
        api_key: env::var("GEMINI_KEY").unwrap(),
        model: GeminiModel::Gemini25Flash,
    };

    let tool = Tool::new(
        "get_weather".to_string(),
        "Get the weather for a city".to_string(),
    )
    .with_parameter(
        "city".to_string(),
        "string".to_string(),
        "the city for which to get the weather".to_string(),
        true,
    );

    let response = model
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message {
            content: "what is the weather like in Paris?".to_string(),
            role: Some(Role::User),
        })
        .with_settings(Settings {
            max_tokens: Some(8000),
            timeout: None,
            temperature: None,
            thinking_budget: None,
        })
        .with_tool(tool)
        .completion()
        .await;
    assert!(response.is_ok());
    assert!(response.unwrap().function.unwrap().name == "get_weather".to_string());
}

#[tokio::test]
async fn test_gemini_vertex_function_call() {
    let model = GeminiVertexModel {
        region: env::var("VERTEX_REGION").unwrap(),
        project_name: env::var("VERTEX_PROJECT").unwrap(),
        client: reqwest::Client::new(),
        model: GeminiModel::Gemini25Flash,
    };

    let tool = Tool::new(
        "get_weather".to_string(),
        "Get the weather for a city".to_string(),
    )
    .with_parameter(
        "city".to_string(),
        "string".to_string(),
        "the city for which to get the weather".to_string(),
        true,
    );

    let response = model
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message {
            content: "what is the weather like in Paris?".to_string(),
            role: Some(Role::User),
        })
        .with_settings(Settings {
            max_tokens: Some(8000),
            timeout: None,
            temperature: None,
            thinking_budget: None,
        })
        .with_tool(tool)
        .completion()
        .await;
    assert!(response.is_ok());
    assert!(response.unwrap().function.unwrap().name == "get_weather".to_string());
}
