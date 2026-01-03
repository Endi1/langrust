use std::{collections::HashMap, env};

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;

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
        .with_message(Message::user("hello, how are you?".to_string()))
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
        .with_message(Message::user("hello, how are you?".to_string()))
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

    let tool = Tool::new("get_weather", "Get the weather for a city")
        .with_parameter::<WeatherRequest>()
        .unwrap();

    let response = model
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message::user(
            "What is the weather like in Paris?".to_string(),
        ))
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

    let tool = Tool::new("get_weather", "Get the weather for a city")
        .with_parameter::<WeatherRequest>()
        .unwrap();

    let response = model
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message::user(
            "What is the weather like in Paris?".to_string(),
        ))
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

struct ExecutableTool<A, R> {
    tool: Tool,
    executable: fn(A) -> R,
}

impl<A, R> ExecutableTool<A, R> {
    pub fn new(tool: &Tool, executable: fn(A) -> R) -> ExecutableTool<A, R> {
        ExecutableTool {
            tool: tool.clone(),
            executable: executable,
        }
    }
    pub fn run(self, arg: A) -> R {
        (self.executable)(arg)
    }
}

#[derive(Deserialize, JsonSchema)]
struct WeatherRequest {
    city: String,
}

#[tokio::test]
async fn test_function_execution() {
    let model = GeminiVertexModel {
        region: env::var("VERTEX_REGION").unwrap(),
        project_name: env::var("VERTEX_PROJECT").unwrap(),
        client: reqwest::Client::new(),
        model: GeminiModel::Gemini25Flash,
    };

    let tool = Tool::new("get_weather", "Get the weather for a city")
        .with_parameter::<WeatherRequest>()
        .unwrap();

    let weather_tool: ExecutableTool<WeatherRequest, String> =
        ExecutableTool::new(&tool, |arg: WeatherRequest| {
            format!("The weather in {} is great!", arg.city)
        });

    let response = model
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message::user(
            "What is the weather like in Paris?".to_string(),
        ))
        .with_settings(Settings {
            max_tokens: Some(8000),
            timeout: None,
            temperature: None,
            thinking_budget: None,
        })
        .with_tool(weather_tool.tool.clone())
        .completion()
        .await;
    assert!(response.is_ok());
    let function = response.unwrap().function.unwrap();
    let args = function.args;

    let parsed_args: WeatherRequest =
        serde_json::from_value(serde_json::to_value(args).unwrap()).unwrap();

    let function_response = weather_tool.run(parsed_args);
    assert!(function_response == "The weather in Paris is great!".to_string());
}
