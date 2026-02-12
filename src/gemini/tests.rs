use std::env;

use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::{
    client::{Message, Model, Settings, StreamEvent, Tool},
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
async fn test_with_messages_gemini_direct() {
    let model = GeminiApiModel {
        client: reqwest::Client::new(),
        api_key: env::var("GEMINI_KEY").unwrap(),
        model: GeminiModel::Gemini25Flash,
    };
    let messages = vec![
        Message::user("hello, how are you?".to_string()),
        Message::model("I am fine, and you?".to_string()),
    ];
    let mut binding = model.new_request();
    let request_builder = binding
        .with_system("you are a helpful assistant".to_string())
        .with_messages(messages)
        .with_message(Message::user("I am fine, thanks for asking".to_string()))
        .with_settings(Settings {
            max_tokens: Some(8000),
            timeout: None,
            temperature: None,
            thinking_budget: None,
        });

    let response = request_builder.completion().await;

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

#[tokio::test]
async fn test_stream_generate_content_direct() {
    let model = GeminiApiModel {
        client: reqwest::Client::new(),
        api_key: env::var("GEMINI_KEY").unwrap(),
        model: GeminiModel::Gemini25Flash,
    };

    let mut stream = model
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message::user("hello, how are you?".to_string()))
        .with_settings(Settings {
            max_tokens: Some(8000),
            timeout: None,
            temperature: None,
            thinking_budget: None,
        })
        .stream()
        .await
        .expect("stream request should succeed");

    let mut got_delta = false;
    let mut got_usage = false;
    let mut full_text = String::new();

    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::Delta(text) => {
                got_delta = true;
                full_text.push_str(&text);
            }
            StreamEvent::Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            } => {
                got_usage = true;
                assert!(prompt_tokens > 0);
                assert!(completion_tokens > 0);
                assert!(total_tokens > 0);
            }
            StreamEvent::FunctionCall(_) => {}
            StreamEvent::Error(e) => panic!("stream event should not be an error: {}", e),
        }
    }

    assert!(got_delta, "should have received at least one Delta event");
    assert!(got_usage, "should have received a Usage event");
    assert!(!full_text.is_empty(), "streamed text should not be empty");
}

#[tokio::test]
async fn test_stream_generate_content_vertex() {
    let model = GeminiVertexModel {
        region: env::var("VERTEX_REGION").unwrap(),
        project_name: env::var("VERTEX_PROJECT").unwrap(),
        client: reqwest::Client::new(),
        model: GeminiModel::Gemini25Flash,
    };

    let mut stream = model
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message::user("hello, how are you?".to_string()))
        .with_settings(Settings {
            max_tokens: Some(8000),
            timeout: None,
            temperature: None,
            thinking_budget: None,
        })
        .stream()
        .await
        .expect("stream request should succeed");

    let mut got_delta = false;
    let mut got_usage = false;
    let mut full_text = String::new();

    while let Some(event) = stream.next().await {

        match event {
            StreamEvent::Delta(text) => {
                got_delta = true;
                full_text.push_str(&text);
            }
            StreamEvent::Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            } => {
                got_usage = true;
                assert!(prompt_tokens > 0);
                assert!(completion_tokens > 0);
                assert!(total_tokens > 0);
            }
            StreamEvent::FunctionCall(_) => {}
            StreamEvent::Error(e) => panic!("stream event should not be an error: {}", e),
        }
    }

    assert!(got_delta, "should have received at least one Delta event");
    assert!(got_usage, "should have received a Usage event");
    assert!(!full_text.is_empty(), "streamed text should not be empty");
}

#[tokio::test]
async fn test_stream_function_call_direct() {
    let model = GeminiApiModel {
        client: reqwest::Client::new(),
        api_key: env::var("GEMINI_KEY").unwrap(),
        model: GeminiModel::Gemini25Flash,
    };

    let tool = Tool::new("get_weather", "Get the weather for a city")
        .with_parameter::<WeatherRequest>()
        .unwrap();

    let mut stream = model
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
        .stream()
        .await
        .expect("stream request should succeed");

    let mut got_function_call = false;

    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::FunctionCall(fc) => {
                got_function_call = true;
                assert_eq!(fc.name, "get_weather");
            }
            StreamEvent::Error(e) => panic!("stream event should not be an error: {}", e),
            _ => {}
        }
    }

    assert!(
        got_function_call,
        "should have received a FunctionCall event"
    );
}
