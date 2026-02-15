use std::env;

use futures::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    client::{Message, Model, Settings, StreamEvent, Tool, Usage},
    gemini::{
        direct_api_client::GeminiApiModel,
        types::{GeminiModel, GeminiTool},
        vertex_client::GeminiVertexModel,
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
            StreamEvent::Usage(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            }) => {
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
            StreamEvent::Usage(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            }) => {
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

#[test]
fn test_gemini_tool_converts_nullable_types() {
    #[derive(JsonSchema, Serialize, Deserialize)]
    struct ReadInput {
        filepath: String,
        offset: Option<u32>,
        limit: Option<u32>,
    }

    let tool = Tool::new("read", "Read a file")
        .with_parameter::<ReadInput>()
        .unwrap();
    let gemini_tool = GeminiTool::from_tool(&tool);
    let json = serde_json::to_value(&gemini_tool).unwrap();

    let params = json.get("parameters").unwrap();
    let props = params.get("properties").unwrap();

    // filepath should be a simple STRING
    let filepath = props.get("filepath").unwrap();
    assert_eq!(filepath.get("type").unwrap(), "STRING");
    assert!(filepath.get("nullable").is_none());

    // offset should be INTEGER + nullable
    let offset = props.get("offset").unwrap();
    assert_eq!(offset.get("type").unwrap(), "INTEGER");
    assert_eq!(offset.get("nullable").unwrap(), true);
    // Should NOT have format or minimum
    assert!(offset.get("format").is_none());
    assert!(offset.get("minimum").is_none());

    // limit should be INTEGER + nullable
    let limit = props.get("limit").unwrap();
    assert_eq!(limit.get("type").unwrap(), "INTEGER");
    assert_eq!(limit.get("nullable").unwrap(), true);

    // Top level type should be OBJECT
    assert_eq!(params.get("type").unwrap(), "OBJECT");
}

#[test]
fn test_gemini_tool_converts_nested_objects() {
    #[derive(JsonSchema, Serialize, Deserialize)]
    struct Inner {
        name: String,
        count: Option<i32>,
    }

    #[derive(JsonSchema, Serialize, Deserialize)]
    struct Outer {
        inner: Inner,
        tags: Vec<String>,
    }

    let tool = Tool::new("test", "Test tool")
        .with_parameter::<Outer>()
        .unwrap();
    let gemini_tool = GeminiTool::from_tool(&tool);
    let json = serde_json::to_value(&gemini_tool).unwrap();

    let params = json.get("parameters").unwrap();
    let props = params.get("properties").unwrap();

    // tags should be ARRAY
    let tags = props.get("tags").unwrap();
    assert_eq!(tags.get("type").unwrap(), "ARRAY");
    let items = tags.get("items").unwrap();
    assert_eq!(items.get("type").unwrap(), "STRING");
}
