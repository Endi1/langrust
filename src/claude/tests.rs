use std::env;

use futures::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    claude::{
        direct_api_client::ClaudeApiModel,
        types::{ClaudeModel, ClaudeTool},
    },
    client::{Message, Model, Settings, StreamEvent, Tool, Usage},
};

#[tokio::test]
async fn test_generate_content_direct() {
    let model = ClaudeApiModel {
        client: reqwest::Client::new(),
        api_key: env::var("CLAUDE_KEY").unwrap(),
        model: ClaudeModel::Sonnet4_5,
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
async fn test_with_messages_claude_direct() {
    let model = ClaudeApiModel {
        client: reqwest::Client::new(),
        api_key: env::var("CLAUDE_KEY").unwrap(),
        model: ClaudeModel::Sonnet4_5,
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
async fn test_claude_direct_function_call() {
    let model = ClaudeApiModel {
        client: reqwest::Client::new(),
        api_key: env::var("CLAUDE_KEY").unwrap(),
        model: ClaudeModel::Sonnet4_5,
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
    let model = ClaudeApiModel {
        client: reqwest::Client::new(),
        api_key: env::var("CLAUDE_KEY").unwrap(),
        model: ClaudeModel::Sonnet4_5,
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
    let model = ClaudeApiModel {
        client: reqwest::Client::new(),
        api_key: env::var("CLAUDE_KEY").unwrap(),
        model: ClaudeModel::Sonnet4_5,
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
    let model = ClaudeApiModel {
        client: reqwest::Client::new(),
        api_key: env::var("CLAUDE_KEY").unwrap(),
        model: ClaudeModel::Sonnet4_5,
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

// ----------------------------------------------------------------------------
// Pure unit tests for tool-schema conversion. Unlike Gemini, Anthropic accepts
// plain JSON Schema, so these verify the schema is passed through unchanged.
// ----------------------------------------------------------------------------

#[test]
fn test_claude_tool_passes_through_nullable_types() {
    #[derive(JsonSchema, Serialize, Deserialize)]
    struct ReadInput {
        filepath: String,
        offset: Option<u32>,
        limit: Option<u32>,
    }

    let tool = Tool::new("read", "Read a file")
        .with_parameter::<ReadInput>()
        .unwrap();
    let claude_tool = ClaudeTool::from_tool(&tool);
    let json = serde_json::to_value(&claude_tool).unwrap();

    let schema = json.get("input_schema").unwrap();
    // Top-level type is standard JSON Schema "object" (lowercase, not "OBJECT")
    assert_eq!(schema.get("type").unwrap(), "object");

    let props = schema.get("properties").unwrap();

    // filepath: plain string
    let filepath = props.get("filepath").unwrap();
    assert_eq!(filepath.get("type").unwrap(), "string");

    // offset / limit remain JSON-Schema-shaped (either ["integer","null"] or
    // {"type":"integer"} with some nullability marker — we just assert the
    // schema is present and not mutated into Gemini-style "INTEGER").
    let offset = props.get("offset").unwrap();
    let offset_type = offset.get("type").unwrap();
    assert!(
        offset_type.is_string() || offset_type.is_array(),
        "offset type should be a JSON Schema type field, got {:?}",
        offset_type
    );
    assert_ne!(offset_type, "INTEGER");
}

#[test]
fn test_claude_tool_preserves_nested_objects() {
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
    let claude_tool = ClaudeTool::from_tool(&tool);
    let json = serde_json::to_value(&claude_tool).unwrap();

    let schema = json.get("input_schema").unwrap();
    assert_eq!(schema.get("type").unwrap(), "object");

    let props = schema.get("properties").unwrap();

    // tags should be an array with string items (JSON Schema, lowercase)
    let tags = props.get("tags").unwrap();
    assert_eq!(tags.get("type").unwrap(), "array");
    let items = tags.get("items").unwrap();
    assert_eq!(items.get("type").unwrap(), "string");
}

#[test]
fn test_claude_tool_emits_required_and_name_and_description() {
    #[derive(JsonSchema, Serialize, Deserialize)]
    struct Args {
        city: String,
    }

    let tool = Tool::new("get_weather", "Get the weather for a city")
        .with_parameter::<Args>()
        .unwrap();
    let claude_tool = ClaudeTool::from_tool(&tool);
    let json = serde_json::to_value(&claude_tool).unwrap();

    assert_eq!(json.get("name").unwrap(), "get_weather");
    assert_eq!(json.get("description").unwrap(), "Get the weather for a city");

    let schema = json.get("input_schema").unwrap();
    let required = schema.get("required").unwrap().as_array().unwrap();
    assert!(required.iter().any(|v| v == "city"));
}
