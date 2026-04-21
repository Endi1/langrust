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

fn make_direct(model: GeminiModel) -> GeminiApiModel {
    GeminiApiModel {
        client: reqwest::Client::new(),
        api_key: env::var("GEMINI_KEY").expect("GEMINI_KEY env var must be set"),
        model,
    }
}

fn make_vertex(model: GeminiModel) -> GeminiVertexModel {
    GeminiVertexModel {
        region: env::var("VERTEX_REGION").expect("VERTEX_REGION env var must be set"),
        project_name: env::var("VERTEX_PROJECT").expect("VERTEX_PROJECT env var must be set"),
        client: reqwest::Client::new(),
        model,
    }
}

fn default_settings() -> Settings {
    Settings {
        max_tokens: Some(8000),
        timeout: None,
        temperature: None,
        thinking_budget: None,
    }
}

#[derive(Deserialize, Serialize, JsonSchema)]
struct WeatherRequest {
    city: String,
}

struct ExecutableTool<A, R> {
    tool: Tool,
    executable: fn(A) -> R,
}

impl<A, R> ExecutableTool<A, R> {
    pub fn new(tool: &Tool, executable: fn(A) -> R) -> ExecutableTool<A, R> {
        ExecutableTool {
            tool: tool.clone(),
            executable,
        }
    }
    pub fn run(self, arg: A) -> R {
        (self.executable)(arg)
    }
}

async fn run_generate_content<M: Model>(m: &M) {
    let response = m
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message::user("hello, how are you?".to_string()))
        .with_settings(default_settings())
        .completion()
        .await;
    assert!(response.is_ok(), "completion failed: {:?}", response.err());
}

async fn run_with_messages<M: Model>(m: &M) {
    let messages = vec![
        Message::user("hello, how are you?".to_string()),
        Message::model("I am fine, and you?".to_string()),
    ];
    let mut binding = m.new_request();
    let request_builder = binding
        .with_system("you are a helpful assistant".to_string())
        .with_messages(messages)
        .with_message(Message::user("I am fine, thanks for asking".to_string()))
        .with_settings(default_settings());

    let response = request_builder.completion().await;
    assert!(response.is_ok(), "completion failed: {:?}", response.err());
}

async fn run_function_call<M: Model>(m: &M) {
    let tool = Tool::new("get_weather", "Get the weather for a city")
        .with_parameter::<WeatherRequest>()
        .unwrap();

    let response = m
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message::user(
            "What is the weather like in Paris?".to_string(),
        ))
        .with_settings(default_settings())
        .with_tool(tool)
        .completion()
        .await;
    assert!(response.is_ok(), "completion failed: {:?}", response.err());
    assert_eq!(response.unwrap().function.unwrap().name, "get_weather");
}

async fn run_function_execution<M: Model>(m: &M) {
    let tool = Tool::new("get_weather", "Get the weather for a city")
        .with_parameter::<WeatherRequest>()
        .unwrap();

    let weather_tool: ExecutableTool<WeatherRequest, String> =
        ExecutableTool::new(&tool, |arg: WeatherRequest| {
            format!("The weather in {} is great!", arg.city)
        });

    let response = m
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message::user(
            "What is the weather like in Paris?".to_string(),
        ))
        .with_settings(default_settings())
        .with_tool(weather_tool.tool.clone())
        .completion()
        .await;
    assert!(response.is_ok(), "completion failed: {:?}", response.err());
    let function = response.unwrap().function.unwrap();
    let parsed_args: WeatherRequest =
        serde_json::from_value(serde_json::to_value(function.args).unwrap()).unwrap();

    let function_response = weather_tool.run(parsed_args);
    assert_eq!(function_response, "The weather in Paris is great!");
}

async fn run_stream_generate_content<M: Model>(m: &M) {
    let mut stream = m
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message::user("hello, how are you?".to_string()))
        .with_settings(default_settings())
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

async fn run_stream_function_call<M: Model>(m: &M) {
    let tool = Tool::new("get_weather", "Get the weather for a city")
        .with_parameter::<WeatherRequest>()
        .unwrap();

    let mut stream = m
        .new_request()
        .with_system("you are a helpful assistant".to_string())
        .with_message(Message::user(
            "What is the weather like in Paris?".to_string(),
        ))
        .with_settings(default_settings())
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

macro_rules! gemini_model_suite {
    ($mod_name:ident, $factory:ident, $model:expr) => {
        mod $mod_name {
            use super::*;

            #[tokio::test]
            async fn generate_content() {
                run_generate_content(&$factory($model)).await;
            }

            #[tokio::test]
            async fn with_messages() {
                run_with_messages(&$factory($model)).await;
            }

            #[tokio::test]
            async fn function_call() {
                run_function_call(&$factory($model)).await;
            }

            #[tokio::test]
            async fn function_execution() {
                run_function_execution(&$factory($model)).await;
            }

            #[tokio::test]
            async fn stream_generate_content() {
                run_stream_generate_content(&$factory($model)).await;
            }

            #[tokio::test]
            async fn stream_function_call() {
                run_stream_function_call(&$factory($model)).await;
            }
        }
    };
}

// Direct Gemini API (uses GEMINI_KEY).
mod direct {
    use super::*;
    gemini_model_suite!(gemini_2_5_flash, make_direct, GeminiModel::Gemini25Flash);
}

// Vertex AI (uses VERTEX_REGION + VERTEX_PROJECT + gcloud ADC).
mod vertex {
    use super::*;
    gemini_model_suite!(gemini_2_5_flash, make_vertex, GeminiModel::Gemini25Flash);
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
