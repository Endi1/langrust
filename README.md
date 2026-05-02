# langrust

A unified, async Rust API for working with multiple LLM providers behind a single
ergonomic interface. Swap providers without rewriting your call sites: messages,
tools, streaming and function-calling all use the same types regardless of which
backend you target.

## Features

- One `Model` trait implemented by every provider
- Fluent `ModelRequestBuilder` for composing requests
- Streaming completions (`StreamEvent::Delta`, `Usage`, `FunctionCall`, `Error`)
- Function / tool calling with JSON Schema generated from your Rust types via
  [`schemars`](https://docs.rs/schemars)
- System prompts, temperature, max tokens and thinking budget via `Settings`
- Provider-agnostic `Message`, `Role`, `FunctionCall`, `Usage` and `Completion` types

## Supported providers and models

| Provider        | Client type          | Auth                               |
|-----------------|----------------------|------------------------------------|
| Anthropic       | `ClaudeApiModel`     | `x-api-key` (API key)              |
| OpenAI          | `OpenAiApiModel`     | `Authorization: Bearer <key>`      |
| Google (Direct) | `GeminiApiModel`     | `x-goog-api-key` (API key)         |
| Google (Vertex) | `GeminiVertexModel`  | `gcloud` application default creds |

### Model enums

**Claude** (`ClaudeModel`)
- `Sonnet4_5`     → `claude-sonnet-4-5`
- `Opus4_6`       → `claude-opus-4-6`
- `Opus4_7`       → `claude-opus-4-7`

**OpenAI** (`OpenAiModel`)
- `Gpt5_4`        → `gpt-5.4`
- `Gpt5_4Mini`    → `gpt-5.4-mini`
- `Gpt5_4Nano`    → `gpt-5.4-nano`
- `Gpt5_5`    → `gpt-5.5`
- `Gpt5_3Codex`    → `gpt-5.3-codex`

**Gemini** (`GeminiModel`) — used by both `GeminiApiModel` and `GeminiVertexModel`
- `Gemini25Flash`      → `gemini-2.5-flash`
- `Gemini31Pro`        → `gemini-3.1-pro-preview`
- `Gemini3Flash`       → `gemini-3-flash-preview`
- `Gemini31FlashLite`  → `gemini-3.1-flash-lite-preview`

## Installation

Add [`langrust`](https://crates.io/crates/langrust) from crates.io:

```sh
cargo add langrust
```

Or add it manually to your `Cargo.toml`:

```toml
[dependencies]
langrust = "0.1"
tokio = { version = "1", features = ["full"] }
futures = "0.3"
schemars = "1"        # only needed if you define tools
serde = { version = "1", features = ["derive"] }
```
de = { version = "1", features = ["derive"] }
```

Environment variables used by the examples:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_KEY=...
VERTEX_PROJECT=...
```

## Quick start

A minimal completion against Claude:

```rust
use langrust::{ClaudeApiModel, ClaudeModel, Message, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = ClaudeApiModel {
        api_key: std::env::var("ANTHROPIC_API_KEY")?,
        client: reqwest::Client::new(),
        model: ClaudeModel::Sonnet4_5,
    };

    let completion = model
        .new_request()
        .with_system("You are a concise assistant.".to_string())
        .with_message(Message::user("Say hi in one word.".to_string()))
        .completion()
        .await?;

    println!("{}", completion.completion);
    println!("tokens: {:?}", completion.usage);
    Ok(())
}
```

## Example scenarios

### 1. OpenAI completion with settings

```rust
use langrust::{Message, Model, OpenAiApiModel, OpenAiModel, Settings};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = OpenAiApiModel {
        api_key: std::env::var("OPENAI_API_KEY")?,
        client: reqwest::Client::new(),
        model: OpenAiModel::Gpt5_4Mini,
    };

    let settings = Settings {
        max_tokens: Some(256),
        timeout: None,
        temperature: Some(0),
        thinking_budget: None,
    };

    let completion = model
        .new_request()
        .with_system("You translate to French.".to_string())
        .with_message(Message::user("Hello, world!".to_string()))
        .with_settings(settings)
        .completion()
        .await?;

    println!("{}", completion.completion);
    Ok(())
}
```

### 2. Gemini (direct API) — multi-turn conversation

```rust
use langrust::{GeminiApiModel, GeminiModel, Message, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = GeminiApiModel {
        api_key: std::env::var("GEMINI_KEY")?,
        client: reqwest::Client::new(),
        model: GeminiModel::Gemini25Flash,
    };

    let history = vec![
        Message::user("What is the capital of Japan?".to_string()),
        Message::model("Tokyo.".to_string()),
        Message::user("And its population?".to_string()),
    ];

    let out = model
        .new_request()
        .with_messages(history)
        .completion()
        .await?;

    println!("{}", out.completion);
    Ok(())
}
```

### 3. Gemini on Vertex AI (gcloud ADC)

`GeminiVertexModel` authenticates via `gcloud auth application-default login`
(or a service account activated through `gcloud`). No API key needed.

```rust
use langrust::{GeminiModel, GeminiVertexModel, Message, Model};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = GeminiVertexModel {
        project_name: std::env::var("VERTEX_PROJECT")?,
        client: reqwest::Client::new(),
        model: GeminiModel::Gemini31Pro,
    };

    let completion = model
        .new_request()
        .with_message(Message::user("Summarise the theory of relativity in one sentence.".to_string()))
        .completion()
        .await?;

    println!("{}", completion.completion);
    Ok(())
}
```

### 4. Streaming responses

Works identically across providers — call `.stream()` instead of `.completion()`
and consume a `Stream<Item = StreamEvent>`.

```rust
use futures::StreamExt;
use langrust::{ClaudeApiModel, ClaudeModel, Message, Model, StreamEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = ClaudeApiModel {
        api_key: std::env::var("ANTHROPIC_API_KEY")?,
        client: reqwest::Client::new(),
        model: ClaudeModel::Sonnet4_5,
    };

    let mut stream = model
        .new_request()
        .with_message(Message::user("Write a haiku about Rust.".to_string()))
        .stream()
        .await?;

    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::Delta(text)       => print!("{}", text),
            StreamEvent::Usage(u)          => eprintln!("\n[usage] {:?}", u),
            StreamEvent::FunctionCall(fc)  => eprintln!("\n[tool call] {:?}", fc),
            StreamEvent::Error(e)          => eprintln!("\n[error] {}", e),
        }
    }
    Ok(())
}
```

### 5. Tool / function calling

Define the arguments as a regular Rust struct and derive `JsonSchema`. The
schema is generated and translated to the provider's native tool format
(OpenAI `function`, Claude `tool_use`, Gemini `functionDeclarations`).

```rust
use langrust::{Message, Model, OpenAiApiModel, OpenAiModel, Tool};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct GetWeatherArgs {
    /// City name, e.g. "Berlin"
    city: String,
    /// Optional unit: "celsius" or "fahrenheit"
    unit: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = OpenAiApiModel {
        api_key: std::env::var("OPENAI_API_KEY")?,
        client: reqwest::Client::new(),
        model: OpenAiModel::Gpt5_4,
    };

    let tool = Tool::new("get_weather", "Fetch the current weather for a city.")
        .with_parameter::<GetWeatherArgs>()?;

    // Turn 1: model asks to call the tool
    let first = model
        .new_request()
        .with_message(Message::user("What's the weather in Berlin?".to_string()))
        .with_tool(tool.clone())
        .completion()
        .await?;

    if let Some(call) = first.function {
        println!("Model wants to call: {}({:?})", call.name, call.args);

        // Turn 2: pretend we ran the tool and feed the result back
        let tool_result = serde_json::json!({ "temp_c": 17, "conditions": "cloudy" });

        let final_resp = model
            .new_request()
            .with_message(Message::user("What's the weather in Berlin?".to_string()))
            .with_message(Message::function_call(call.clone()))
            .with_message(Message::function_result(call.name.clone(), tool_result))
            .with_tool(tool)
            .completion()
            .await?;

        println!("{}", final_resp.completion);
    } else {
        println!("{}", first.completion);
    }

    Ok(())
}
```

The same `Tool` value can be passed to `ClaudeApiModel`, `GeminiApiModel` or
`GeminiVertexModel` — the schema is translated automatically (including
Gemini's uppercase type names and nullable-handling).

## Core types cheat-sheet

- `Model` — trait with `completion()` and `stream_completion()`; all providers implement it.
- `ModelRequestBuilder` — returned by `model.new_request()`; chain `with_system`,
  `with_message`, `with_messages`, `with_settings`, `with_tool`, `with_tools`, then
  call `.completion().await` or `.stream().await`.
- `Message::user(..)`, `Message::model(..)`, `Message::function_call(..)`,
  `Message::function_result(name, value)` — constructors for every message shape.
- `Settings { max_tokens, timeout, temperature, thinking_budget }` — all `Option`.
- `Completion { completion, usage, function }` — unified non-streaming response.
- `StreamEvent` — `Delta | Usage | FunctionCall | Error` for streaming.

## Known limitations

- Tool-call IDs are synthesised deterministically from the function name on
  Claude and OpenAI. Two calls to the same tool in a single assistant turn will
  collide; this will be fixed by adding an explicit `id` field to `FunctionCall`.
- Claude "thinking" blocks are currently ignored in the response.

## License

MIT
