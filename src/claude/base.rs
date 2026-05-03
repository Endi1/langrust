use std::collections::HashMap;
use std::error::Error;

use eventsource_stream::Eventsource;
use futures::{StreamExt, TryFutureExt, stream};
use reqwest::RequestBuilder;

use crate::{
    claude::types::{
        BlockDelta, ClaudeMessage, ClaudeRequest, ClaudeResponse, ClaudeTool, ContentBlock,
        DEFAULT_MAX_TOKENS, ResponseBlock, StreamContentBlock, StreamingEvent, ThinkingConfig,
        synth_tool_use_id,
    },
    client::{
        Completion, FunctionCall, MessageType, Model, ModelRequest, StreamEvent, StreamResult,
        Usage,
    },
};

pub trait ClaudeClient: Model {
    fn create_request_body(&self, request: ModelRequest, stream: bool) -> ClaudeRequest {
        let settings = request.settings.clone();

        let max_tokens = settings
            .as_ref()
            .and_then(|s| s.max_tokens)
            .map(|v| v as i32)
            .unwrap_or(DEFAULT_MAX_TOKENS);

        let temperature = settings
            .as_ref()
            .and_then(|s| s.temperature)
            .map(|v| v as f32);

        // Extended thinking: enabled iff caller passed a non-zero budget.
        let thinking = settings
            .as_ref()
            .and_then(|s| s.thinking_budget)
            .filter(|b| *b > 0)
            .map(|b| ThinkingConfig {
                kind: "enabled",
                budget_tokens: b as i32,
            });

        let messages: Vec<ClaudeMessage> = request
            .messages
            .clone()
            .unwrap_or_default()
            .iter()
            .map(|m| match &m.message_type {
                MessageType::Text => ClaudeMessage {
                    role: match m.role {
                        Some(crate::client::Role::Model) => "assistant",
                        _ => "user",
                    },
                    content: vec![ContentBlock::Text {
                        text: m.content.clone(),
                    }],
                },
                MessageType::FunctionCall(fc) => ClaudeMessage {
                    role: "assistant",
                    content: vec![ContentBlock::ToolUse {
                        id: synth_tool_use_id(&fc.name),
                        name: fc.name.clone(),
                        input: fc.args.clone(),
                    }],
                },
                MessageType::FunctionResponse { name, response } => ClaudeMessage {
                    role: "user",
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: synth_tool_use_id(name),
                        content: response
                            .as_ref()
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "null".to_string()),
                    }],
                },
            })
            .collect();

        let tools = request
            .tools
            .clone()
            .map(|ts| ts.iter().map(ClaudeTool::from_tool).collect());

        ClaudeRequest {
            model: self.model_name(),
            max_tokens,
            system: request.system.clone(),
            messages,
            temperature,
            tools,
            thinking,
            stream: if stream { Some(true) } else { None },
        }
    }

    async fn generate_content(
        &self,
        request: ModelRequest,
    ) -> Result<Completion, Box<dyn Error + Send + Sync>> {
        let endpoint = self.get_endpoint();
        let body = self.create_request_body(request, false);
        let response = self.build_request(&endpoint, &body).await?.send().await?;

        let status = response.status();
        if !status.is_success() {
            let err = response.text().map_err(|e| e.to_string()).await?;
            return Err(format!("Claude request failed with status {}: {}", status, err).into());
        }

        let body: ClaudeResponse = response.json().await?;

        let mut text = String::new();
        let mut function: Option<FunctionCall> = None;
        for block in body.content {
            match block {
                ResponseBlock::Text { text: t } => text.push_str(&t),
                ResponseBlock::ToolUse { name, input, .. } => {
                    function = Some(FunctionCall { name, args: input });
                }
                ResponseBlock::Other => {}
            }
        }

        let total = body.usage.input_tokens + body.usage.output_tokens;
        Ok(Completion {
            completion: text,
            usage: Usage {
                prompt_tokens: body.usage.input_tokens,
                completion_tokens: body.usage.output_tokens,
                total_tokens: total,
            },
            function,
        })
    }

    async fn stream_generate_content(
        &self,
        request: ModelRequest,
    ) -> Result<StreamResult, Box<dyn Error + Send + Sync>> {
        let endpoint = self.get_endpoint();
        let body = self.create_request_body(request, true);
        let response = self.build_request(&endpoint, &body).await?.send().await?;

        let status = response.status();
        if !status.is_success() {
            let err = response.text().map_err(|e| e.to_string()).await?;
            return Err(format!(
                "Claude streaming request failed with status {}: {}",
                status, err
            )
            .into());
        }

        // State threaded through `unfold`. Defined at module scope below.
        let sse = Box::pin(response.bytes_stream().eventsource());
        let state = State {
            sse,
            buffer: std::collections::VecDeque::new(),
            tool_blocks: HashMap::new(),
            prompt_tokens: 0,
        };

        let out = stream::unfold(state, |mut state| async move {
            loop {
                if let Some(ev) = state.buffer.pop_front() {
                    return Some((ev, state));
                }

                let next = state.sse.next().await?;
                match next {
                    Err(e) => {
                        state.buffer.push_back(StreamEvent::Error(e.to_string()));
                    }
                    Ok(event) => {
                        if event.data.is_empty() || event.data == "[DONE]" {
                            continue;
                        }
                        let parsed: Result<StreamingEvent, _> = serde_json::from_str(&event.data);
                        match parsed {
                            Err(e) => state.buffer.push_back(StreamEvent::Error(e.to_string())),
                            Ok(ev) => handle_event(ev, &mut state),
                        }
                    }
                }
            }
        });

        Ok(Box::pin(out))
    }

    fn get_endpoint(&self) -> String;

    async fn build_request(
        &self,
        endpoint: &String,
        request_body: &ClaudeRequest,
    ) -> Result<RequestBuilder, Box<dyn Error + Send + Sync>>;
}

// Separate free function so we can mutate `State` fields cleanly.
fn handle_event(ev: StreamingEvent, state: &mut State) {
    match ev {
        StreamingEvent::MessageStart { message } => {
            state.set_prompt_tokens(message.usage.input_tokens);
        }
        StreamingEvent::ContentBlockStart {
            index,
            content_block,
        } => {
            if let StreamContentBlock::ToolUse { name, .. } = content_block {
                state.tool_block_insert(index, name);
            }
        }
        StreamingEvent::ContentBlockDelta { index, delta } => match delta {
            BlockDelta::TextDelta { text } => {
                if !text.is_empty() {
                    state.push_event(StreamEvent::Delta(text));
                }
            }
            BlockDelta::InputJsonDelta { partial_json } => {
                state.tool_block_append(index, &partial_json);
            }
            BlockDelta::Other => {}
        },
        StreamingEvent::ContentBlockStop { index } => {
            if let Some((name, json_buf)) = state.tool_block_take(index) {
                let args: HashMap<String, serde_json::Value> = if json_buf.is_empty() {
                    HashMap::new()
                } else {
                    match serde_json::from_str(&json_buf) {
                        Ok(v) => v,
                        Err(e) => {
                            state.push_event(StreamEvent::Error(format!(
                                "failed to parse streamed tool input JSON: {}",
                                e
                            )));
                            return;
                        }
                    }
                };
                state.push_event(StreamEvent::FunctionCall(FunctionCall { name, args }));
            }
        }
        StreamingEvent::MessageDelta { usage, .. } => {
            let prompt = state.prompt_tokens();
            state.push_event(StreamEvent::Usage(Usage {
                prompt_tokens: prompt,
                completion_tokens: usage.output_tokens,
                total_tokens: prompt + usage.output_tokens,
            }));
        }
        StreamingEvent::MessageStop => {}
        StreamingEvent::Ping => {}
        StreamingEvent::Error { error } => {
            state.push_event(StreamEvent::Error(error.message));
        }
        StreamingEvent::Other => {}
    }
}

// Streaming state (module-scope so `handle_event` can reference it).
struct State {
    sse: std::pin::Pin<
        Box<
            dyn futures::Stream<
                    Item = Result<
                        eventsource_stream::Event,
                        eventsource_stream::EventStreamError<reqwest::Error>,
                    >,
                > + Send,
        >,
    >,
    buffer: std::collections::VecDeque<StreamEvent>,
    tool_blocks: HashMap<u32, ToolBlockAcc>,
    prompt_tokens: i32,
}

struct ToolBlockAcc {
    name: String,
    json_buf: String,
}

impl State {
    fn push_event(&mut self, ev: StreamEvent) {
        self.buffer.push_back(ev);
    }
    fn set_prompt_tokens(&mut self, v: i32) {
        self.prompt_tokens = v;
    }
    fn prompt_tokens(&self) -> i32 {
        self.prompt_tokens
    }
    fn tool_block_insert(&mut self, index: u32, name: String) {
        self.tool_blocks.insert(
            index,
            ToolBlockAcc {
                name,
                json_buf: String::new(),
            },
        );
    }
    fn tool_block_append(&mut self, index: u32, s: &str) {
        if let Some(acc) = self.tool_blocks.get_mut(&index) {
            acc.json_buf.push_str(s);
        }
    }
    fn tool_block_take(&mut self, index: u32) -> Option<(String, String)> {
        self.tool_blocks
            .remove(&index)
            .map(|acc| (acc.name, acc.json_buf))
    }
}
