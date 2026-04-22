use std::collections::HashMap;
use std::error::Error;

use eventsource_stream::Eventsource;
use futures::{StreamExt, TryFutureExt, stream};
use reqwest::RequestBuilder;

use crate::{
    client::{
        Completion, FunctionCall, MessageType, ModelRequest, StreamEvent, StreamResult, Usage,
    },
    openai::types::{
        OpenAiFunctionCall, OpenAiMessage, OpenAiRequest, OpenAiResponse, OpenAiTool,
        OpenAiToolCall, StreamChunk, StreamOptions, synth_tool_call_id,
    },
};

pub trait OpenAiClient {
    fn model(&self) -> String;

    fn create_request_body(&self, request: ModelRequest, stream: bool) -> OpenAiRequest {
        let settings = request.settings.clone();

        let max_completion_tokens = settings
            .as_ref()
            .and_then(|s| s.max_tokens)
            .map(|v| v as i32);

        let temperature = settings
            .as_ref()
            .and_then(|s| s.temperature)
            .map(|v| v as f32);

        let mut messages: Vec<OpenAiMessage> = Vec::new();

        if let Some(system) = request.system.clone() {
            messages.push(OpenAiMessage::System {
                role: "system",
                content: system,
            });
        }

        for m in request.messages.clone().unwrap_or_default().iter() {
            match &m.message_type {
                MessageType::Text => match m.role {
                    Some(crate::client::Role::Model) => {
                        messages.push(OpenAiMessage::Assistant {
                            role: "assistant",
                            content: Some(m.content.clone()),
                            tool_calls: None,
                        });
                    }
                    _ => {
                        messages.push(OpenAiMessage::User {
                            role: "user",
                            content: m.content.clone(),
                        });
                    }
                },
                MessageType::FunctionCall(fc) => {
                    let arguments = serde_json::to_string(&fc.args).unwrap_or("{}".to_string());
                    messages.push(OpenAiMessage::Assistant {
                        role: "assistant",
                        content: None,
                        tool_calls: Some(vec![OpenAiToolCall {
                            id: synth_tool_call_id(&fc.name),
                            kind: "function",
                            function: OpenAiFunctionCall {
                                name: fc.name.clone(),
                                arguments,
                            },
                        }]),
                    });
                }
                MessageType::FunctionResponse { name, response } => {
                    messages.push(OpenAiMessage::Tool {
                        role: "tool",
                        tool_call_id: synth_tool_call_id(name),
                        content: response
                            .as_ref()
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "null".to_string()),
                    });
                }
            }
        }

        let tools = request
            .tools
            .clone()
            .map(|ts| ts.iter().map(OpenAiTool::from_tool).collect());

        let (stream_flag, stream_options) = if stream {
            (
                Some(true),
                Some(StreamOptions {
                    include_usage: true,
                }),
            )
        } else {
            (None, None)
        };

        OpenAiRequest {
            model: self.model(),
            messages,
            max_completion_tokens,
            temperature,
            tools,
            stream: stream_flag,
            stream_options,
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
            return Err(
                format!("OpenAI request failed with status {}: {}", status, err).into(),
            );
        }

        let body: OpenAiResponse = response.json().await?;

        let text = body.get_text();
        let function = body.get_function().map(|(name, args)| FunctionCall {
            name,
            args,
        });

        let usage = body.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(Completion {
            completion: text,
            usage: usage.unwrap_or(Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            }),
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
                "OpenAI streaming request failed with status {}: {}",
                status, err
            )
            .into());
        }

        let sse = Box::pin(response.bytes_stream().eventsource());
        let state = State {
            sse,
            buffer: std::collections::VecDeque::new(),
            tool_calls: HashMap::new(),
        };

        let out = stream::unfold(state, |mut state| async move {
            loop {
                if let Some(ev) = state.buffer.pop_front() {
                    return Some((ev, state));
                }

                let next = state.sse.next().await?;
                match next {
                    Err(e) => {
                        state
                            .buffer
                            .push_back(StreamEvent::Error(e.to_string()));
                    }
                    Ok(event) => {
                        if event.data.is_empty() || event.data == "[DONE]" {
                            // On [DONE], emit any completed tool calls.
                            if event.data == "[DONE]" {
                                flush_tool_calls(&mut state);
                            }
                            continue;
                        }
                        let parsed: Result<StreamChunk, _> = serde_json::from_str(&event.data);
                        match parsed {
                            Err(e) => state
                                .buffer
                                .push_back(StreamEvent::Error(e.to_string())),
                            Ok(chunk) => handle_chunk(chunk, &mut state),
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
        request_body: &OpenAiRequest,
    ) -> Result<RequestBuilder, Box<dyn Error + Send + Sync>>;
}

fn handle_chunk(chunk: StreamChunk, state: &mut State) {
    for choice in chunk.choices {
        if let Some(text) = choice.delta.content {
            if !text.is_empty() {
                state.push_event(StreamEvent::Delta(text));
            }
        }

        if let Some(tool_calls) = choice.delta.tool_calls {
            for tc in tool_calls {
                let entry = state.tool_calls.entry(tc.index).or_insert_with(|| {
                    ToolCallAcc {
                        name: String::new(),
                        arguments: String::new(),
                    }
                });
                if let Some(func) = tc.function {
                    if let Some(name) = func.name {
                        if !name.is_empty() {
                            entry.name.push_str(&name);
                        }
                    }
                    if let Some(args) = func.arguments {
                        entry.arguments.push_str(&args);
                    }
                }
            }
        }

        if choice.finish_reason.is_some() {
            // End of this choice — emit any accumulated tool calls.
            flush_tool_calls(state);
        }
    }

    if let Some(usage) = chunk.usage {
        state.push_event(StreamEvent::Usage(Usage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        }));
    }
}

fn flush_tool_calls(state: &mut State) {
    if state.tool_calls.is_empty() {
        return;
    }
    let mut indices: Vec<u32> = state.tool_calls.keys().copied().collect();
    indices.sort();
    for idx in indices {
        if let Some(acc) = state.tool_calls.remove(&idx) {
            if acc.name.is_empty() {
                continue;
            }
            let args: HashMap<String, serde_json::Value> = if acc.arguments.is_empty() {
                HashMap::new()
            } else {
                match serde_json::from_str(&acc.arguments) {
                    Ok(v) => v,
                    Err(e) => {
                        state.push_event(StreamEvent::Error(format!(
                            "failed to parse streamed tool arguments JSON: {}",
                            e
                        )));
                        continue;
                    }
                }
            };
            state.push_event(StreamEvent::FunctionCall(FunctionCall {
                name: acc.name,
                args,
            }));
        }
    }
}

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
    tool_calls: HashMap<u32, ToolCallAcc>,
}

struct ToolCallAcc {
    name: String,
    arguments: String,
}

impl State {
    fn push_event(&mut self, ev: StreamEvent) {
        self.buffer.push_back(ev);
    }
}
