use std::collections::HashMap;
use std::error::Error;

use eventsource_stream::Eventsource;
use futures::{StreamExt, TryFutureExt, stream};
use reqwest::RequestBuilder;

use crate::{
    client::{
        Completion, FunctionCall, MessageType, Model, ModelRequest, StreamEvent, StreamResult,
        Usage,
    },
    openai::types::{
        OpenAiInputItem, OpenAiRequest, OpenAiResponse, OpenAiTool, ResponsesStreamEvent,
        synth_call_id,
    },
};

pub trait OpenAiClient: Model {
    fn create_request_body(&self, request: ModelRequest, stream: bool) -> OpenAiRequest {
        let settings = request.settings.clone();

        let max_output_tokens = settings
            .as_ref()
            .and_then(|s| s.max_tokens)
            .map(|v| v as i32);

        let temperature = settings
            .as_ref()
            .and_then(|s| s.temperature)
            .map(|v| v as f32);

        // Build input items (no system message — that goes to `instructions`).
        let mut input: Vec<OpenAiInputItem> = Vec::new();

        for m in request.messages.clone().unwrap_or_default().iter() {
            match &m.message_type {
                MessageType::Text => match m.role {
                    Some(crate::client::Role::Model) => {
                        input.push(OpenAiInputItem::Message {
                            role: "assistant".to_string(),
                            content: m.content.clone(),
                        });
                    }
                    _ => {
                        input.push(OpenAiInputItem::Message {
                            role: "user".to_string(),
                            content: m.content.clone(),
                        });
                    }
                },
                MessageType::FunctionCall(fc) => {
                    let arguments = serde_json::to_string(&fc.args).unwrap_or("{}".to_string());
                    input.push(OpenAiInputItem::FunctionCall {
                        call_id: synth_call_id(&fc.name),
                        name: fc.name.clone(),
                        arguments,
                    });
                }
                MessageType::FunctionResponse { name, response } => {
                    input.push(OpenAiInputItem::FunctionCallOutput {
                        call_id: synth_call_id(name),
                        output: response
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

        let stream_flag = if stream { Some(true) } else { None };

        OpenAiRequest {
            model: self.model_name(),
            input,
            instructions: request.system.clone(),
            max_output_tokens,
            temperature,
            tools,
            stream: stream_flag,
            store: false,
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
            return Err(format!("OpenAI request failed with status {}: {}", status, err).into());
        }

        let body: OpenAiResponse = response.json().await?;

        let text = body.get_text();
        let function = body
            .get_function()
            .map(|(name, args)| FunctionCall { name, args });

        let usage = body.usage.map(|u| Usage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
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
                        if event.data.is_empty() {
                            continue;
                        }
                        let parsed: Result<ResponsesStreamEvent, _> =
                            serde_json::from_str(&event.data);
                        match parsed {
                            Err(e) => state.buffer.push_back(StreamEvent::Error(e.to_string())),
                            Ok(ev) => handle_stream_event(ev, &mut state),
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

fn handle_stream_event(event: ResponsesStreamEvent, state: &mut State) {
    match event.event_type.as_str() {
        "response.output_text.delta" => {
            if let Some(delta) = event.delta {
                if !delta.is_empty() {
                    state.push_event(StreamEvent::Delta(delta));
                }
            }
        }
        "response.output_item.done" => {
            if let Some(item) = event.item {
                if item.item_type.as_deref() == Some("function_call") {
                    if let Some(name) = item.name {
                        let args_str = item.arguments.unwrap_or_default();
                        let args: HashMap<String, serde_json::Value> = if args_str.is_empty() {
                            HashMap::new()
                        } else {
                            match serde_json::from_str(&args_str) {
                                Ok(v) => v,
                                Err(e) => {
                                    state.push_event(StreamEvent::Error(format!(
                                        "failed to parse streamed tool arguments JSON: {}",
                                        e
                                    )));
                                    return;
                                }
                            }
                        };
                        state.push_event(StreamEvent::FunctionCall(FunctionCall { name, args }));
                    }
                }
            }
        }
        "response.completed" => {
            if let Some(resp) = event.response {
                if let Some(usage) = resp.usage {
                    state.push_event(StreamEvent::Usage(Usage {
                        prompt_tokens: usage.input_tokens,
                        completion_tokens: usage.output_tokens,
                        total_tokens: usage.total_tokens,
                    }));
                }
            }
        }
        _ => {
            // Ignore other event types (response.created, response.in_progress, etc.)
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
}

impl State {
    fn push_event(&mut self, ev: StreamEvent) {
        self.buffer.push_back(ev);
    }
}
