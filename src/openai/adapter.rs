use std::collections::HashMap;

use crate::{
    client::{Completion, FunctionCall, MessageType, ModelRequest, StreamEvent, Usage},
    openai::types::{
        OpenAiInputItem, OpenAiRequest, OpenAiResponse, OpenAiTool, ResponsesStreamEvent,
        synth_call_id,
    },
    provider::{BoxError, ProviderAdapter},
};

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAiAdapter;

impl ProviderAdapter for OpenAiAdapter {
    const NAME: &'static str = "OpenAI";
    type Request = OpenAiRequest;
    type StreamState = ();

    fn build_body(&self, request: &ModelRequest, model: &str, stream: bool) -> OpenAiRequest {
        let settings = request.settings.as_ref();

        let max_output_tokens = settings.and_then(|s| s.max_tokens).map(|v| v as i32);
        let temperature = settings.and_then(|s| s.temperature);

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
            .as_ref()
            .map(|ts| ts.iter().map(OpenAiTool::from_tool).collect());

        OpenAiRequest {
            model: model.to_string(),
            input,
            instructions: request.system.clone(),
            max_output_tokens,
            temperature,
            tools,
            stream: if stream { Some(true) } else { None },
            store: false,
        }
    }

    fn parse_completion(&self, body: &[u8]) -> Result<Completion, BoxError> {
        let body: OpenAiResponse = serde_json::from_slice(body)?;

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

    fn map_sse_event(&self, data: &str, _state: &mut ()) -> Vec<StreamEvent> {
        if data == "[DONE]" {
            return Vec::new();
        }
        let mut out = Vec::new();
        match serde_json::from_str::<ResponsesStreamEvent>(data) {
            Err(e) => out.push(StreamEvent::Error(e.to_string())),
            Ok(ev) => handle_event(ev, &mut out),
        }
        out
    }
}

fn handle_event(event: ResponsesStreamEvent, out: &mut Vec<StreamEvent>) {
    match event.event_type.as_str() {
        "response.output_text.delta" => {
            if let Some(delta) = event.delta {
                if !delta.is_empty() {
                    out.push(StreamEvent::Delta(delta));
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
                                    out.push(StreamEvent::Error(format!(
                                        "failed to parse streamed tool arguments JSON: {}",
                                        e
                                    )));
                                    return;
                                }
                            }
                        };
                        out.push(StreamEvent::FunctionCall(FunctionCall { name, args }));
                    }
                }
            }
        }
        "response.completed" => {
            if let Some(resp) = event.response {
                if let Some(usage) = resp.usage {
                    out.push(StreamEvent::Usage(Usage {
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
