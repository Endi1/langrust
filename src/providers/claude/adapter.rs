use std::collections::HashMap;

use crate::{
    error::LlmError,
    provider::ProviderAdapter,
    request::ModelRequest,
    types::{Completion, FunctionCall, MessageType, StreamEvent, Usage},
};

use super::types::{
    BlockDelta, ClaudeMessage, ClaudeRequest, ClaudeResponse, ClaudeTool, ContentBlock,
    DEFAULT_MAX_TOKENS, ResponseBlock, StreamContentBlock, StreamingEvent, ThinkingConfig,
    synth_tool_use_id,
};

/// Pure conversions between the common request/response types and the
/// Anthropic Messages API wire format. No I/O.
#[derive(Debug, Default, Clone, Copy)]
pub struct ClaudeAdapter;

impl ProviderAdapter for ClaudeAdapter {
    const NAME: &'static str = "Claude";
    type Request = ClaudeRequest;
    type StreamState = ClaudeStreamState;

    fn build_body(&self, request: &ModelRequest, model: &str, stream: bool) -> ClaudeRequest {
        let settings = request.settings.as_ref();

        let max_tokens = settings
            .and_then(|s| s.max_tokens)
            .map(|v| v as i32)
            .unwrap_or(DEFAULT_MAX_TOKENS);

        let temperature = settings.and_then(|s| s.temperature);

        let thinking = settings
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
                        Some(crate::types::Role::Model) => "assistant",
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
            .as_ref()
            .map(|ts| ts.iter().map(ClaudeTool::from_tool).collect());

        ClaudeRequest {
            model: model.to_string(),
            max_tokens,
            system: request.system.clone(),
            messages,
            temperature,
            tools,
            thinking,
            stream: if stream { Some(true) } else { None },
        }
    }

    fn parse_completion(&self, body: &[u8]) -> Result<Completion, LlmError> {
        let body: ClaudeResponse = serde_json::from_slice(body)?;

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

    fn map_sse_event(&self, data: &str, state: &mut ClaudeStreamState) -> Vec<StreamEvent> {
        if data == "[DONE]" {
            return Vec::new();
        }
        let mut out = Vec::new();
        match serde_json::from_str::<StreamingEvent>(data) {
            Err(e) => out.push(StreamEvent::Error(e.to_string())),
            Ok(ev) => handle_event(ev, state, &mut out),
        }
        out
    }
}

#[derive(Default)]
pub struct ClaudeStreamState {
    tool_blocks: HashMap<u32, ToolBlockAcc>,
    prompt_tokens: i32,
}

struct ToolBlockAcc {
    name: String,
    json_buf: String,
}

fn handle_event(ev: StreamingEvent, state: &mut ClaudeStreamState, out: &mut Vec<StreamEvent>) {
    match ev {
        StreamingEvent::MessageStart { message } => {
            state.prompt_tokens = message.usage.input_tokens;
        }
        StreamingEvent::ContentBlockStart {
            index,
            content_block,
        } => {
            if let StreamContentBlock::ToolUse { name, .. } = content_block {
                state.tool_blocks.insert(
                    index,
                    ToolBlockAcc {
                        name,
                        json_buf: String::new(),
                    },
                );
            }
        }
        StreamingEvent::ContentBlockDelta { index, delta } => match delta {
            BlockDelta::TextDelta { text } => {
                if !text.is_empty() {
                    out.push(StreamEvent::Delta(text));
                }
            }
            BlockDelta::InputJsonDelta { partial_json } => {
                if let Some(acc) = state.tool_blocks.get_mut(&index) {
                    acc.json_buf.push_str(&partial_json);
                }
            }
            BlockDelta::Other => {}
        },
        StreamingEvent::ContentBlockStop { index } => {
            if let Some(acc) = state.tool_blocks.remove(&index) {
                let args: HashMap<String, serde_json::Value> = if acc.json_buf.is_empty() {
                    HashMap::new()
                } else {
                    match serde_json::from_str(&acc.json_buf) {
                        Ok(v) => v,
                        Err(e) => {
                            out.push(StreamEvent::Error(format!(
                                "failed to parse streamed tool input JSON: {}",
                                e
                            )));
                            return;
                        }
                    }
                };
                out.push(StreamEvent::FunctionCall(FunctionCall {
                    name: acc.name,
                    args,
                }));
            }
        }
        StreamingEvent::MessageDelta { usage, .. } => {
            out.push(StreamEvent::Usage(Usage {
                prompt_tokens: state.prompt_tokens,
                completion_tokens: usage.output_tokens,
                total_tokens: state.prompt_tokens + usage.output_tokens,
            }));
        }
        StreamingEvent::MessageStop => {}
        StreamingEvent::Ping => {}
        StreamingEvent::Error { error } => {
            out.push(StreamEvent::Error(error.message));
        }
        StreamingEvent::Other => {}
    }
}
