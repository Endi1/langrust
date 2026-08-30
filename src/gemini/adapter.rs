use crate::{
    client::{Completion, FunctionCall, MessageType, ModelRequest, Role, StreamEvent, Usage},
    gemini::types::{
        Content, FunctionCallPart, FunctionResponsePart, GeminiRequest, GeminiResponse, GeminiTool,
        GeminiTools, GenerationConfig, Part, SystemInstructionContent, ThinkingConfig,
    },
    provider::{BoxError, ProviderAdapter},
};

#[derive(Debug, Default, Clone, Copy)]
pub struct GeminiAdapter;

impl ProviderAdapter for GeminiAdapter {
    const NAME: &'static str = "Gemini";
    type Request = GeminiRequest;
    type StreamState = ();

    fn build_body(&self, request: &ModelRequest, _model: &str, _stream: bool) -> GeminiRequest {
        // Model and streaming are part of the endpoint URL, not the body.
        let settings = request.settings.as_ref();

        let thinking_config = settings
            .and_then(|s| s.thinking_budget)
            .map(|thinking_budget| ThinkingConfig { thinking_budget });

        let generation_config = GenerationConfig {
            max_output_tokens: settings.and_then(|s| s.max_tokens),
            temperature: settings.and_then(|s| s.temperature),
            thinking_config,
        };

        let contents: Vec<Content> = request
            .messages
            .clone()
            .unwrap_or_default()
            .iter()
            .map(|message| match &message.message_type {
                MessageType::Text => Content {
                    parts: vec![Part::Text {
                        text: message.content.clone(),
                    }],
                    role: message.role.clone().unwrap_or(Role::User),
                },
                MessageType::FunctionCall(fc) => Content {
                    parts: vec![Part::FunctionCall {
                        function_call: FunctionCallPart {
                            name: fc.name.clone(),
                            args: fc.args.clone(),
                        },
                    }],
                    role: Role::Model,
                },
                MessageType::FunctionResponse { name, response } => Content {
                    parts: vec![Part::FunctionResponse {
                        function_response: FunctionResponsePart {
                            name: name.clone(),
                            response: response.clone().unwrap_or(serde_json::Value::Null),
                        },
                    }],
                    role: Role::User,
                },
            })
            .collect();

        let system_instruction = request.system.clone().map(|m| SystemInstructionContent {
            parts: vec![Part::Text { text: m }],
        });

        GeminiRequest {
            system_instruction,
            contents,
            generation_config,
            tools: request.tools.as_ref().map(|ts| {
                vec![GeminiTools {
                    function_declarations: ts.iter().map(GeminiTool::from_tool).collect(),
                }]
            }),
        }
    }

    fn parse_completion(&self, body: &[u8]) -> Result<Completion, BoxError> {
        let response_body: GeminiResponse = serde_json::from_slice(body)?;

        let content = response_body
            .get_text()
            .ok_or_else(|| -> BoxError { "Missing completion from response".into() })?;

        Ok(Completion {
            completion: content,
            usage: Usage {
                prompt_tokens: response_body.get_prompt_tokens().unwrap_or(0),
                completion_tokens: response_body.get_completion_tokens().unwrap_or(0),
                total_tokens: response_body.get_total_tokens().unwrap_or(0),
            },
            function: response_body.get_function().map(|gf| FunctionCall {
                name: gf.name,
                args: gf.args,
            }),
        })
    }

    fn map_sse_event(&self, data: &str, _state: &mut ()) -> Vec<StreamEvent> {
        let response: GeminiResponse = match serde_json::from_str(data) {
            Ok(r) => r,
            Err(e) => return vec![StreamEvent::Error(e.to_string())],
        };

        let mut events = Vec::new();

        if let Some(text) = response.get_text() {
            if !text.is_empty() {
                events.push(StreamEvent::Delta(text));
            }
        }

        if let Some(gf) = response.get_function() {
            events.push(StreamEvent::FunctionCall(FunctionCall {
                name: gf.name,
                args: gf.args,
            }));
        }

        if let Some(usage) = &response.usage_metadata {
            if let (Some(pt), Some(ct), Some(tt)) = (
                usage.prompt_token_count,
                usage.candidates_token_count,
                usage.total_token_count,
            ) {
                events.push(StreamEvent::Usage(Usage {
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    total_tokens: tt,
                }));
            }
        }

        events
    }
}
