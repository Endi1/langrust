use futures::{TryFutureExt, stream};
use std::error::Error;

use reqwest::RequestBuilder;

use crate::{
    client::{Completion, FunctionCall, ModelRequest, Role, StreamEvent, StreamResult},
    gemini::types::{
        Content, GeminiRequest, GeminiResponse, GeminiTool, GeminiTools, GenerationConfig, Part,
        SystemInstructionContent, ThinkingConfig,
    },
};

pub trait GeminiClient {
    fn model(&self) -> String;
    fn create_request_body(&self, request: ModelRequest) -> GeminiRequest {
        let thinking_config = if !self.model().contains("1.5") && !self.model().contains("2.0") {
            Some(ThinkingConfig {
                thinking_budget: request
                    .settings
                    .clone()
                    .map(|s| s.thinking_budget.unwrap_or_default())
                    .unwrap_or_default(),
            })
        } else {
            None
        };

        let generation_config = GenerationConfig {
            max_output_tokens: request.settings.clone().and_then(|s| s.max_tokens),
            temperature: request
                .settings
                .clone()
                .map(|s| s.temperature.unwrap_or_default())
                .unwrap_or_default(),
            thinking_config,
        };

        let contents: Vec<Content> = request
            .messages
            .clone()
            .unwrap_or(vec![])
            .iter()
            .map(|message| Content {
                parts: Vec::from([Part {
                    text: message.content.clone(),
                }]),
                role: message.role.clone().unwrap_or_else(|| Role::User),
            })
            .collect();

        let system_instruction = request.system.clone().map(|m| SystemInstructionContent {
            parts: vec![Part { text: m }],
        });

        let req = GeminiRequest {
            system_instruction,
            contents,
            generation_config,
            tools: request.tools.clone().map(|ts| {
                vec![GeminiTools {
                    function_declarations: ts
                        .clone()
                        .iter()
                        .map(|t| GeminiTool::from_tool(t))
                        .collect(),
                }]
            }),
        };
        req
    }

    async fn generate_content(
        &self,
        request: ModelRequest,
    ) -> Result<Completion, Box<dyn Error + Send + Sync>> {
        let endpoint = self.get_endpoint(&self.model(), String::from("generateContent"));
        let request_body = self.create_request_body(request);
        let response = self
            .build_request(&endpoint, &request_body)
            .await?
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().map_err(|e| e.to_string()).await?;
            return Err(format!(
                "Gemini request failed with status {}: {}",
                status, error_text
            )
            .into());
        }

        let response_body: GeminiResponse = response.json().await?;

        let content: String =
            response_body
                .get_text()
                .ok_or_else(|| -> Box<dyn Error + Send + Sync> {
                    "Missing completion from response".into()
                })?;

        let prompt_tokens =
            response_body
                .get_prompt_tokens()
                .ok_or_else(|| -> Box<dyn Error + Send + Sync> {
                    "Missing prompt tokens from response".into()
                })?;

        let completion_tokens = response_body.get_completion_tokens().ok_or_else(
            || -> Box<dyn Error + Send + Sync> { "Missing completion tokens from response".into() },
        )?;
        let total_tokens =
            response_body
                .get_total_tokens()
                .ok_or_else(|| -> Box<dyn Error + Send + Sync> {
                    "Missing total tokens from response".into()
                })?;

        return Ok(Completion {
            completion: content,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            function: response_body.get_function().map(|gf| FunctionCall {
                name: gf.name,
                args: gf.args,
            }),
        });
    }

    async fn stream_generate_content(
        &self,
        request: ModelRequest,
    ) -> Result<StreamResult, Box<dyn Error + Send + Sync>> {
        let endpoint = self.get_endpoint(
            &self.model(),
            String::from("streamGenerateContent?alt=sse"),
        );
        let request_body = self.create_request_body(request);
        let response = self
            .build_request(&endpoint, &request_body)
            .await?
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().map_err(|e| e.to_string()).await?;
            return Err(format!(
                "Gemini streaming request failed with status {}: {}",
                status, error_text
            )
            .into());
        }

        use futures::StreamExt;
        use bytes::BytesMut;

        let byte_stream = response.bytes_stream();

        let event_stream = stream::unfold(
            (byte_stream.boxed(), BytesMut::new()),
            |(mut byte_stream, mut buffer)| async move {
                loop {
                    // Try to extract a complete SSE event from the buffer
                    let buf_str = String::from_utf8_lossy(&buffer).to_string();
                    if let Some(event_end) = buf_str.find("\n\n") {
                        let event_text = buf_str[..event_end].to_string();
                        let remaining = buf_str[event_end + 2..].to_string();
                        buffer = BytesMut::from(remaining.as_bytes());

                        // Parse SSE: lines starting with "data: "
                        let data: String = event_text
                            .lines()
                            .filter_map(|line| line.strip_prefix("data: "))
                            .collect::<Vec<_>>()
                            .join("");

                        if data.is_empty() {
                            continue;
                        }

                        let parsed: Result<GeminiResponse, _> = serde_json::from_str(&data);
                        match parsed {
                            Ok(gemini_response) => {
                                let mut events: Vec<
                                    Result<StreamEvent, Box<dyn Error + Send + Sync>>,
                                > = Vec::new();

                                if let Some(text) = gemini_response.get_text() {
                                    if !text.is_empty() {
                                        events.push(Ok(StreamEvent::Delta(text)));
                                    }
                                }

                                if let Some(gf) = gemini_response.get_function() {
                                    events.push(Ok(StreamEvent::FunctionCall(FunctionCall {
                                        name: gf.name,
                                        args: gf.args,
                                    })));
                                }

                                if let Some(usage) = &gemini_response.usage_metadata {
                                    if let (Some(pt), Some(ct), Some(tt)) = (
                                        usage.prompt_token_count,
                                        usage.candidates_token_count,
                                        usage.total_token_count,
                                    ) {
                                        events.push(Ok(StreamEvent::Usage {
                                            prompt_tokens: pt,
                                            completion_tokens: ct,
                                            total_tokens: tt,
                                        }));
                                    }
                                }

                                if !events.is_empty() {
                                    return Some((
                                        stream::iter(events),
                                        (byte_stream, buffer),
                                    ));
                                }
                                continue;
                            }
                            Err(e) => {
                                return Some((
                                    stream::iter(vec![Err(Box::new(e)
                                        as Box<dyn Error + Send + Sync>)]),
                                    (byte_stream, buffer),
                                ));
                            }
                        }
                    }

                    // Need more data from the network
                    use futures::TryStreamExt;
                    match byte_stream.try_next().await {
                        Ok(Some(chunk)) => {
                            buffer.extend_from_slice(&chunk);
                        }
                        Ok(None) => {
                            // Stream ended
                            return None;
                        }
                        Err(e) => {
                            return Some((
                                stream::iter(vec![Err(
                                    Box::new(e) as Box<dyn Error + Send + Sync>
                                )]),
                                (byte_stream, buffer),
                            ));
                        }
                    }
                }
            },
        )
        .flat_map(|s| s);

        Ok(Box::pin(event_stream))
    }

    fn get_endpoint(&self, model: &String, method: String) -> String;
    async fn build_request(
        &self,
        endpoint: &String,
        request_body: &GeminiRequest,
    ) -> Result<RequestBuilder, Box<dyn Error + Send + Sync>>;
}
