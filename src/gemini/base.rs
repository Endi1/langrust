use futures::TryFutureExt;
use std::error::Error;

use reqwest::RequestBuilder;

use crate::{
    client::{Completion, FunctionCall, ModelRequest, Role},
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

        return Ok(Completion {
            completion: content,
            prompt_tokens,
            completion_tokens,
            function: response_body.get_function().map(|gf| FunctionCall {
                name: gf.name,
                args: gf.args,
            }),
        });
    }

    fn get_endpoint(&self, model: &String, method: String) -> String;
    async fn build_request(
        &self,
        endpoint: &String,
        request_body: &GeminiRequest,
    ) -> Result<RequestBuilder, Box<dyn Error + Send + Sync>>;
}
