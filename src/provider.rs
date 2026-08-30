//! Provider abstraction: one generic client composed from two small traits.
//!
//! - [`ProviderAdapter`] — pure request/response conversions for one provider
//!   wire format (no I/O). Trivially unit-testable.
//! - [`Transport`] — endpoint + authentication for one provider backend
//!   (only I/O).
//! - [`LlmClient`] — composes the two and implements [`Model`] exactly once.
//!
//! Concrete client types like `ClaudeApiModel` are type aliases of
//! [`LlmClient`] with the right adapter/transport pair.

use async_trait::async_trait;
use serde::Serialize;

use crate::error::LlmError;
use crate::http::{require_success, sse_events};
use crate::request::{Model, ModelRequest};
use crate::types::{Completion, StreamEvent, StreamResult};

/// Which provider action a request is for. Transports use this to select the
/// endpoint when the URL differs between plain and streaming completions
/// (e.g. Gemini's `generateContent` vs `streamGenerateContent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Generate,
    Stream,
}

/// Pure request/response conversions for one provider wire format.
///
/// Implementations are stateless unit structs; per-stream accumulation
/// (e.g. partially streamed tool-call JSON) lives in [`Self::StreamState`].
pub trait ProviderAdapter: Default + Send + Sync + 'static {
    /// Provider name used in error messages.
    const NAME: &'static str;
    /// Wire request body type.
    type Request: Serialize;
    /// State threaded through an SSE stream (use `()` when none is needed).
    type StreamState: Default + Send + 'static;

    /// Convert the common request into the provider's wire request.
    fn build_body(&self, request: &ModelRequest, model: &str, stream: bool) -> Self::Request;

    /// Parse a successful non-streaming response body.
    fn parse_completion(&self, body: &[u8]) -> Result<Completion, LlmError>;

    /// Map one SSE `data:` payload to zero or more common stream events.
    fn map_sse_event(&self, data: &str, state: &mut Self::StreamState) -> Vec<StreamEvent>;
}

/// Endpoint + authentication for one provider backend.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send `body` as an authenticated JSON POST for `model`/`action` and
    /// return the raw response (status is checked by the caller).
    async fn send(
        &self,
        model: &str,
        action: Action,
        body: serde_json::Value,
    ) -> Result<reqwest::Response, LlmError>;
}

/// Generic LLM client: a [`ProviderAdapter`] (wire format) plus a
/// [`Transport`] (endpoint + auth). Implements [`Model`] once for all
/// providers.
pub struct LlmClient<A: ProviderAdapter, T: Transport> {
    pub adapter: A,
    pub transport: T,
    pub model: String,
}

impl<A: ProviderAdapter, T: Transport> LlmClient<A, T> {
    pub fn from_parts(transport: T, model: impl Into<String>) -> Self {
        LlmClient {
            adapter: A::default(),
            transport,
            model: model.into(),
        }
    }
}

#[async_trait]
impl<A: ProviderAdapter, T: Transport> Model for LlmClient<A, T> {
    async fn completion(&self, request: ModelRequest) -> Result<Completion, LlmError> {
        let body = serde_json::to_value(self.adapter.build_body(&request, &self.model, false))?;
        let response = self.transport.send(&self.model, Action::Generate, body).await?;
        let response = require_success(A::NAME, response).await?;
        let bytes = response.bytes().await?;
        self.adapter.parse_completion(&bytes)
    }

    async fn stream_completion(&self, request: ModelRequest) -> Result<StreamResult, LlmError> {
        let body = serde_json::to_value(self.adapter.build_body(&request, &self.model, true))?;
        let response = self.transport.send(&self.model, Action::Stream, body).await?;
        let response = require_success(A::NAME, response).await?;

        let adapter = A::default();
        Ok(sse_events(
            response,
            A::StreamState::default(),
            move |data, state| adapter.map_sse_event(data, state),
        ))
    }

    fn model_name(&self) -> String {
        self.model.clone()
    }
}
