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

use std::collections::VecDeque;
use std::error::Error;
use std::pin::Pin;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{StreamExt, stream};
use serde::Serialize;

use crate::client::{Completion, Model, ModelRequest, StreamEvent, StreamResult};

pub type BoxError = Box<dyn Error + Send + Sync>;

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
    fn parse_completion(&self, body: &[u8]) -> Result<Completion, BoxError>;

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
    ) -> Result<reqwest::Response, BoxError>;
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
    async fn completion(&self, request: ModelRequest) -> Result<Completion, BoxError> {
        let body = serde_json::to_value(self.adapter.build_body(&request, &self.model, false))?;
        let response = self.transport.send(&self.model, Action::Generate, body).await?;
        let response = check_status(A::NAME, response).await?;
        let bytes = response.bytes().await?;
        self.adapter.parse_completion(&bytes)
    }

    async fn stream_completion(&self, request: ModelRequest) -> Result<StreamResult, BoxError> {
        let body = serde_json::to_value(self.adapter.build_body(&request, &self.model, true))?;
        let response = self.transport.send(&self.model, Action::Stream, body).await?;
        let response = check_status(A::NAME, response).await?;
        Ok(sse_stream(A::default(), response))
    }

    fn model_name(&self) -> String {
        self.model.clone()
    }
}

/// Pass through successful responses; turn error statuses into a readable error.
async fn check_status(
    provider: &str,
    response: reqwest::Response,
) -> Result<reqwest::Response, BoxError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_default();
    Err(format!("{} request failed with status {}: {}", provider, status, body).into())
}

type SseEvents = Pin<
    Box<
        dyn futures::Stream<
                Item = Result<
                    eventsource_stream::Event,
                    eventsource_stream::EventStreamError<reqwest::Error>,
                >,
            > + Send,
    >,
>;

/// Turn an SSE response into a `StreamResult` by feeding each `data:` payload
/// through the adapter's [`ProviderAdapter::map_sse_event`].
fn sse_stream<A: ProviderAdapter>(adapter: A, response: reqwest::Response) -> StreamResult {
    struct State<A: ProviderAdapter> {
        adapter: A,
        provider_state: A::StreamState,
        buffer: VecDeque<StreamEvent>,
        sse: SseEvents,
    }

    let state = State {
        adapter,
        provider_state: A::StreamState::default(),
        buffer: VecDeque::new(),
        sse: Box::pin(response.bytes_stream().eventsource()),
    };

    let out = stream::unfold(state, |mut st| async move {
        loop {
            if let Some(ev) = st.buffer.pop_front() {
                return Some((ev, st));
            }
            match st.sse.next().await? {
                Err(e) => st.buffer.push_back(StreamEvent::Error(e.to_string())),
                Ok(event) => {
                    if event.data.is_empty() {
                        continue;
                    }
                    let events = st.adapter.map_sse_event(&event.data, &mut st.provider_state);
                    st.buffer.extend(events);
                }
            }
        }
    });

    Box::pin(out)
}
