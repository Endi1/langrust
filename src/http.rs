//! Shared HTTP/SSE plumbing used by every provider: status checking and the
//! SSE-to-`StreamEvent` pump. Exists exactly once.

use std::collections::VecDeque;
use std::pin::Pin;

use eventsource_stream::Eventsource;
use futures::{StreamExt, stream};

use crate::error::LlmError;
use crate::types::{StreamEvent, StreamResult};

/// Pass successful responses through; turn error statuses into [`LlmError::Http`].
pub async fn require_success(
    provider: &'static str,
    response: reqwest::Response,
) -> Result<reqwest::Response, LlmError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_default();
    Err(LlmError::Http {
        provider,
        status: status.as_u16(),
        body,
    })
}

type SseSource = Pin<
    Box<
        dyn futures::Stream<
                Item = Result<
                    eventsource_stream::Event,
                    eventsource_stream::EventStreamError<reqwest::Error>,
                >,
            > + Send,
    >,
>;

/// Turn an SSE response into a `StreamResult`.
///
/// Each non-empty `data:` payload is passed to `handler` together with the
/// caller's accumulation state; the handler returns zero or more common
/// stream events. Transport errors surface as [`StreamEvent::Error`].
pub fn sse_events<S: Send + 'static>(
    response: reqwest::Response,
    state: S,
    handler: impl Fn(&str, &mut S) -> Vec<StreamEvent> + Send + 'static,
) -> StreamResult {
    struct Pump<S, F> {
        sse: SseSource,
        state: S,
        handler: F,
        buffer: VecDeque<StreamEvent>,
    }

    let pump = Pump {
        sse: Box::pin(response.bytes_stream().eventsource()),
        state,
        handler,
        buffer: VecDeque::new(),
    };

    let out = stream::unfold(pump, |mut p| async move {
        loop {
            if let Some(ev) = p.buffer.pop_front() {
                return Some((ev, p));
            }
            match p.sse.next().await? {
                Err(e) => p.buffer.push_back(StreamEvent::Error(e.to_string())),
                Ok(event) => {
                    if event.data.is_empty() {
                        continue;
                    }
                    let events = (p.handler)(&event.data, &mut p.state);
                    p.buffer.extend(events);
                }
            }
        }
    });

    Box::pin(out)
}
