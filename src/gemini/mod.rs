mod base;
mod direct_api_client;
mod gcloud_helpers;
mod types;
mod vertex_client;

#[cfg(test)]
mod tests;

pub use direct_api_client::GeminiApiModel;
pub use types::GeminiModel;
pub use vertex_client::GeminiVertexModel;
