use reqwest::Client;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct MetadataTokenResponse {
    access_token: String,
}

pub async fn get_access_token() -> Result<String, String> {
    let remote_token = get_access_token_server().await;
    if remote_token.is_ok() {
        return remote_token;
    }

    return get_access_token_local().await;
}

async fn get_access_token_local() -> Result<String, String> {
    use std::process::Command;

    let output = Command::new("gcloud")
        .args(&["auth", "print-access-token"])
        .output()
        .map_err(|e| format!("Failed to run gcloud command: {}", e))?;

    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr);
        return Err(format!("gcloud auth failed: {}", error));
    }

    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(token)
}

async fn get_access_token_server() -> Result<String, String> {
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;

    // Google Cloud metadata server endpoint
    let metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";

    let response = client
        .get(metadata_url)
        .header("Metadata-Flavor", "Google")
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!(
            "Metadata server returned error {}: {}. Make sure your service account has proper permissions.",
            status, error_text
        ));
    }

    let token_response: MetadataTokenResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse metadata server response: {}", e))?;

    Ok(token_response.access_token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_access_token() {
        let response = get_access_token().await;
        assert!(response.is_ok());

        let access_token = response.unwrap();
        assert!(!access_token.is_empty());
    }
}
