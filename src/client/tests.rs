use super::*;
use async_trait::async_trait;
use schemars::JsonSchema;

struct MockModel;

#[async_trait]
impl Model for MockModel {
    async fn completion(
        &self,
        _request: ModelRequest,
    ) -> Result<Completion, Box<dyn Error + Send + Sync>> {
        Ok(Completion {
            completion: "test".to_string(),
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            function: None,
        })
    }
}

#[test]
fn test_new_request_builder() {
    let model = MockModel;
    let builder = ModelRequestBuilder::new(&model);

    assert!(builder.system.is_none());
    assert!(builder.messages.is_none());
    assert!(builder.settings.is_none());
    assert!(builder.tools.is_none());
}

#[test]
fn test_with_system() {
    let model = MockModel;
    let mut builder = ModelRequestBuilder::new(&model);
    builder.with_system("You are a helpful assistant.".to_string());

    assert_eq!(
        builder.system,
        Some("You are a helpful assistant.".to_string())
    );
}

#[test]
fn test_with_message_single() {
    let model = MockModel;
    let mut builder = ModelRequestBuilder::new(&model);
    builder.with_message(Message::user("Hello".to_string()));

    let messages = builder.messages.unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].content, "Hello");
    assert_eq!(messages[0].role, Some(Role::User));
}

#[test]
fn test_with_message_multiple() {
    let model = MockModel;
    let mut builder = ModelRequestBuilder::new(&model);
    builder
        .with_message(Message::user("Hello".to_string()))
        .with_message(Message::model("Hi there!".to_string()));

    let messages = builder.messages.unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content, "Hello");
    assert_eq!(messages[1].content, "Hi there!");
}

#[test]
fn test_with_messages() {
    let model = MockModel;
    let mut builder = ModelRequestBuilder::new(&model);
    let msgs = vec![
        Message::user("First".to_string()),
        Message::model("Second".to_string()),
    ];
    builder.with_messages(msgs);

    let messages = builder.messages.unwrap();
    assert_eq!(messages.len(), 2);
}

#[test]
fn test_with_messages_extends_existing() {
    let model = MockModel;
    let mut builder = ModelRequestBuilder::new(&model);
    builder.with_message(Message::user("Existing".to_string()));
    builder.with_messages(vec![
        Message::user("New1".to_string()),
        Message::user("New2".to_string()),
    ]);

    let messages = builder.messages.unwrap();
    assert_eq!(messages.len(), 3);
}

#[test]
fn test_with_settings() {
    let model = MockModel;
    let mut builder = ModelRequestBuilder::new(&model);
    let settings = Settings {
        max_tokens: Some(100),
        timeout: Some(30),
        temperature: Some(7),
        thinking_budget: None,
    };
    builder.with_settings(settings);

    let s = builder.settings.unwrap();
    assert_eq!(s.max_tokens, Some(100));
    assert_eq!(s.timeout, Some(30));
    assert_eq!(s.temperature, Some(7));
}

#[test]
fn test_with_tool() {
    let model = MockModel;
    let mut builder = ModelRequestBuilder::new(&model);
    let tool = Tool::new("test_tool", "A test tool");
    builder.with_tool(tool);

    let tools = builder.tools.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "test_tool");
}

#[test]
fn test_with_tools() {
    let model = MockModel;
    let mut builder = ModelRequestBuilder::new(&model);
    let tools = vec![
        Tool::new("tool1", "First tool"),
        Tool::new("tool2", "Second tool"),
    ];
    builder.with_tools(tools);

    let t = builder.tools.unwrap();
    assert_eq!(t.len(), 2);
}

#[test]
fn test_to_model_request() {
    let model = MockModel;
    let mut builder = ModelRequestBuilder::new(&model);
    builder
        .with_system("System prompt".to_string())
        .with_message(Message::user("Hello".to_string()));

    let request = builder.to_model_request();

    assert_eq!(request.system, Some("System prompt".to_string()));
    assert!(request.messages.is_some());
    assert_eq!(request.messages.unwrap().len(), 1);
}

#[test]
fn test_chaining() {
    let model = MockModel;
    let mut builder = ModelRequestBuilder::new(&model);
    builder
        .with_system("System".to_string())
        .with_message(Message::user("User msg".to_string()))
        .with_tool(Tool::new("tool", "desc"))
        .with_settings(Settings {
            max_tokens: Some(50),
            timeout: None,
            temperature: None,
            thinking_budget: None,
        });

    assert!(builder.system.is_some());
    assert!(builder.messages.is_some());
    assert!(builder.tools.is_some());
    assert!(builder.settings.is_some());
}

#[test]
fn test_tool_with_parameter() {
    #[derive(JsonSchema)]
    struct TestArgs {
        query: String,
        limit: i32,
    }

    let tool = Tool::new("search", "Search for items")
        .with_parameter::<TestArgs>()
        .unwrap();

    assert!(tool.parameters.is_some());
    let params = tool.parameters.unwrap();
    assert_eq!(params._type, "object");
    assert!(params.properties.contains_key("query"));
    assert!(params.properties.contains_key("limit"));
}

#[tokio::test]
async fn test_completion() {
    let model = MockModel;
    let mut builder = ModelRequestBuilder::new(&model);
    builder.with_message(Message::user("Test".to_string()));

    let result = builder.completion().await;

    assert!(result.is_ok());
    let completion = result.unwrap();
    assert_eq!(completion.completion, "test");
    assert_eq!(completion.total_tokens, 15);
}

#[test]
fn test_message_user() {
    let msg = Message::user("Hello".to_string());
    assert_eq!(msg.content, "Hello");
    assert_eq!(msg.role, Some(Role::User));
}

#[test]
fn test_message_model() {
    let msg = Message::model("Response".to_string());
    assert_eq!(msg.content, "Response");
    assert_eq!(msg.role, Some(Role::Model));
}

#[test]
fn test_message_function_call() {
    let mut args = HashMap::new();
    args.insert("query".to_string(), Value::String("test".to_string()));

    let fc = FunctionCall {
        name: "search".to_string(),
        args,
    };

    let msg = Message::function_call(fc);
    assert_eq!(msg.role, Some(Role::Model));
    assert!(msg.content.contains("search"));
}

#[test]
fn test_message_function_result() {
    let msg = Message::function_result("search".to_string(), vec!["result1", "result2"]);
    assert_eq!(msg.role, Some(Role::Model));
    assert!(msg.content.contains("search"));
}
