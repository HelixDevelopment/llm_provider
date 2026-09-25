package mistral

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"digital.vasic.llmprovider/pkg/models"
)

func TestDefaultRetryConfig(t *testing.T) {
	config := DefaultRetryConfig()

	assert.Equal(t, 3, config.MaxRetries)
	assert.Equal(t, 1*time.Second, config.InitialDelay)
	assert.Equal(t, 30*time.Second, config.MaxDelay)
	assert.Equal(t, 2.0, config.Multiplier)
}

func TestNewMistralProvider(t *testing.T) {
	provider := NewMistralProvider("test-key", "", "")

	assert.NotNil(t, provider)
	assert.Equal(t, "test-key", provider.apiKey)
	assert.Equal(t, MistralAPIURL, provider.baseURL)
	assert.Equal(t, MistralModel, provider.model)
}

func TestNewMistralProvider_CustomValues(t *testing.T) {
	provider := NewMistralProvider("api-key", "https://custom.api.com", "custom-model")

	assert.Equal(t, "api-key", provider.apiKey)
	assert.Equal(t, "https://custom.api.com", provider.baseURL)
	assert.Equal(t, "custom-model", provider.model)
}

func TestNewMistralProviderWithRetry(t *testing.T) {
	retryConfig := RetryConfig{
		MaxRetries:   5,
		InitialDelay: 500 * time.Millisecond,
		MaxDelay:     60 * time.Second,
		Multiplier:   3.0,
	}

	provider := NewMistralProviderWithRetry("key", "", "", retryConfig)

	assert.Equal(t, 5, provider.retryConfig.MaxRetries)
	assert.Equal(t, 500*time.Millisecond, provider.retryConfig.InitialDelay)
	assert.Equal(t, 60*time.Second, provider.retryConfig.MaxDelay)
	assert.Equal(t, 3.0, provider.retryConfig.Multiplier)
}

func TestMistralProvider_ConvertRequest(t *testing.T) {
	provider := NewMistralProvider("key", "", "")

	req := &models.LLMRequest{
		ID:     "req-1",
		Prompt: "System prompt",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi"},
		},
		ModelParams: models.ModelParameters{
			Temperature: 0.7,
			MaxTokens:   1000,
			TopP:        0.9,
		},
	}

	mistralReq := provider.convertRequest(req)

	assert.Equal(t, MistralModel, mistralReq.Model)
	assert.Len(t, mistralReq.Messages, 3) // System + 2 messages
	assert.Equal(t, "system", mistralReq.Messages[0].Role)
	assert.Equal(t, "System prompt", mistralReq.Messages[0].Content)
	assert.Equal(t, 0.7, mistralReq.Temperature)
	assert.Equal(t, 1000, mistralReq.MaxTokens)
	assert.Equal(t, 0.9, mistralReq.TopP)
	assert.False(t, mistralReq.Stream)
	assert.False(t, mistralReq.SafePrompt)
}

func TestMistralProvider_ConvertRequest_WithTools(t *testing.T) {
	provider := NewMistralProvider("key", "", "")

	req := &models.LLMRequest{
		Messages: []models.Message{{Role: "user", Content: "test"}},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.ToolFunction{
					Name:        "get_weather",
					Description: "Get the current weather",
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{"type": "string"},
						},
					},
				},
			},
		},
		ToolChoice: "auto",
	}

	mistralReq := provider.convertRequest(req)

	require.Len(t, mistralReq.Tools, 1)
	assert.Equal(t, "function", mistralReq.Tools[0].Type)
	assert.Equal(t, "get_weather", mistralReq.Tools[0].Function.Name)
	assert.Equal(t, "auto", mistralReq.ToolChoice)
}

func TestMistralProvider_ConvertRequest_MaxTokensLimit(t *testing.T) {
	provider := NewMistralProvider("key", "", "")

	// Test exceeding max limit
	req := &models.LLMRequest{
		Messages: []models.Message{{Role: "user", Content: "test"}},
		ModelParams: models.ModelParameters{
			MaxTokens: 100000, // Exceeds 32768 limit
		},
	}

	mistralReq := provider.convertRequest(req)
	assert.Equal(t, 32768, mistralReq.MaxTokens)

	// Test default when 0
	req.ModelParams.MaxTokens = 0
	mistralReq = provider.convertRequest(req)
	assert.Equal(t, 4096, mistralReq.MaxTokens)
}

func TestMistralProvider_CalculateConfidence(t *testing.T) {
	provider := NewMistralProvider("key", "", "")

	tests := []struct {
		content       string
		finishReason  string
		minConfidence float64
		maxConfidence float64
	}{
		{"short", "stop", 0.85, 0.95},
		{"short", "length", 0.65, 0.75},
		{"short", "model_length", 0.60, 0.70},
		{string(make([]byte, 150)), "stop", 0.90, 1.0},
		{string(make([]byte, 600)), "stop", 0.95, 1.0},
	}

	for _, tc := range tests {
		confidence := provider.calculateConfidence(tc.content, tc.finishReason)
		assert.GreaterOrEqual(t, confidence, tc.minConfidence, "Content len=%d, finishReason=%s", len(tc.content), tc.finishReason)
		assert.LessOrEqual(t, confidence, tc.maxConfidence, "Content len=%d, finishReason=%s", len(tc.content), tc.finishReason)
	}
}

func TestMistralProvider_GetCapabilities(t *testing.T) {
	provider := NewMistralProvider("", "", "")

	caps := provider.GetCapabilities()

	assert.NotNil(t, caps)
	assert.Empty(t, caps.SupportedModels,
		"offline discovery must yield no models (CONST-036: no hardcoded fallback)")
	assert.Contains(t, caps.SupportedFeatures, "function_calling")
	assert.Contains(t, caps.SupportedFeatures, "streaming")
	assert.True(t, caps.SupportsStreaming)
	assert.True(t, caps.SupportsFunctionCalling)
	assert.True(t, caps.SupportsTools)
	assert.False(t, caps.SupportsVision)
	assert.Equal(t, 32768, caps.Limits.MaxTokens)
}

func TestMistralProvider_ValidateConfig(t *testing.T) {
	// Note: NewMistralProvider sets default values for empty baseURL and model,
	// so only the apiKey check can fail via the constructor.
	tests := []struct {
		apiKey  string
		baseURL string
		model   string
		valid   bool
		errLen  int
	}{
		{"key", "url", "model", true, 0},
		{"", "url", "model", false, 1}, // Only apiKey error (baseURL and model use defaults if empty)
		{"key", "", "model", true, 0},  // Empty baseURL gets default
		{"key", "url", "", true, 0},    // Empty model gets default
		{"", "", "", false, 1},         // Only apiKey error (others get defaults)
	}

	for _, tc := range tests {
		provider := NewMistralProvider(tc.apiKey, tc.baseURL, tc.model)
		valid, errors := provider.ValidateConfig(nil)

		assert.Equal(t, tc.valid, valid, "apiKey=%s, baseURL=%s, model=%s", tc.apiKey, tc.baseURL, tc.model)
		assert.Len(t, errors, tc.errLen, "apiKey=%s, baseURL=%s, model=%s", tc.apiKey, tc.baseURL, tc.model)
	}
}

func TestMistralProvider_NextDelay(t *testing.T) {
	provider := NewMistralProviderWithRetry("key", "", "", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     10 * time.Second,
		Multiplier:   2.0,
	})

	// Test exponential increase
	delay1 := provider.nextDelay(1 * time.Second)
	assert.Equal(t, 2*time.Second, delay1)

	delay2 := provider.nextDelay(2 * time.Second)
	assert.Equal(t, 4*time.Second, delay2)

	// Test max cap
	delay3 := provider.nextDelay(8 * time.Second)
	assert.Equal(t, 10*time.Second, delay3) // Capped at max
}

func TestIsRetryableStatus(t *testing.T) {
	tests := []struct {
		statusCode int
		retryable  bool
	}{
		{http.StatusOK, false},
		{http.StatusBadRequest, false},
		{http.StatusUnauthorized, false},
		{http.StatusForbidden, false},
		{http.StatusTooManyRequests, true},
		{http.StatusInternalServerError, true},
		{http.StatusBadGateway, true},
		{http.StatusServiceUnavailable, true},
		{http.StatusGatewayTimeout, true},
	}

	for _, tc := range tests {
		assert.Equal(t, tc.retryable, isRetryableStatus(tc.statusCode), "Status %d", tc.statusCode)
	}
}

func TestIsAuthRetryableStatus(t *testing.T) {
	assert.True(t, isAuthRetryableStatus(http.StatusUnauthorized))
	assert.False(t, isAuthRetryableStatus(http.StatusForbidden))
	assert.False(t, isAuthRetryableStatus(http.StatusOK))
}

func TestMin(t *testing.T) {
	assert.Equal(t, 1, min(1, 2))
	assert.Equal(t, 1, min(2, 1))
	assert.Equal(t, 0, min(0, 5))
	assert.Equal(t, -1, min(-1, 0))
}

func TestMistralRequest_Fields(t *testing.T) {
	req := MistralRequest{
		Model:       "mistral-large-latest",
		Messages:    []MistralMessage{{Role: "user", Content: "test"}},
		Temperature: 0.7,
		MaxTokens:   1000,
		TopP:        0.9,
		Stream:      true,
		SafePrompt:  false,
	}

	assert.Equal(t, "mistral-large-latest", req.Model)
	assert.Len(t, req.Messages, 1)
	assert.Equal(t, 0.7, req.Temperature)
	assert.True(t, req.Stream)
}

func TestMistralMessage_Fields(t *testing.T) {
	msg := MistralMessage{
		Role:    "assistant",
		Content: "Hello, how can I help?",
		ToolCalls: []MistralToolCall{
			{
				ID:   "call-1",
				Type: "function",
				Function: MistralToolCallFunction{
					Name:      "get_weather",
					Arguments: `{"location": "Paris"}`,
				},
			},
		},
	}

	assert.Equal(t, "assistant", msg.Role)
	assert.Equal(t, "Hello, how can I help?", msg.Content)
	assert.Len(t, msg.ToolCalls, 1)
	assert.Equal(t, "get_weather", msg.ToolCalls[0].Function.Name)
}

func TestMistralResponse_Fields(t *testing.T) {
	resp := MistralResponse{
		ID:      "resp-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "mistral-large-latest",
		Choices: []MistralChoice{
			{
				Index:        0,
				Message:      MistralMessage{Role: "assistant", Content: "Response"},
				FinishReason: "stop",
			},
		},
		Usage: MistralUsage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}

	assert.Equal(t, "resp-123", resp.ID)
	assert.Len(t, resp.Choices, 1)
	assert.Equal(t, 30, resp.Usage.TotalTokens)
}

func TestMistralUsage_Fields(t *testing.T) {
	usage := MistralUsage{
		PromptTokens:     100,
		CompletionTokens: 200,
		TotalTokens:      300,
	}

	assert.Equal(t, 100, usage.PromptTokens)
	assert.Equal(t, 200, usage.CompletionTokens)
	assert.Equal(t, 300, usage.TotalTokens)
}

func TestMistralTool_Fields(t *testing.T) {
	tool := MistralTool{
		Type: "function",
		Function: MistralToolFunc{
			Name:        "search",
			Description: "Search for information",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"query": map[string]interface{}{"type": "string"},
				},
			},
		},
	}

	assert.Equal(t, "function", tool.Type)
	assert.Equal(t, "search", tool.Function.Name)
}

func TestMistralProvider_Complete_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer")

		response := `{
			"id": "test-id",
			"object": "chat.completion",
			"created": 1234567890,
			"model": "mistral-large-latest",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "Test response"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
		}`

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()

	provider := NewMistralProvider("test-key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "test-id", resp.ID)
	assert.Equal(t, "Test response", resp.Content)
	assert.Equal(t, "mistral", resp.ProviderID)
	assert.Equal(t, 30, resp.TokensUsed)
}

func TestMistralProvider_Complete_WithToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := `{
			"id": "test-id",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "",
					"tool_calls": [{
						"id": "call-1",
						"type": "function",
						"function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}
					}]
				},
				"finish_reason": "tool_calls"
			}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
		}`

		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()

	provider := NewMistralProvider("key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "What's the weather in Paris?"}},
		Tools: []models.Tool{{
			Type:     "function",
			Function: models.ToolFunction{Name: "get_weather"},
		}},
	}

	resp, err := provider.Complete(context.Background(), req)

	require.NoError(t, err)
	require.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, "get_weather", resp.ToolCalls[0].Function.Name)
	assert.Equal(t, "tool_calls", resp.FinishReason)
}

func TestMistralProvider_Complete_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := `{"message": "Invalid API key", "type": "auth_error"}`
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()

	provider := NewMistralProvider("invalid-key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "Invalid API key")
}

func TestMistralProvider_Complete_EmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := `{"id": "test", "choices": [], "usage": {}}`
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()

	provider := NewMistralProvider("key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "no choices")
}

func TestMistralProvider_ConvertResponse(t *testing.T) {
	provider := NewMistralProvider("key", "", "")

	req := &models.LLMRequest{ID: "req-1"}
	mistralResp := &MistralResponse{
		ID:    "resp-1",
		Model: "mistral-large-latest",
		Choices: []MistralChoice{
			{
				Index: 0,
				Message: MistralMessage{
					Role:    "assistant",
					Content: "Test content",
					ToolCalls: []MistralToolCall{
						{
							ID:   "call-1",
							Type: "function",
							Function: MistralToolCallFunction{
								Name:      "test_func",
								Arguments: "{}",
							},
						},
					},
				},
				FinishReason: "stop",
			},
		},
		Usage: MistralUsage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}

	startTime := time.Now().Add(-100 * time.Millisecond)
	resp := provider.convertResponse(req, mistralResp, startTime)

	assert.Equal(t, "resp-1", resp.ID)
	assert.Equal(t, "req-1", resp.RequestID)
	assert.Equal(t, "mistral", resp.ProviderID)
	assert.Equal(t, "Mistral", resp.ProviderName)
	assert.Equal(t, "Test content", resp.Content)
	assert.Equal(t, 30, resp.TokensUsed)
	assert.Equal(t, "tool_calls", resp.FinishReason) // Changed due to tool calls
	assert.Len(t, resp.ToolCalls, 1)
}

func TestMistralStreamResponse_Fields(t *testing.T) {
	finishReason := "stop"
	resp := MistralStreamResponse{
		ID:      "stream-1",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "mistral-large-latest",
		Choices: []MistralStreamChoice{
			{
				Index:        0,
				Delta:        MistralMessage{Role: "assistant", Content: "chunk"},
				FinishReason: &finishReason,
			},
		},
	}

	assert.Equal(t, "stream-1", resp.ID)
	assert.Len(t, resp.Choices, 1)
	assert.Equal(t, "chunk", resp.Choices[0].Delta.Content)
	assert.NotNil(t, resp.Choices[0].FinishReason)
}

func TestMistralErrorResponse_Fields(t *testing.T) {
	code := "invalid_api_key"
	resp := MistralErrorResponse{
		Object:  "error",
		Message: "Invalid API key provided",
		Type:    "auth_error",
		Code:    &code,
	}

	assert.Equal(t, "error", resp.Object)
	assert.Equal(t, "Invalid API key provided", resp.Message)
	assert.Equal(t, "auth_error", resp.Type)
	assert.NotNil(t, resp.Code)
}

func TestMistralProvider_CompleteStream_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer")

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send streaming chunks
		_, _ = w.Write([]byte("data: {\"id\":\"chunk-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chunk-2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	provider := NewMistralProvider("test-key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-stream",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, ch)

	// Collect responses
	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	// Should have at least one response
	assert.GreaterOrEqual(t, len(responses), 1)
	// First response should be a chunk
	if len(responses) > 0 {
		assert.Equal(t, "mistral", responses[0].ProviderID)
	}
}

func TestMistralProvider_CompleteStream_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("Internal Server Error"))
	}))
	defer server.Close()

	provider := NewMistralProvider("test-key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-stream",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	ch, err := provider.CompleteStream(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, ch)
	assert.Contains(t, err.Error(), "500")
}

func TestMistralProvider_CompleteStream_WithFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send chunk with finish reason
		_, _ = w.Write([]byte("data: {\"id\":\"chunk-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":\"stop\"}]}\n\n"))
	}))
	defer server.Close()

	provider := NewMistralProvider("test-key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-stream",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	// Collect all responses
	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	assert.GreaterOrEqual(t, len(responses), 1)
}

func TestMistralProvider_Complete_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("not valid json"))
	}))
	defer server.Close()

	provider := NewMistralProvider("key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "failed to parse")
}

func TestMistralProvider_Complete_ContextCancelled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate slow response
		time.Sleep(2 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	provider := NewMistralProvider("key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	resp, err := provider.Complete(ctx, req)

	assert.Error(t, err)
	assert.Nil(t, resp)
}

func TestMistralProvider_Complete_RetryOnServerError(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}

		response := `{
			"id": "test-id",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
			"usage": {"total_tokens": 10}
		}`
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()

	provider := NewMistralProviderWithRetry("key", server.URL, "", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     100 * time.Millisecond,
		Multiplier:   2.0,
	})

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, 3, attempts)
}

func TestMistralProvider_Complete_RetryExhausted(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	provider := NewMistralProviderWithRetry("key", server.URL, "", RetryConfig{
		MaxRetries:   2,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     50 * time.Millisecond,
		Multiplier:   2.0,
	})

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Equal(t, 3, attempts) // Initial + 2 retries
}

func TestMistralProvider_Complete_RateLimited429(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}

		response := `{"id": "ok", "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}], "usage": {"total_tokens": 5}}`
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()

	provider := NewMistralProviderWithRetry("key", server.URL, "", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     50 * time.Millisecond,
		Multiplier:   2.0,
	})

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, 2, attempts)
}

func TestMistralProvider_Complete_AuthRetry401(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 2 {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"message": "Unauthorized"}`))
			return
		}

		response := `{"id": "ok", "choices": [{"message": {"content": "Authenticated"}, "finish_reason": "stop"}], "usage": {"total_tokens": 5}}`
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()

	provider := NewMistralProviderWithRetry("key", server.URL, "", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     50 * time.Millisecond,
		Multiplier:   2.0,
	})

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, 2, attempts)
}

func TestMistralProvider_HealthCheck_Success(t *testing.T) {
	var hitPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			hitPath = r.URL.Path
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"data": []}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	// HealthCheck derives the models endpoint from the configured baseURL via
	// modelsURL() = TrimSuffix(baseURL, "/chat/completions") + "/models".
	// Pointing baseURL at the test server's /v1/chat/completions makes the
	// health check hit the test server's /v1/models endpoint (CONST-051(B)
	// config-injection — NOT the hardcoded api.mistral.ai literal).
	provider := NewMistralProvider("test-key", server.URL+"/v1/chat/completions", "")
	provider.httpClient = &http.Client{Timeout: 5 * time.Second}

	err := provider.HealthCheck()

	// Real success path: a real 200 from the real test server.
	require.NoError(t, err)
	// Observable: the health check actually reached the models endpoint.
	assert.Equal(t, "/v1/models", hitPath)
}

func TestMistralProvider_HealthCheck_Error(t *testing.T) {
	// Use invalid URL to simulate connection error
	provider := NewMistralProvider("test-key", "http://invalid.local", "")
	provider.httpClient = &http.Client{Timeout: 100 * time.Millisecond}

	err := provider.HealthCheck()

	// Should return error (either connection refused or timeout)
	assert.Error(t, err)
}

func TestMistralProvider_Complete_ErrorResponseParsing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"message": "Bad request: invalid model", "type": "invalid_request_error"}`))
	}))
	defer server.Close()

	provider := NewMistralProvider("key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "Bad request: invalid model")
}

func TestMistralProvider_Complete_NonJSONErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("Internal Server Error"))
	}))
	defer server.Close()

	provider := NewMistralProvider("key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "500")
}

func TestMistralProvider_ConvertResponse_EmptyChoices(t *testing.T) {
	provider := NewMistralProvider("key", "", "")

	req := &models.LLMRequest{ID: "req-1"}
	mistralResp := &MistralResponse{
		ID:      "resp-1",
		Choices: []MistralChoice{},
		Usage:   MistralUsage{TotalTokens: 0},
	}

	startTime := time.Now()
	resp := provider.convertResponse(req, mistralResp, startTime)

	assert.Empty(t, resp.Content)
	assert.Empty(t, resp.FinishReason)
}

func TestMistralProvider_CalculateConfidence_Bounds(t *testing.T) {
	provider := NewMistralProvider("key", "", "")

	// Test that confidence stays within 0-1 bounds
	veryLongContent := string(make([]byte, 10000))
	confidence := provider.calculateConfidence(veryLongContent, "stop")
	assert.LessOrEqual(t, confidence, 1.0)
	assert.GreaterOrEqual(t, confidence, 0.0)

	// Test minimum bound
	shortContent := ""
	confidence = provider.calculateConfidence(shortContent, "model_length")
	assert.LessOrEqual(t, confidence, 1.0)
	assert.GreaterOrEqual(t, confidence, 0.0)
}

func TestMistralProvider_WaitWithJitter(t *testing.T) {
	provider := NewMistralProvider("key", "", "")

	ctx := context.Background()
	baseDelay := 50 * time.Millisecond
	start := time.Now()
	provider.waitWithJitter(ctx, baseDelay)
	elapsed := time.Since(start)

	// Should wait at least the delay time. Load-robust: extra CPU pressure
	// can only delay a timer, never fire it early.
	assert.GreaterOrEqual(t, elapsed, baseDelay)

	// Upper bound: deliberately NOT asserted in wall-clock time. Elapsed time
	// also contains scheduler latency this package does not control, so a
	// wall-clock ceiling measures the host at least as much as it measures
	// waitWithJitter. Reproduced under deliberate CPU load as
	// `"116.893467ms" is not less than "100ms"`.
	// The property that ceiling was proxying -- the timer is armed for the
	// base delay plus AT MOST 10% jitter -- is asserted directly below, on
	// every value the production path actually computes.
	minD, maxD := jitteredDelay(baseDelay), jitteredDelay(baseDelay)
	for i := 0; i < 100000; i++ {
		d := jitteredDelay(baseDelay)
		if d < minD {
			minD = d
		}
		if d > maxD {
			maxD = d
		}
	}
	assert.GreaterOrEqual(t, minD, baseDelay,
		"jittered delay must never be shorter than the base delay")
	assert.Less(t, maxD, baseDelay+baseDelay/10,
		"jitter must never exceed 10% of the base delay")
}

func TestMistralProvider_WaitWithJitter_ContextCancelled(t *testing.T) {
	provider := NewMistralProvider("key", "", "")

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	start := time.Now()
	provider.waitWithJitter(ctx, 1*time.Second)
	elapsed := time.Since(start)

	// Should return immediately due to cancelled context
	assert.Less(t, elapsed, 100*time.Millisecond)
}

func TestMistralProvider_CompleteStream_MalformedJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send valid chunk, then malformed, then valid
		_, _ = w.Write([]byte("data: {\"id\":\"chunk-1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: not valid json\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chunk-2\",\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	provider := NewMistralProvider("test-key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-stream",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	// Should still receive valid chunks despite malformed JSON
	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	assert.GreaterOrEqual(t, len(responses), 1)
}

func TestMistralProvider_CompleteStream_EmptyLines(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send with empty lines and lines without data prefix
		_, _ = w.Write([]byte("\n\n"))
		_, _ = w.Write([]byte("comment: this is ignored\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chunk-1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"))
		_, _ = w.Write([]byte("\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	provider := NewMistralProvider("test-key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-stream",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	assert.GreaterOrEqual(t, len(responses), 1)
}

func TestRetryConfig_Fields(t *testing.T) {
	config := RetryConfig{
		MaxRetries:   5,
		InitialDelay: 2 * time.Second,
		MaxDelay:     1 * time.Minute,
		Multiplier:   3.0,
	}

	assert.Equal(t, 5, config.MaxRetries)
	assert.Equal(t, 2*time.Second, config.InitialDelay)
	assert.Equal(t, 1*time.Minute, config.MaxDelay)
	assert.Equal(t, 3.0, config.Multiplier)
}

func TestMistralProvider_Complete_GeneratesRequestID(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := `{"id": "test", "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}], "usage": {"total_tokens": 5}}`
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()

	provider := NewMistralProvider("key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "", // Empty ID - should generate one
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	require.NoError(t, err)
	assert.NotNil(t, resp)
}

func TestMistralProvider_ConvertRequest_NoSystemPrompt(t *testing.T) {
	provider := NewMistralProvider("key", "", "")

	req := &models.LLMRequest{
		ID:     "req-1",
		Prompt: "", // No system prompt
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	mistralReq := provider.convertRequest(req)

	// Should only have user message, no system message
	assert.Len(t, mistralReq.Messages, 1)
	assert.Equal(t, "user", mistralReq.Messages[0].Role)
}

// Benchmarks

func BenchmarkMistralProvider_ConvertRequest(b *testing.B) {
	provider := NewMistralProvider("test-key", "https://api.mistral.ai/v1/chat/completions", "mistral-large-latest")
	req := &models.LLMRequest{
		ID:     "bench-request",
		Prompt: "You are a helpful assistant.",
		Messages: []models.Message{
			{Role: "user", Content: "Hello, how are you?"},
		},
		ModelParams: models.ModelParameters{
			MaxTokens:   100,
			Temperature: 0.7,
			TopP:        0.9,
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.convertRequest(req)
	}
}

func BenchmarkMistralProvider_CalculateConfidence(b *testing.B) {
	provider := NewMistralProvider("test-key", "", "mistral-large-latest")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.calculateConfidence("This is a detailed response with sufficient content to test confidence calculation.", "stop")
	}
}

func BenchmarkMistralProvider_ConvertResponse(b *testing.B) {
	provider := NewMistralProvider("test-key", "", "mistral-large-latest")
	req := &models.LLMRequest{
		ID: "bench-request",
	}
	resp := &MistralResponse{
		ID:    "resp-1",
		Model: "mistral-large-latest",
		Choices: []MistralChoice{
			{
				Index:        0,
				Message:      MistralMessage{Role: "assistant", Content: "This is a benchmark response with enough content."},
				FinishReason: "stop",
			},
		},
		Usage: MistralUsage{
			PromptTokens:     50,
			CompletionTokens: 30,
			TotalTokens:      80,
		},
	}
	startTime := time.Now()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.convertResponse(req, resp, startTime)
	}
}
