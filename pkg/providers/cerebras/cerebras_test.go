package cerebras

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

func TestNewCerebrasProvider(t *testing.T) {
	provider := NewCerebrasProvider("test-key", "", "")

	assert.NotNil(t, provider)
	assert.Equal(t, "test-key", provider.apiKey)
	assert.Equal(t, CerebrasAPIURL, provider.baseURL)
	assert.Equal(t, CerebrasModel, provider.model)
}

func TestNewCerebrasProvider_CustomValues(t *testing.T) {
	provider := NewCerebrasProvider("api-key", "https://custom.api.com", "custom-model")

	assert.Equal(t, "api-key", provider.apiKey)
	assert.Equal(t, "https://custom.api.com", provider.baseURL)
	assert.Equal(t, "custom-model", provider.model)
}

func TestNewCerebrasProviderWithRetry(t *testing.T) {
	retryConfig := RetryConfig{
		MaxRetries:   5,
		InitialDelay: 500 * time.Millisecond,
		MaxDelay:     60 * time.Second,
		Multiplier:   3.0,
	}

	provider := NewCerebrasProviderWithRetry("key", "", "", retryConfig)

	assert.Equal(t, 5, provider.retryConfig.MaxRetries)
	assert.Equal(t, 500*time.Millisecond, provider.retryConfig.InitialDelay)
	assert.Equal(t, 60*time.Second, provider.retryConfig.MaxDelay)
	assert.Equal(t, 3.0, provider.retryConfig.Multiplier)
}

func TestCerebrasProvider_ConvertRequest(t *testing.T) {
	provider := NewCerebrasProvider("key", "", "")

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

	cerebrasReq := provider.convertRequest(req)

	assert.Equal(t, CerebrasModel, cerebrasReq.Model)
	assert.Len(t, cerebrasReq.Messages, 3) // System + 2 messages
	assert.Equal(t, "system", cerebrasReq.Messages[0].Role)
	assert.Equal(t, "System prompt", cerebrasReq.Messages[0].Content)
	assert.Equal(t, 0.7, cerebrasReq.Temperature)
	assert.Equal(t, 1000, cerebrasReq.MaxTokens)
	assert.Equal(t, 0.9, cerebrasReq.TopP)
	assert.False(t, cerebrasReq.Stream)
}

func TestCerebrasProvider_ConvertRequest_MaxTokensLimit(t *testing.T) {
	provider := NewCerebrasProvider("key", "", "")

	// Test exceeding max limit
	req := &models.LLMRequest{
		Messages: []models.Message{{Role: "user", Content: "test"}},
		ModelParams: models.ModelParameters{
			MaxTokens: 100000, // Exceeds 8192 limit
		},
	}

	cerebrasReq := provider.convertRequest(req)
	assert.Equal(t, 8192, cerebrasReq.MaxTokens)

	// Test default when 0
	req.ModelParams.MaxTokens = 0
	cerebrasReq = provider.convertRequest(req)
	assert.Equal(t, 4096, cerebrasReq.MaxTokens)
}

func TestCerebrasProvider_CalculateConfidence(t *testing.T) {
	provider := NewCerebrasProvider("key", "", "")

	tests := []struct {
		content       string
		finishReason  string
		minConfidence float64
		maxConfidence float64
	}{
		{"short", "stop", 0.85, 0.95},
		{"short", "length", 0.65, 0.75},
		{string(make([]byte, 150)), "stop", 0.90, 1.0},
		{string(make([]byte, 600)), "stop", 0.95, 1.0},
		{"", "stop", 0.85, 0.95},
	}

	for _, tc := range tests {
		confidence := provider.calculateConfidence(tc.content, tc.finishReason)
		assert.GreaterOrEqual(t, confidence, tc.minConfidence, "Content len=%d, finishReason=%s", len(tc.content), tc.finishReason)
		assert.LessOrEqual(t, confidence, tc.maxConfidence, "Content len=%d, finishReason=%s", len(tc.content), tc.finishReason)
	}
}

func TestCerebrasProvider_GetCapabilities(t *testing.T) {
	provider := NewCerebrasProvider("", "", "")

	caps := provider.GetCapabilities()

	assert.NotNil(t, caps)
	assert.Empty(t, caps.SupportedModels,
		"offline discovery must yield no models (CONST-036: no hardcoded fallback)")
	assert.Contains(t, caps.SupportedFeatures, "streaming")
	assert.True(t, caps.SupportsStreaming)
	assert.False(t, caps.SupportsFunctionCalling)
	assert.False(t, caps.SupportsVision)
	assert.Equal(t, 8192, caps.Limits.MaxTokens)
}

func TestCerebrasProvider_ValidateConfig(t *testing.T) {
	// Note: NewCerebrasProvider sets default values for empty baseURL and model,
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
		provider := NewCerebrasProvider(tc.apiKey, tc.baseURL, tc.model)
		valid, errors := provider.ValidateConfig(nil)

		assert.Equal(t, tc.valid, valid, "apiKey=%s, baseURL=%s, model=%s", tc.apiKey, tc.baseURL, tc.model)
		assert.Len(t, errors, tc.errLen, "apiKey=%s, baseURL=%s, model=%s", tc.apiKey, tc.baseURL, tc.model)
	}
}

func TestCerebrasProvider_NextDelay(t *testing.T) {
	provider := NewCerebrasProviderWithRetry("key", "", "", RetryConfig{
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

func TestCerebrasRequest_Fields(t *testing.T) {
	req := CerebrasRequest{
		Model:       "llama-3.3-70b",
		Messages:    []CerebrasMessage{{Role: "user", Content: "test"}},
		Temperature: 0.7,
		MaxTokens:   1000,
		TopP:        0.9,
		Stream:      true,
	}

	assert.Equal(t, "llama-3.3-70b", req.Model)
	assert.Len(t, req.Messages, 1)
	assert.Equal(t, 0.7, req.Temperature)
	assert.Equal(t, 1000, req.MaxTokens)
	assert.True(t, req.Stream)
}

func TestCerebrasMessage_Fields(t *testing.T) {
	msg := CerebrasMessage{
		Role:    "assistant",
		Content: "Hello, how can I help?",
	}

	assert.Equal(t, "assistant", msg.Role)
	assert.Equal(t, "Hello, how can I help?", msg.Content)
}

func TestCerebrasResponse_Fields(t *testing.T) {
	resp := CerebrasResponse{
		ID:      "resp-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "llama-3.3-70b",
		Choices: []CerebrasChoice{
			{
				Index:        0,
				Message:      CerebrasMessage{Role: "assistant", Content: "Response"},
				FinishReason: "stop",
			},
		},
		Usage: CerebrasUsage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}

	assert.Equal(t, "resp-123", resp.ID)
	assert.Len(t, resp.Choices, 1)
	assert.Equal(t, 30, resp.Usage.TotalTokens)
}

func TestCerebrasUsage_Fields(t *testing.T) {
	usage := CerebrasUsage{
		PromptTokens:     100,
		CompletionTokens: 200,
		TotalTokens:      300,
	}

	assert.Equal(t, 100, usage.PromptTokens)
	assert.Equal(t, 200, usage.CompletionTokens)
	assert.Equal(t, 300, usage.TotalTokens)
}

func TestCerebrasProvider_Complete_Success(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer")

		response := `{
			"id": "test-id",
			"object": "chat.completion",
			"created": 1234567890,
			"model": "llama-3.3-70b",
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

	provider := NewCerebrasProvider("test-key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "test-id", resp.ID)
	assert.Equal(t, "Test response", resp.Content)
	assert.Equal(t, "cerebras", resp.ProviderID)
	assert.Equal(t, 30, resp.TokensUsed)
}

func TestCerebrasProvider_Complete_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := `{"error": {"message": "Invalid API key", "type": "auth_error", "code": "401"}}`
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()

	provider := NewCerebrasProvider("invalid-key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "Invalid API key")
}

func TestCerebrasProvider_Complete_EmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := `{"id": "test", "choices": [], "usage": {}}`
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()

	provider := NewCerebrasProvider("key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "no choices")
}

func TestCerebrasProvider_ConvertResponse(t *testing.T) {
	provider := NewCerebrasProvider("key", "", "")

	req := &models.LLMRequest{ID: "req-1"}
	cerebrasResp := &CerebrasResponse{
		ID:    "resp-1",
		Model: "llama-3.3-70b",
		Choices: []CerebrasChoice{
			{
				Index:        0,
				Message:      CerebrasMessage{Role: "assistant", Content: "Test content"},
				FinishReason: "stop",
			},
		},
		Usage: CerebrasUsage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}

	startTime := time.Now().Add(-100 * time.Millisecond)
	resp := provider.convertResponse(req, cerebrasResp, startTime)

	assert.Equal(t, "resp-1", resp.ID)
	assert.Equal(t, "req-1", resp.RequestID)
	assert.Equal(t, "cerebras", resp.ProviderID)
	assert.Equal(t, "Cerebras", resp.ProviderName)
	assert.Equal(t, "Test content", resp.Content)
	assert.Equal(t, 30, resp.TokensUsed)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.GreaterOrEqual(t, resp.ResponseTime, int64(100))
}

func TestCerebrasStreamResponse_Fields(t *testing.T) {
	finishReason := "stop"
	resp := CerebrasStreamResponse{
		ID:      "stream-1",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "llama-3.3-70b",
		Choices: []CerebrasStreamChoice{
			{
				Index:        0,
				Delta:        CerebrasMessage{Role: "assistant", Content: "chunk"},
				FinishReason: &finishReason,
			},
		},
	}

	assert.Equal(t, "stream-1", resp.ID)
	assert.Len(t, resp.Choices, 1)
	assert.Equal(t, "chunk", resp.Choices[0].Delta.Content)
	assert.NotNil(t, resp.Choices[0].FinishReason)
}

func TestCerebrasErrorResponse_Fields(t *testing.T) {
	errResp := CerebrasErrorResponse{}
	errResp.Error.Message = "Invalid key"
	errResp.Error.Type = "auth_error"
	errResp.Error.Code = "401"

	assert.Equal(t, "Invalid key", errResp.Error.Message)
	assert.Equal(t, "auth_error", errResp.Error.Type)
	assert.Equal(t, "401", errResp.Error.Code)
}

func TestCerebrasProvider_CompleteStream_Success(t *testing.T) {
	// Create mock streaming server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send streaming chunks
		_, _ = w.Write([]byte("data: {\"id\":\"chunk-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"chunk-2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	provider := NewCerebrasProvider("test-key", server.URL, "")

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

	// Should have received chunks + final response
	assert.GreaterOrEqual(t, len(responses), 1)
}

func TestCerebrasProvider_CompleteStream_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("Server error"))
	}))
	defer server.Close()

	provider := NewCerebrasProvider("test-key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-error",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	ch, err := provider.CompleteStream(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, ch)
	assert.Contains(t, err.Error(), "500")
}

func TestCerebrasProvider_CompleteStream_WithFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send chunk with finish_reason
		_, _ = w.Write([]byte("data: {\"id\":\"chunk-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Response\"},\"finish_reason\":\"stop\"}]}\n\n"))
	}))
	defer server.Close()

	provider := NewCerebrasProvider("test-key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-finish",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	assert.NotEmpty(t, responses)
}

func TestCerebrasProvider_HealthCheck_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"models": []}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	// Create provider with custom HTTP client that redirects to mock server
	provider := NewCerebrasProvider("test-key", server.URL, "")

	// The health check goes to a hardcoded URL, so we test the error path
	// for the actual implementation
	err := provider.HealthCheck()

	// Will fail because it tries to reach real Cerebras API
	// This tests that the function executes without panicking
	assert.Error(t, err)
}

func TestCerebrasProvider_Complete_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("not valid json"))
	}))
	defer server.Close()

	provider := NewCerebrasProvider("key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "parse")
}

func TestCerebrasProvider_Complete_NonJSONError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("plain text error"))
	}))
	defer server.Close()

	provider := NewCerebrasProvider("key", server.URL, "")

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "plain text error")
}

func TestCerebrasProvider_Complete_ContextCancelled(t *testing.T) {
	// Create a server that blocks
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
	}))
	defer server.Close()

	provider := NewCerebrasProviderWithRetry("key", server.URL, "", RetryConfig{
		MaxRetries:   0,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     100 * time.Millisecond,
		Multiplier:   2.0,
	})

	ctx, cancel := context.WithCancel(context.Background())

	// Cancel context immediately
	cancel()

	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := provider.Complete(ctx, req)

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "context")
}

func TestCerebrasProvider_Complete_RetryOnServerError(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte("Service unavailable"))
			return
		}
		// Success on 3rd attempt
		response := `{"id":"test","choices":[{"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"total_tokens":10}}`
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()

	provider := NewCerebrasProviderWithRetry("key", server.URL, "", RetryConfig{
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

func TestCerebrasProvider_Complete_RetryExhausted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte("Rate limited"))
	}))
	defer server.Close()

	provider := NewCerebrasProviderWithRetry("key", server.URL, "", RetryConfig{
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
}

func TestCerebrasProvider_Complete_AuthRetry(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			// First attempt returns 401
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"message":"Unauthorized"}}`))
			return
		}
		// Second attempt succeeds
		response := `{"id":"test","choices":[{"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"total_tokens":10}}`
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(response))
	}))
	defer server.Close()

	provider := NewCerebrasProviderWithRetry("key", server.URL, "", RetryConfig{
		MaxRetries:   1,
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

func TestCerebrasProvider_WaitWithJitter(t *testing.T) {
	provider := NewCerebrasProvider("key", "", "")
	ctx := context.Background()

	baseDelay := 10 * time.Millisecond

	// Test that wait completes. The lower bound is load-robust: extra CPU
	// pressure can only delay a timer, never fire it early.
	start := time.Now()
	provider.waitWithJitter(ctx, baseDelay)
	elapsed := time.Since(start)

	assert.GreaterOrEqual(t, elapsed, baseDelay)

	// Upper bound: deliberately NOT asserted in wall-clock time. Elapsed time
	// also contains scheduler latency this package does not control, so a
	// wall-clock ceiling measures the host at least as much as it measures
	// waitWithJitter. Reproduced under deliberate CPU load as
	// `"49.495637ms" is not less than or equal to "20ms"`.
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

func TestCerebrasProvider_WaitWithJitter_ContextCancelled(t *testing.T) {
	provider := NewCerebrasProvider("key", "", "")
	ctx, cancel := context.WithCancel(context.Background())

	// Cancel immediately
	cancel()

	start := time.Now()
	provider.waitWithJitter(ctx, 1*time.Second)
	elapsed := time.Since(start)

	// Should return immediately due to cancelled context
	assert.LessOrEqual(t, elapsed, 50*time.Millisecond)
}

func TestCerebrasProvider_ConvertRequest_NoPrompt(t *testing.T) {
	provider := NewCerebrasProvider("key", "", "")

	req := &models.LLMRequest{
		ID:     "req-1",
		Prompt: "", // Empty prompt
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
		ModelParams: models.ModelParameters{
			Temperature: 0.5,
			MaxTokens:   500,
		},
	}

	cerebrasReq := provider.convertRequest(req)

	// Should only have 1 message (no system prompt added)
	assert.Len(t, cerebrasReq.Messages, 1)
	assert.Equal(t, "user", cerebrasReq.Messages[0].Role)
}

func TestCerebrasProvider_ConvertResponse_EmptyChoices(t *testing.T) {
	provider := NewCerebrasProvider("key", "", "")

	req := &models.LLMRequest{ID: "req-1"}
	cerebrasResp := &CerebrasResponse{
		ID:      "resp-1",
		Choices: []CerebrasChoice{}, // Empty choices
		Usage:   CerebrasUsage{TotalTokens: 10},
	}

	startTime := time.Now()
	resp := provider.convertResponse(req, cerebrasResp, startTime)

	assert.Equal(t, "", resp.Content)
	assert.Equal(t, "", resp.FinishReason)
}

func TestCerebrasProvider_CalculateConfidence_EdgeCases(t *testing.T) {
	provider := NewCerebrasProvider("key", "", "")

	// Test unknown finish reason
	confidence := provider.calculateConfidence("test", "unknown_reason")
	assert.Equal(t, 0.8, confidence) // Base confidence

	// Test negative confidence floor (shouldn't happen but verify it's bounded)
	confidence = provider.calculateConfidence("", "length")
	assert.GreaterOrEqual(t, confidence, 0.0)

	// Test long content with length finish reason
	longContent := string(make([]byte, 1000))
	confidence = provider.calculateConfidence(longContent, "length")
	assert.LessOrEqual(t, confidence, 1.0)
}

func TestCerebrasChoice_Fields(t *testing.T) {
	choice := CerebrasChoice{
		Index: 0,
		Message: CerebrasMessage{
			Role:    "assistant",
			Content: "Test content",
		},
		FinishReason: "stop",
	}

	assert.Equal(t, 0, choice.Index)
	assert.Equal(t, "assistant", choice.Message.Role)
	assert.Equal(t, "Test content", choice.Message.Content)
	assert.Equal(t, "stop", choice.FinishReason)
}

func TestCerebrasStreamChoice_Fields(t *testing.T) {
	finishReason := "length"
	choice := CerebrasStreamChoice{
		Index: 1,
		Delta: CerebrasMessage{
			Role:    "assistant",
			Content: "Delta content",
		},
		FinishReason: &finishReason,
	}

	assert.Equal(t, 1, choice.Index)
	assert.Equal(t, "Delta content", choice.Delta.Content)
	assert.NotNil(t, choice.FinishReason)
	assert.Equal(t, "length", *choice.FinishReason)
}

func TestCerebrasStreamChoice_NilFinishReason(t *testing.T) {
	choice := CerebrasStreamChoice{
		Index: 0,
		Delta: CerebrasMessage{Content: "test"},
	}

	assert.Nil(t, choice.FinishReason)
}

// Benchmarks

func BenchmarkCerebrasProvider_ConvertRequest(b *testing.B) {
	provider := NewCerebrasProvider("test-key", "https://api.cerebras.ai/v1/chat/completions", "llama3.1-70b")
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

func BenchmarkCerebrasProvider_CalculateConfidence(b *testing.B) {
	provider := NewCerebrasProvider("test-key", "", "llama3.1-70b")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.calculateConfidence("This is a detailed response with sufficient content to test confidence calculation.", "stop")
	}
}

func BenchmarkCerebrasProvider_ConvertResponse(b *testing.B) {
	provider := NewCerebrasProvider("test-key", "", "llama3.1-70b")
	req := &models.LLMRequest{
		ID: "bench-request",
	}
	resp := &CerebrasResponse{
		ID:    "resp-1",
		Model: "llama3.1-70b",
		Choices: []CerebrasChoice{
			{
				Index:        0,
				Message:      CerebrasMessage{Role: "assistant", Content: "This is a benchmark response with enough content."},
				FinishReason: "stop",
			},
		},
		Usage: CerebrasUsage{
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

// TestCompleteStream_EOFLastChunkNoNewline reproduces the EOF-last-chunk
// data-loss bug: when the final SSE data chunk arrives WITHOUT a trailing
// newline (server closes the connection right after writing it),
// bufio.Reader.ReadBytes('\\n') returns the partial line ALONGSIDE
// io.EOF. The buggy loop breaks on io.EOF before processing that line,
// silently dropping the last chunk content. RED: aggregated content lacks "LAST".
func TestCompleteStream_EOFLastChunkNoNewline(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"id\":\"s1\",\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"s2\",\"choices\":[{\"delta\":{\"content\":\"LAST\"}}]}"))
	}))
	defer server.Close()

	provider := NewCerebrasProvider("test-key", server.URL, "")
	req := &models.LLMRequest{ID: "stream-eof", Messages: []models.Message{{Role: "user", Content: "Hi"}}}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var agg string
	for resp := range ch {
		agg += resp.Content
	}
	assert.Contains(t, agg, "LAST",
		"final SSE chunk lacking a trailing newline must not be dropped on EOF")
	assert.Contains(t, agg, "Hello ")
}
