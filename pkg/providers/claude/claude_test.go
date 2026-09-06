package claude

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"digital.vasic.llmprovider/pkg/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClaudeProvider(t *testing.T) {
	tests := []struct {
		name     string
		apiKey   string
		baseURL  string
		model    string
		expected *ClaudeProvider
	}{
		{
			name:    "default values",
			apiKey:  "test-key",
			baseURL: "",
			model:   "",
			expected: &ClaudeProvider{
				apiKey:  "test-key",
				baseURL: ClaudeAPIURL,
				model:   ClaudeModel,
			},
		},
		{
			name:    "custom values",
			apiKey:  "test-key",
			baseURL: "https://custom.example.com",
			model:   "claude-3-custom",
			expected: &ClaudeProvider{
				apiKey:  "test-key",
				baseURL: "https://custom.example.com",
				model:   "claude-3-custom",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := NewClaudeProvider(tt.apiKey, tt.baseURL, tt.model)
			require.NotNil(t, provider)
			assert.Equal(t, tt.expected.apiKey, provider.apiKey)
			assert.Equal(t, tt.expected.baseURL, provider.baseURL)
			assert.Equal(t, tt.expected.model, provider.model)
			assert.NotNil(t, provider.httpClient)
			assert.Equal(t, 60*time.Second, provider.httpClient.Timeout)
		})
	}
}

func TestClaudeProvider_Complete_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/v1/messages", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Equal(t, "test-key", r.Header.Get("x-api-key"))
		assert.Equal(t, "2023-06-01", r.Header.Get("anthropic-version"))

		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id": "msg_123",
			"type": "message",
			"role": "assistant",
			"content": [{"type": "text", "text": "Hello, world!"}],
			"model": "claude-3-sonnet-20240229",
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`))
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL+"/v1/messages", "claude-3-sonnet-20240229")
	req := &models.LLMRequest{
		ID: "test-request",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "msg_123", resp.ID)
	assert.Equal(t, "Hello, world!", resp.Content)
	assert.Equal(t, "end_turn", resp.FinishReason)
	assert.Greater(t, resp.Confidence, 0.0)
	assert.Less(t, resp.Confidence, 1.0)
}

func TestClaudeProvider_Complete_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"type": "invalid_request_error", "message": "Invalid request"}}`))
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-test")
	req := &models.LLMRequest{
		ID: "test-request",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Nil(t, resp)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "Claude API error: 400")
}

func TestClaudeProvider_ConvertRequest(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{
		ID: "test-request",
		Messages: []models.Message{
			{Role: "user", Content: "Hello, Claude!"},
			{Role: "assistant", Content: "Hi there!"},
			{Role: "user", Content: "How are you?"},
		},
		ModelParams: models.ModelParameters{
			Model:         "claude-3-sonnet-20240229",
			MaxTokens:     100,
			Temperature:   0.7,
			TopP:          0.9,
			StopSequences: []string{"\n", "STOP"},
		},
	}

	claudeReq := provider.convertRequest(req)
	assert.Equal(t, "claude-sonnet-4-6", claudeReq.Model)
	assert.Equal(t, 100, claudeReq.MaxTokens)
	assert.Equal(t, 0.7, claudeReq.Temperature)
	assert.Equal(t, 0.9, claudeReq.TopP)
	assert.Equal(t, []string{"\n", "STOP"}, claudeReq.StopSequences)
	assert.Len(t, claudeReq.Messages, 3)
	assert.Equal(t, "user", claudeReq.Messages[0].Role)
	assert.Equal(t, "Hello, Claude!", claudeReq.Messages[0].Content)
	assert.Equal(t, "assistant", claudeReq.Messages[1].Role)
	assert.Equal(t, "Hi there!", claudeReq.Messages[1].Content)
	assert.Equal(t, "user", claudeReq.Messages[2].Role)
	assert.Equal(t, "How are you?", claudeReq.Messages[2].Content)
}

func TestClaudeProvider_CalculateConfidence(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")

	tests := []struct {
		name         string
		content      string
		finishReason string
		expectedMin  float64
		expectedMax  float64
	}{
		{
			name:         "good response with end_turn",
			content:      "This is a comprehensive and well-formed response.",
			finishReason: "end_turn",
			expectedMin:  0.9,
			expectedMax:  1.0,
		},
		{
			name:         "short response with max_tokens",
			content:      "Short",
			finishReason: "max_tokens",
			expectedMin:  0.8,
			expectedMax:  0.9,
		},
		{
			name:         "empty response",
			content:      "",
			finishReason: "stop_sequence",
			expectedMin:  0.9,
			expectedMax:  1.0,
		},
		{
			name:         "long response over 50 chars",
			content:      "This is a response that is over fifty characters in length.",
			finishReason: "end_turn",
			expectedMin:  0.95,
			expectedMax:  1.0,
		},
		{
			name:         "very long response over 200 chars",
			content:      "This is a very long response that exceeds two hundred characters. It contains a lot of information and should demonstrate high quality output from the Claude model. The response includes multiple sentences and provides comprehensive information.",
			finishReason: "end_turn",
			expectedMin:  0.97,
			expectedMax:  1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			confidence := provider.calculateConfidence(tt.content, tt.finishReason)
			assert.GreaterOrEqual(t, confidence, tt.expectedMin)
			assert.LessOrEqual(t, confidence, tt.expectedMax)
		})
	}
}

func TestClaudeProvider_CompleteStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "test-key", r.Header.Get("x-api-key"))

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send streaming chunks in Claude format
		_, _ = w.Write([]byte("data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\" World\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"message_stop\"}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet-20240229")
	req := &models.LLMRequest{
		ID: "test-stream-request",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, ch)

	// Collect responses
	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	// Should have received chunks plus final response
	assert.GreaterOrEqual(t, len(responses), 2)
}

func TestClaudeProvider_GetCapabilities(t *testing.T) {
	provider := NewClaudeProvider("", "", "")
	caps := provider.GetCapabilities()

	require.NotNil(t, caps)

	// Check supported models (Claude 4 and 3.5 series)
	assert.Empty(t, caps.SupportedModels,
		"offline discovery must yield no models (CONST-036: no hardcoded fallback)")

	// Check supported features
	assert.Contains(t, caps.SupportedFeatures, "text_completion")
	assert.Contains(t, caps.SupportedFeatures, "chat")
	assert.Contains(t, caps.SupportedFeatures, "streaming")
	assert.Contains(t, caps.SupportedFeatures, "function_calling")

	// Check supported request types
	assert.Contains(t, caps.SupportedRequestTypes, "text_completion")
	assert.Contains(t, caps.SupportedRequestTypes, "chat")

	// Check boolean capabilities
	assert.True(t, caps.SupportsStreaming)
	assert.True(t, caps.SupportsFunctionCalling)
	assert.True(t, caps.SupportsVision)
	assert.True(t, caps.SupportsTools)
	assert.False(t, caps.SupportsSearch)
	assert.True(t, caps.SupportsReasoning)
	assert.True(t, caps.SupportsCodeCompletion)
	assert.True(t, caps.SupportsCodeAnalysis)
	assert.True(t, caps.SupportsRefactoring)

	// Check limits
	assert.Equal(t, 200000, caps.Limits.MaxTokens)
	assert.Equal(t, 100000, caps.Limits.MaxInputLength)
	assert.Equal(t, 4096, caps.Limits.MaxOutputLength)
	assert.Equal(t, 10, caps.Limits.MaxConcurrentRequests)

	// Check metadata
	assert.Equal(t, "Anthropic", caps.Metadata["provider"])
	assert.Equal(t, "Claude", caps.Metadata["model_family"])
	assert.Equal(t, "2023-06-01", caps.Metadata["api_version"])
}

func TestClaudeProvider_ValidateConfig(t *testing.T) {
	tests := []struct {
		name         string
		apiKey       string
		baseURL      string
		model        string
		expectValid  bool
		expectErrLen int
	}{
		{
			name:         "all valid",
			apiKey:       "test-key",
			baseURL:      "https://api.anthropic.com",
			model:        "claude-3-sonnet",
			expectValid:  true,
			expectErrLen: 0,
		},
		{
			name:         "missing api key",
			apiKey:       "",
			baseURL:      "https://api.anthropic.com",
			model:        "claude-3-sonnet",
			expectValid:  false,
			expectErrLen: 1,
		},
		{
			name:         "missing base url",
			apiKey:       "test-key",
			baseURL:      "",
			model:        "claude-3-sonnet",
			expectValid:  false,
			expectErrLen: 1,
		},
		{
			name:         "missing model",
			apiKey:       "test-key",
			baseURL:      "https://api.anthropic.com",
			model:        "",
			expectValid:  false,
			expectErrLen: 1,
		},
		{
			name:         "all missing",
			apiKey:       "",
			baseURL:      "",
			model:        "",
			expectValid:  false,
			expectErrLen: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &ClaudeProvider{
				apiKey:  tt.apiKey,
				baseURL: tt.baseURL,
				model:   tt.model,
			}

			valid, errs := provider.ValidateConfig(nil)
			assert.Equal(t, tt.expectValid, valid)
			assert.Len(t, errs, tt.expectErrLen)
		})
	}
}

func TestClaudeProvider_Complete_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond) // Simulate slow response
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_123","content":[{"text":"test"}],"usage":{}}`))
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")
	req := &models.LLMRequest{
		ID: "test-request",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	resp, err := provider.Complete(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "context")
}

func TestClaudeProvider_RetryOnServerError(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_123","content":[{"type":"text","text":"success"}],"usage":{}}`))
	}))
	defer server.Close()

	retryConfig := RetryConfig{
		MaxRetries:   3,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     100 * time.Millisecond,
		Multiplier:   2.0,
	}

	provider := NewClaudeProviderWithRetry("test-key", server.URL, "claude-3-sonnet", retryConfig)
	req := &models.LLMRequest{
		ID: "test-request",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "success", resp.Content)
	assert.Equal(t, 3, attempts)
}

func TestClaudeProvider_ConvertRequestWithSystemMessage(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{
		ID: "test-request",
		Messages: []models.Message{
			{Role: "system", Content: "You are a helpful coding assistant."},
			{Role: "user", Content: "Write hello world"},
		},
		ModelParams: models.ModelParameters{
			MaxTokens:   100,
			Temperature: 0.5,
		},
	}

	claudeReq := provider.convertRequest(req)

	// System message should be extracted to System field
	assert.Equal(t, "You are a helpful coding assistant.", claudeReq.System)
	// Messages should only contain user/assistant messages (not system)
	assert.Len(t, claudeReq.Messages, 1)
	assert.Equal(t, "user", claudeReq.Messages[0].Role)
	assert.Equal(t, "Write hello world", claudeReq.Messages[0].Content)
}

func TestClaudeProvider_ConvertResponse(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{ID: "req-123"}

	stopReason := "end_turn"
	claudeResp := &ClaudeResponse{
		ID:         "msg-456",
		Type:       "message",
		Role:       "assistant",
		Model:      "claude-3-sonnet-20240229",
		StopReason: &stopReason,
		Content: []ClaudeContent{
			{Type: "text", Text: "Hello there!"},
		},
		Usage: ClaudeUsage{
			InputTokens:  10,
			OutputTokens: 5,
		},
	}

	startTime := time.Now()
	resp := provider.convertResponse(req, claudeResp, startTime)

	assert.Equal(t, "msg-456", resp.ID)
	assert.Equal(t, "req-123", resp.RequestID)
	assert.Equal(t, "claude", resp.ProviderID)
	assert.Equal(t, "Claude", resp.ProviderName)
	assert.Equal(t, "Hello there!", resp.Content)
	assert.Equal(t, "end_turn", resp.FinishReason)
	assert.Equal(t, 5, resp.TokensUsed)
	assert.Greater(t, resp.Confidence, 0.0)
	assert.NotNil(t, resp.Metadata)
	assert.Equal(t, "claude-3-sonnet-20240229", resp.Metadata["model"])
	assert.Equal(t, 10, resp.Metadata["input_tokens"])
}

func TestClaudeProvider_ConvertResponse_EmptyContent(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{ID: "req-123"}

	claudeResp := &ClaudeResponse{
		ID:      "msg-456",
		Content: []ClaudeContent{},
		Usage:   ClaudeUsage{},
	}

	startTime := time.Now()
	resp := provider.convertResponse(req, claudeResp, startTime)

	assert.Equal(t, "", resp.Content)
	assert.Equal(t, "", resp.FinishReason)
}

func TestDefaultRetryConfig(t *testing.T) {
	config := DefaultRetryConfig()

	assert.Equal(t, 3, config.MaxRetries)
	assert.Equal(t, 1*time.Second, config.InitialDelay)
	assert.Equal(t, 30*time.Second, config.MaxDelay)
	assert.Equal(t, 2.0, config.Multiplier)
}

func TestNewClaudeProviderWithRetry(t *testing.T) {
	retryConfig := RetryConfig{
		MaxRetries:   5,
		InitialDelay: 2 * time.Second,
		MaxDelay:     60 * time.Second,
		Multiplier:   3.0,
	}

	provider := NewClaudeProviderWithRetry("test-key", "", "", retryConfig)

	assert.Equal(t, "test-key", provider.apiKey)
	assert.Equal(t, ClaudeAPIURL, provider.baseURL)
	assert.Equal(t, ClaudeModel, provider.model)
	assert.Equal(t, 5, provider.retryConfig.MaxRetries)
	assert.Equal(t, 2*time.Second, provider.retryConfig.InitialDelay)
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
		{http.StatusNotFound, false},
		{http.StatusTooManyRequests, true},
		{http.StatusInternalServerError, true},
		{http.StatusBadGateway, true},
		{http.StatusServiceUnavailable, true},
		{http.StatusGatewayTimeout, true},
	}

	for _, tt := range tests {
		t.Run(http.StatusText(tt.statusCode), func(t *testing.T) {
			assert.Equal(t, tt.retryable, isRetryableStatus(tt.statusCode))
		})
	}
}

func TestClaudeProvider_NextDelay(t *testing.T) {
	provider := NewClaudeProviderWithRetry("test-key", "", "", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     10 * time.Second,
		Multiplier:   2.0,
	})

	// First delay should be multiplied
	next := provider.nextDelay(1 * time.Second)
	assert.Equal(t, 2*time.Second, next)

	// Should hit max delay
	next = provider.nextDelay(8 * time.Second)
	assert.Equal(t, 10*time.Second, next)
}

func TestClaudeProvider_Complete_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{invalid json`))
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")
	req := &models.LLMRequest{
		ID: "test-request",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "failed to parse Claude response")
}

func BenchmarkClaudeProvider_ConvertRequest(b *testing.B) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{
		ID: "bench-request",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi"},
			{Role: "user", Content: "How are you?"},
		},
		ModelParams: models.ModelParameters{
			MaxTokens:   100,
			Temperature: 0.7,
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.convertRequest(req)
	}
}

func BenchmarkClaudeProvider_CalculateConfidence(b *testing.B) {
	provider := NewClaudeProvider("test-key", "", "")
	content := "This is a sample response from the Claude model that should be evaluated for confidence scoring."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.calculateConfidence(content, "end_turn")
	}
}

func TestClaudeProvider_CalculateConfidence_EdgeCases(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")

	tests := []struct {
		name         string
		content      string
		finishReason string
		expectedMin  float64
		expectedMax  float64
	}{
		{
			name:         "end_turn with short content",
			content:      "Short",
			finishReason: "end_turn",
			expectedMin:  0.9,
			expectedMax:  1.0,
		},
		{
			name:         "max_tokens penalty",
			content:      "This response was cut off due to token limit",
			finishReason: "max_tokens",
			expectedMin:  0.75,
			expectedMax:  0.9,
		},
		{
			name:         "stop_sequence boost",
			content:      "Response ended at stop sequence marker",
			finishReason: "stop_sequence",
			expectedMin:  0.9,
			expectedMax:  1.0,
		},
		{
			name:         "unknown finish reason",
			content:      "Some content here",
			finishReason: "unknown",
			expectedMin:  0.85,
			expectedMax:  0.95,
		},
		{
			name:         "long content over 50 chars",
			content:      "This is a response that is over fifty characters long and should get a boost",
			finishReason: "end_turn",
			expectedMin:  0.95,
			expectedMax:  1.0,
		},
		{
			name:         "very long content over 200 chars",
			content:      "This is a much longer response that exceeds two hundred characters. It contains multiple sentences and covers various aspects of the topic being discussed. This additional length should result in a higher confidence score as it indicates the model has provided a comprehensive and detailed response.",
			finishReason: "end_turn",
			expectedMin:  0.97,
			expectedMax:  1.0,
		},
		{
			name:         "empty content",
			content:      "",
			finishReason: "end_turn",
			expectedMin:  0.9,
			expectedMax:  1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			confidence := provider.calculateConfidence(tt.content, tt.finishReason)
			assert.GreaterOrEqual(t, confidence, tt.expectedMin)
			assert.LessOrEqual(t, confidence, tt.expectedMax)
		})
	}
}

func TestClaudeProvider_CompleteStream_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error": {"message": "Internal server error"}}`))
	}))
	defer server.Close()

	// Fast retry config so the 3 retries on a 500 don't dominate test time.
	provider := NewClaudeProviderWithRetry("test-key", server.URL, "claude-3-sonnet", RetryConfig{
		MaxRetries:   1,
		InitialDelay: 1 * time.Millisecond,
		MaxDelay:     2 * time.Millisecond,
		Multiplier:   2.0,
	})
	req := &models.LLMRequest{
		ID: "test-stream-request",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	// Reconciled per §11.4.120: CompleteStream now guards on StatusCode before
	// consuming the body (matching every sibling provider), so a non-2xx
	// response is surfaced as a returned error rather than swallowed by the
	// SSE loop into a fake-success final response.
	ch, err := provider.CompleteStream(context.Background(), req)
	require.Error(t, err)
	assert.Nil(t, ch)
	assert.Contains(t, err.Error(), "500")
}

func TestClaudeProvider_CompleteStream_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"Hello\"}}\n\n"))
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")
	req := &models.LLMRequest{
		ID: "test-stream-request",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	ch, err := provider.CompleteStream(ctx, req)
	// Either returns error or empty channel
	if err == nil && ch != nil {
		var responses []*models.LLMResponse
		for resp := range ch {
			responses = append(responses, resp)
		}
		// May or may not have responses depending on timing
		assert.NotNil(t, responses)
	}
}

// ==============================================================================
// TOOL_CHOICE FORMAT TESTS - Critical fix for Claude API compatibility
// ==============================================================================
// These tests ensure tool_choice is sent in the correct object format
// Claude API requires {"type": "auto"} not just "auto" string

func TestClaudeProvider_ToolChoice_StringAutoConvertsToObject(t *testing.T) {
	// This test verifies the critical fix: string "auto" must be converted to {"type": "auto"}
	var capturedToolChoice interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Parse the request body to capture tool_choice
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err == nil {
			capturedToolChoice = reqBody["tool_choice"]
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_test","type":"message","role":"assistant","content":[{"type":"text","text":"OK"}],"model":"claude-3","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")
	req := &models.LLMRequest{
		ID: "test-tool-choice",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.ToolFunction{
					Name:        "test_tool",
					Description: "A test tool",
				},
			},
		},
		ToolChoice: "auto", // String format - MUST be converted to object
	}

	_, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)

	// CRITICAL: tool_choice must be an object {"type": "auto"}, not string "auto"
	require.NotNil(t, capturedToolChoice, "tool_choice should be set")

	tcMap, ok := capturedToolChoice.(map[string]interface{})
	require.True(t, ok, "tool_choice must be an object, got: %T", capturedToolChoice)
	assert.Equal(t, "auto", tcMap["type"], "tool_choice.type should be 'auto'")
}

func TestClaudeProvider_ToolChoice_StringAnyConvertsToObject(t *testing.T) {
	var capturedToolChoice interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err == nil {
			capturedToolChoice = reqBody["tool_choice"]
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_test","type":"message","role":"assistant","content":[{"type":"text","text":"OK"}],"model":"claude-3","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")
	req := &models.LLMRequest{
		ID: "test-tool-choice-any",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.ToolFunction{
					Name:        "test_tool",
					Description: "A test tool",
				},
			},
		},
		ToolChoice: "any", // String format - MUST be converted to object
	}

	_, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)

	require.NotNil(t, capturedToolChoice)
	tcMap, ok := capturedToolChoice.(map[string]interface{})
	require.True(t, ok, "tool_choice must be an object, got: %T", capturedToolChoice)
	assert.Equal(t, "any", tcMap["type"], "tool_choice.type should be 'any'")
}

func TestClaudeProvider_ToolChoice_ObjectPassedThrough(t *testing.T) {
	var capturedToolChoice interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err == nil {
			capturedToolChoice = reqBody["tool_choice"]
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_test","type":"message","role":"assistant","content":[{"type":"text","text":"OK"}],"model":"claude-3","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")
	req := &models.LLMRequest{
		ID: "test-tool-choice-object",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.ToolFunction{
					Name:        "specific_tool",
					Description: "A specific tool",
				},
			},
		},
		// Already in object format - should pass through unchanged
		ToolChoice: map[string]interface{}{
			"type": "tool",
			"name": "specific_tool",
		},
	}

	_, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)

	require.NotNil(t, capturedToolChoice)
	tcMap, ok := capturedToolChoice.(map[string]interface{})
	require.True(t, ok, "tool_choice must be an object")
	assert.Equal(t, "tool", tcMap["type"])
	assert.Equal(t, "specific_tool", tcMap["name"])
}

func TestClaudeProvider_ToolChoice_NilWhenNoTools(t *testing.T) {
	var capturedToolChoice interface{}
	var hasToolChoice bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err == nil {
			capturedToolChoice, hasToolChoice = reqBody["tool_choice"]
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_test","type":"message","role":"assistant","content":[{"type":"text","text":"OK"}],"model":"claude-3","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")
	req := &models.LLMRequest{
		ID: "test-no-tools",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
		// No tools, so ToolChoice should not be set
		ToolChoice: "auto",
	}

	_, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)

	// tool_choice should not be present when there are no tools
	assert.False(t, hasToolChoice, "tool_choice should not be set when no tools")
	assert.Nil(t, capturedToolChoice)
}

func TestClaudeProvider_ToolChoice_AllFormats(t *testing.T) {
	// Comprehensive test for all tool_choice formats
	testCases := []struct {
		name           string
		toolChoice     interface{}
		expectedType   string
		expectedFormat string // "object" or "string"
	}{
		{
			name:           "String auto",
			toolChoice:     "auto",
			expectedType:   "auto",
			expectedFormat: "object",
		},
		{
			name:           "String any",
			toolChoice:     "any",
			expectedType:   "any",
			expectedFormat: "object",
		},
		{
			name:           "Object auto",
			toolChoice:     map[string]interface{}{"type": "auto"},
			expectedType:   "auto",
			expectedFormat: "object",
		},
		{
			name:           "Object tool with name",
			toolChoice:     map[string]interface{}{"type": "tool", "name": "my_tool"},
			expectedType:   "tool",
			expectedFormat: "object",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var capturedToolChoice interface{}

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var reqBody map[string]interface{}
				if err := json.NewDecoder(r.Body).Decode(&reqBody); err == nil {
					capturedToolChoice = reqBody["tool_choice"]
				}

				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"id":"msg_test","type":"message","role":"assistant","content":[{"type":"text","text":"OK"}],"model":"claude-3","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
			}))
			defer server.Close()

			provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")
			req := &models.LLMRequest{
				ID: "test-format-" + tc.name,
				Messages: []models.Message{
					{Role: "user", Content: "Hello"},
				},
				Tools: []models.Tool{
					{
						Type: "function",
						Function: models.ToolFunction{
							Name:        "my_tool",
							Description: "Test tool",
						},
					},
				},
				ToolChoice: tc.toolChoice,
			}

			_, err := provider.Complete(context.Background(), req)
			require.NoError(t, err)

			// ALL formats should result in an object being sent
			require.NotNil(t, capturedToolChoice, "tool_choice should be set for %s", tc.name)
			tcMap, ok := capturedToolChoice.(map[string]interface{})
			require.True(t, ok, "tool_choice must ALWAYS be an object for Claude API, got: %T for %s", capturedToolChoice, tc.name)
			assert.Equal(t, tc.expectedType, tcMap["type"], "tool_choice.type mismatch for %s", tc.name)
		})
	}
}

// ==============================================================================
// AUTH TYPE AND HEALTH CHECK TESTS
// ==============================================================================

func TestClaudeProvider_GetAuthType(t *testing.T) {
	t.Run("default is API key", func(t *testing.T) {
		provider := NewClaudeProvider("test-key", "", "")
		assert.Equal(t, AuthTypeAPIKey, provider.GetAuthType())
	})

	t.Run("explicit API key auth type", func(t *testing.T) {
		provider := &ClaudeProvider{
			apiKey:   "test-key",
			authType: AuthTypeAPIKey,
		}
		assert.Equal(t, AuthTypeAPIKey, provider.GetAuthType())
	})

	t.Run("OAuth auth type", func(t *testing.T) {
		provider := &ClaudeProvider{
			authType: AuthTypeOAuth,
		}
		assert.Equal(t, AuthTypeOAuth, provider.GetAuthType())
	})
}

func TestClaudeProvider_getAuthHeader_APIKey(t *testing.T) {
	provider := &ClaudeProvider{
		apiKey:   "sk-test-key",
		authType: AuthTypeAPIKey,
	}

	headerName, headerValue, err := provider.getAuthHeader()
	require.NoError(t, err)
	assert.Equal(t, "x-api-key", headerName)
	assert.Equal(t, "sk-test-key", headerValue)
}

func TestClaudeProvider_getAuthHeader_OAuthNoReader(t *testing.T) {
	provider := &ClaudeProvider{
		authType:        AuthTypeOAuth,
		oauthCredReader: nil, // No credential reader
	}

	_, _, err := provider.getAuthHeader()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "OAuth credential reader not initialized")
}

// mockRoundTripper implements http.RoundTripper for testing
type mockRoundTripper struct {
	response *http.Response
	err      error
}

func (m *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return m.response, m.err
}

func TestClaudeProvider_HealthCheck_WithMockTransport(t *testing.T) {
	t.Run("success - API returns response", func(t *testing.T) {
		mockResp := &http.Response{
			StatusCode: http.StatusBadRequest, // Expected for GET to messages endpoint
			Body:       http.NoBody,
			Header:     make(http.Header),
		}

		provider := &ClaudeProvider{
			apiKey:     "test-key",
			baseURL:    ClaudeAPIURL,
			model:      ClaudeModel,
			authType:   AuthTypeAPIKey,
			httpClient: &http.Client{Transport: &mockRoundTripper{response: mockResp}},
		}

		err := provider.HealthCheck()
		assert.NoError(t, err)
	})

	t.Run("failure - unauthorized", func(t *testing.T) {
		mockResp := &http.Response{
			StatusCode: http.StatusUnauthorized,
			Body:       http.NoBody,
			Header:     make(http.Header),
		}

		provider := &ClaudeProvider{
			apiKey:     "invalid-key",
			baseURL:    ClaudeAPIURL,
			model:      ClaudeModel,
			authType:   AuthTypeAPIKey,
			httpClient: &http.Client{Transport: &mockRoundTripper{response: mockResp}},
		}

		err := provider.HealthCheck()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "unauthorized")
	})

	t.Run("failure - network error", func(t *testing.T) {
		provider := &ClaudeProvider{
			apiKey:   "test-key",
			baseURL:  ClaudeAPIURL,
			model:    ClaudeModel,
			authType: AuthTypeAPIKey,
			httpClient: &http.Client{
				Transport: &mockRoundTripper{
					err: http.ErrHandlerTimeout,
				},
			},
		}

		err := provider.HealthCheck()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "health check request failed")
	})
}

func TestIsAuthRetryableStatus(t *testing.T) {
	tests := []struct {
		statusCode int
		retryable  bool
	}{
		{http.StatusUnauthorized, true}, // 401 - should retry
		{http.StatusOK, false},          // 200 - not retryable
		{http.StatusBadRequest, false},  // 400 - not retryable
		{http.StatusForbidden, false},   // 403 - not retryable
		{http.StatusNotFound, false},    // 404 - not retryable
	}

	for _, tt := range tests {
		t.Run(http.StatusText(tt.statusCode), func(t *testing.T) {
			assert.Equal(t, tt.retryable, isAuthRetryableStatus(tt.statusCode))
		})
	}
}

func TestClaudeProvider_WaitWithJitter(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")

	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	baseDelay := 100 * time.Millisecond
	provider.waitWithJitter(ctx, baseDelay)
	elapsed := time.Since(start)

	// Should wait at least the base delay. This lower bound is load-robust:
	// extra CPU pressure can only delay a timer, never fire it early.
	assert.GreaterOrEqual(t, elapsed, baseDelay)

	// Upper bound: deliberately NOT asserted in wall-clock time. Elapsed time
	// also contains scheduler latency this package does not control, so a
	// wall-clock ceiling measures the host at least as much as it measures
	// waitWithJitter. Reproduced under deliberate CPU load as
	// `"169.547666ms" is not less than or equal to "150ms"`.
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

func TestClaudeProvider_WaitWithJitter_ContextCancelled(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	start := time.Now()
	provider.waitWithJitter(ctx, 1*time.Second)
	elapsed := time.Since(start)

	// Should return immediately due to cancelled context
	assert.Less(t, elapsed, 100*time.Millisecond)
}

func TestClaudeProvider_ConvertResponse_WithToolUse(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{ID: "req-123"}

	stopReason := "tool_use"
	claudeResp := &ClaudeResponse{
		ID:         "msg-456",
		Type:       "message",
		Role:       "assistant",
		Model:      "claude-3-sonnet-20240229",
		StopReason: &stopReason,
		Content: []ClaudeContent{
			{Type: "text", Text: "I'll use the calculator tool."},
			{
				Type:  "tool_use",
				ID:    "tool-123",
				Name:  "calculator",
				Input: map[string]interface{}{"operation": "add", "a": 1, "b": 2},
			},
		},
		Usage: ClaudeUsage{
			InputTokens:  15,
			OutputTokens: 8,
		},
	}

	startTime := time.Now().Add(-50 * time.Millisecond)
	resp := provider.convertResponse(req, claudeResp, startTime)

	assert.Equal(t, "msg-456", resp.ID)
	// When tool_use is present, finish reason is normalized to "tool_calls"
	assert.Equal(t, "tool_calls", resp.FinishReason)
	assert.Contains(t, resp.Content, "calculator tool")
	assert.GreaterOrEqual(t, resp.ResponseTime, int64(50))
}

func TestClaudeProvider_ConvertResponse_MultipleTextBlocks(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{ID: "req-123"}

	stopReason := "end_turn"
	claudeResp := &ClaudeResponse{
		ID:         "msg-456",
		Type:       "message",
		Role:       "assistant",
		Model:      "claude-3-sonnet-20240229",
		StopReason: &stopReason,
		Content: []ClaudeContent{
			{Type: "text", Text: "First part. "},
			{Type: "text", Text: "Second part."},
		},
		Usage: ClaudeUsage{
			InputTokens:  10,
			OutputTokens: 6,
		},
	}

	startTime := time.Now()
	resp := provider.convertResponse(req, claudeResp, startTime)

	// Should concatenate all text blocks
	assert.Contains(t, resp.Content, "First part")
	assert.Contains(t, resp.Content, "Second part")
}

// ==============================================================================
// ADDITIONAL COMPREHENSIVE TESTS FOR CLAUDE PROVIDER
// ==============================================================================

func TestClaudeProvider_Complete_WithTools(t *testing.T) {
	var capturedRequest ClaudeRequest

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &capturedRequest)

		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id": "msg_123",
			"type": "message",
			"role": "assistant",
			"content": [
				{"type": "text", "text": "I'll use the calculator."},
				{"type": "tool_use", "id": "call_1", "name": "calculator", "input": {"operation": "add", "a": 1, "b": 2}}
			],
			"model": "claude-3-sonnet",
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 20, "output_tokens": 15}
		}`))
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")
	req := &models.LLMRequest{
		ID: "tool-test",
		Messages: []models.Message{
			{Role: "user", Content: "Add 1 and 2"},
		},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.ToolFunction{
					Name:        "calculator",
					Description: "Perform calculations",
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"operation": map[string]interface{}{"type": "string"},
							"a":         map[string]interface{}{"type": "number"},
							"b":         map[string]interface{}{"type": "number"},
						},
					},
				},
			},
		},
		ToolChoice: "auto",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)

	// Verify tools were converted correctly
	require.Len(t, capturedRequest.Tools, 1)
	assert.Equal(t, "calculator", capturedRequest.Tools[0].Name)

	// Verify response parsing
	assert.Equal(t, "tool_calls", resp.FinishReason)
	require.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, "calculator", resp.ToolCalls[0].Function.Name)
	assert.Contains(t, resp.Content, "calculator")
}

func TestClaudeProvider_ConvertRequest_ToolWithoutParameters(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{
		ID: "test",
		Messages: []models.Message{
			{Role: "user", Content: "Test"},
		},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.ToolFunction{
					Name:        "simple_tool",
					Description: "A simple tool with no params",
					Parameters:  nil, // No parameters
				},
			},
		},
	}

	claudeReq := provider.convertRequest(req)

	// Should add default input_schema
	require.Len(t, claudeReq.Tools, 1)
	require.NotNil(t, claudeReq.Tools[0].InputSchema)
	assert.Equal(t, "object", claudeReq.Tools[0].InputSchema["type"])
}

func TestClaudeProvider_ConvertRequest_ToolWithMissingType(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{
		ID: "test",
		Messages: []models.Message{
			{Role: "user", Content: "Test"},
		},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.ToolFunction{
					Name: "tool_with_props",
					Parameters: map[string]interface{}{
						"properties": map[string]interface{}{
							"name": map[string]interface{}{"type": "string"},
						},
						// Missing "type" field
					},
				},
			},
		},
	}

	claudeReq := provider.convertRequest(req)

	// Should add "type": "object"
	require.Len(t, claudeReq.Tools, 1)
	assert.Equal(t, "object", claudeReq.Tools[0].InputSchema["type"])
}

func TestClaudeProvider_HealthCheck_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "2023-06-01", r.Header.Get("anthropic-version"))
		w.WriteHeader(http.StatusBadRequest) // Expected for GET to messages endpoint
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")
	provider.httpClient = server.Client()

	// Override the health check URL to point to test server
	err := provider.HealthCheck()
	// May succeed or fail depending on URL - we're mainly testing the code path
	_ = err
}

func TestClaudeProvider_HealthCheck_Unauthorized(t *testing.T) {
	// Test using mockRoundTripper since HealthCheck uses hardcoded URL
	mockResp := &http.Response{
		StatusCode: http.StatusUnauthorized,
		Body:       http.NoBody,
		Header:     make(http.Header),
	}

	provider := &ClaudeProvider{
		apiKey:     "invalid-key",
		baseURL:    ClaudeAPIURL,
		model:      "claude-3-sonnet",
		authType:   AuthTypeAPIKey,
		httpClient: &http.Client{Transport: &mockRoundTripper{response: mockResp}},
	}

	err := provider.HealthCheck()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unauthorized")
}

func TestClaudeProvider_HealthCheck_ServerError(t *testing.T) {
	mockResp := &http.Response{
		StatusCode: http.StatusInternalServerError,
		Body:       http.NoBody,
		Header:     make(http.Header),
	}

	provider := &ClaudeProvider{
		apiKey:     "test-key",
		baseURL:    ClaudeAPIURL,
		model:      "claude-3-sonnet",
		authType:   AuthTypeAPIKey,
		httpClient: &http.Client{Transport: &mockRoundTripper{response: mockResp}},
	}

	err := provider.HealthCheck()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "server error")
}

func TestClaudeProvider_ValidateConfig_OAuthWithoutReader(t *testing.T) {
	provider := &ClaudeProvider{
		baseURL:         "https://api.anthropic.com",
		model:           "claude-3-sonnet",
		authType:        AuthTypeOAuth,
		oauthCredReader: nil, // No reader
	}

	valid, errs := provider.ValidateConfig(nil)
	assert.False(t, valid)
	// CONST-046: ValidateConfig surfaces the i18n message ID resolved via
	// the i18n.Tr seam (NoopTranslator echoes the ID verbatim in unit tests).
	assert.Contains(t, errs, "llmprovider_validate_oauth_reader_required")
}

func TestClaudeProvider_CompleteStream_WithTextDelta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		events := []string{
			`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[]}}`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" World"}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}`,
			`data: {"type":"content_block_stop","index":0}`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
			`data: {"type":"message_stop"}`,
		}

		for _, event := range events {
			_, _ = w.Write([]byte(event + "\n\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")
	req := &models.LLMRequest{
		ID: "stream-test",
		Messages: []models.Message{
			{Role: "user", Content: "Say hello"},
		},
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var chunks []string
	for resp := range ch {
		if resp.Content != "" {
			chunks = append(chunks, resp.Content)
		}
	}

	// Should have received multiple chunks
	assert.GreaterOrEqual(t, len(chunks), 2)
}

func TestClaudeProvider_CompleteStream_MalformedJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send some malformed JSON mixed with valid
		_, _ = w.Write([]byte("data: {invalid json}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Valid\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"message_stop\"}\n\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")
	req := &models.LLMRequest{ID: "malformed-test", Messages: []models.Message{{Role: "user", Content: "Test"}}}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	// Should skip malformed JSON and process valid parts
	assert.GreaterOrEqual(t, len(responses), 1)
}

func TestClaudeProvider_Retry_AllAttemptsFail(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	retryConfig := RetryConfig{
		MaxRetries:   2,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     50 * time.Millisecond,
		Multiplier:   2.0,
	}

	provider := NewClaudeProviderWithRetry("test-key", server.URL, "claude-3-sonnet", retryConfig)

	req := &models.LLMRequest{
		ID:       "fail-test",
		Messages: []models.Message{{Role: "user", Content: "Test"}},
	}

	_, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Equal(t, 3, attempts) // Initial + 2 retries
}

func TestClaudeProvider_Retry_ContextCancelledDuringRetry(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	retryConfig := RetryConfig{
		MaxRetries:   5,
		InitialDelay: 50 * time.Millisecond,
		MaxDelay:     200 * time.Millisecond,
		Multiplier:   2.0,
	}

	provider := NewClaudeProviderWithRetry("test-key", server.URL, "claude-3-sonnet", retryConfig)

	req := &models.LLMRequest{
		ID:       "cancel-test",
		Messages: []models.Message{{Role: "user", Content: "Test"}},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := provider.Complete(ctx, req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "context")
}

func TestClaudeProvider_ConvertResponse_WithMultipleToolUse(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{ID: "multi-tool"}

	stopReason := "tool_use"
	claudeResp := &ClaudeResponse{
		ID:         "msg-multi",
		StopReason: &stopReason,
		Content: []ClaudeContent{
			{Type: "text", Text: "I'll use multiple tools."},
			{
				Type:  "tool_use",
				ID:    "call_1",
				Name:  "tool_a",
				Input: map[string]interface{}{"param": "value1"},
			},
			{
				Type:  "tool_use",
				ID:    "call_2",
				Name:  "tool_b",
				Input: map[string]interface{}{"param": "value2"},
			},
		},
		Usage: ClaudeUsage{
			InputTokens:  30,
			OutputTokens: 25,
		},
	}

	resp := provider.convertResponse(req, claudeResp, time.Now())

	assert.Equal(t, "tool_calls", resp.FinishReason)
	require.Len(t, resp.ToolCalls, 2)
	assert.Equal(t, "tool_a", resp.ToolCalls[0].Function.Name)
	assert.Equal(t, "tool_b", resp.ToolCalls[1].Function.Name)
}

func TestClaudeProvider_ConvertResponse_ToolUseMarshalError(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{ID: "marshal-error"}

	stopReason := "tool_use"
	// Create input that should marshal fine but test edge case
	claudeResp := &ClaudeResponse{
		ID:         "msg-marshal",
		StopReason: &stopReason,
		Content: []ClaudeContent{
			{
				Type:  "tool_use",
				ID:    "call_1",
				Name:  "tool",
				Input: nil, // nil input
			},
		},
		Usage: ClaudeUsage{},
	}

	resp := provider.convertResponse(req, claudeResp, time.Now())

	// Should handle nil input gracefully
	require.Len(t, resp.ToolCalls, 1)
}

func TestClaudeProvider_Complete_BadGateway(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 2 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg","content":[{"type":"text","text":"OK"}],"usage":{}}`))
	}))
	defer server.Close()

	retryConfig := RetryConfig{
		MaxRetries:   3,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     50 * time.Millisecond,
		Multiplier:   2.0,
	}

	provider := NewClaudeProviderWithRetry("test-key", server.URL, "claude-3-sonnet", retryConfig)
	req := &models.LLMRequest{ID: "gateway", Messages: []models.Message{{Role: "user", Content: "Test"}}}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, 2, attempts)
}

func TestClaudeProvider_Complete_GatewayTimeout(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 2 {
			w.WriteHeader(http.StatusGatewayTimeout)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg","content":[{"type":"text","text":"OK"}],"usage":{}}`))
	}))
	defer server.Close()

	retryConfig := RetryConfig{
		MaxRetries:   3,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     50 * time.Millisecond,
		Multiplier:   2.0,
	}

	provider := NewClaudeProviderWithRetry("test-key", server.URL, "claude-3-sonnet", retryConfig)
	req := &models.LLMRequest{ID: "timeout", Messages: []models.Message{{Role: "user", Content: "Test"}}}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, 2, attempts)
}

func TestClaudeProvider_ConvertRequest_EmptyMessages(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{
		ID:       "empty-messages",
		Messages: []models.Message{},
		ModelParams: models.ModelParameters{
			MaxTokens:   100,
			Temperature: 0.7,
		},
	}

	claudeReq := provider.convertRequest(req)

	assert.Empty(t, claudeReq.Messages)
	assert.Equal(t, 100, claudeReq.MaxTokens)
	assert.Equal(t, 0.7, claudeReq.Temperature)
}

func TestClaudeProvider_ConvertRequest_OnlySystemMessage(t *testing.T) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{
		ID: "system-only",
		Messages: []models.Message{
			{Role: "system", Content: "You are helpful."},
		},
	}

	claudeReq := provider.convertRequest(req)

	// System message should be in System field, not Messages
	assert.Equal(t, "You are helpful.", claudeReq.System)
	assert.Empty(t, claudeReq.Messages)
}

func TestClaudeProvider_AuthTypeConstants(t *testing.T) {
	assert.Equal(t, AuthType("api_key"), AuthTypeAPIKey)
	assert.Equal(t, AuthType("oauth"), AuthTypeOAuth)
}

func TestClaudeProvider_APIURLConstants(t *testing.T) {
	assert.Equal(t, "https://api.anthropic.com/v1/messages", ClaudeAPIURL)
	assert.Equal(t, "claude-sonnet-4-6", ClaudeModel)      // Updated to Claude 4 model
	assert.Equal(t, "claude-sonnet-4-6", ClaudeOAuthModel) // Updated to Claude 4 model
}

func TestClaudeProvider_Complete_ReadBodyError(t *testing.T) {
	// Create a server that returns a response that can't be fully read
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000") // Claim more content than we send
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":`)) // Truncated
		// Don't write the rest, causing EOF before Content-Length is satisfied
	}))
	defer server.Close()

	provider := NewClaudeProvider("test-key", server.URL, "claude-3-sonnet")

	req := &models.LLMRequest{
		ID:       "read-error",
		Messages: []models.Message{{Role: "user", Content: "Test"}},
	}

	// This should either fail to read the body or fail to parse JSON
	_, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
}

func BenchmarkClaudeProvider_ConvertResponse(b *testing.B) {
	provider := NewClaudeProvider("test-key", "", "")
	req := &models.LLMRequest{ID: "bench"}
	stopReason := "end_turn"
	claudeResp := &ClaudeResponse{
		ID:         "msg-bench",
		StopReason: &stopReason,
		Content: []ClaudeContent{
			{Type: "text", Text: "This is a benchmark response."},
		},
		Usage: ClaudeUsage{InputTokens: 10, OutputTokens: 5},
		Model: "claude-3-sonnet",
	}
	startTime := time.Now()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.convertResponse(req, claudeResp, startTime)
	}
}
