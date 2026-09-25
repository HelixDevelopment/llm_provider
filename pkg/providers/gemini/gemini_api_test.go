package gemini

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

// ==============================================================================
// API Provider Tests (using GeminiAPIProvider directly)
// ==============================================================================

func TestNewGeminiAPIProvider(t *testing.T) {
	p := NewGeminiAPIProvider("test-key", "", "")
	assert.NotNil(t, p)
	assert.Equal(t, GeminiDefaultModel, p.model)
	assert.Equal(t, "gemini-api", p.GetName())
	assert.Equal(t, "gemini", p.GetProviderType())
}

func TestGeminiAPIProvider_Complete_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Equal(t, "test-api-key", r.Header.Get("x-goog-api-key"))
		assert.Equal(t, "LLMProvider/1.0", r.Header.Get("User-Agent"))

		var reqBody GeminiAPIRequest
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		err = json.Unmarshal(body, &reqBody)
		require.NoError(t, err)

		assert.Len(t, reqBody.Contents, 1)
		assert.Equal(t, "user", reqBody.Contents[0].Role)
		assert.Len(t, reqBody.Contents[0].Parts, 1)
		assert.Equal(t, "Hello, how are you?", reqBody.Contents[0].Parts[0].Text)
		assert.Equal(t, 0.7, reqBody.GenerationConfig.Temperature)
		assert.Equal(t, 1000, reqBody.GenerationConfig.MaxOutputTokens)

		response := GeminiResponse{
			Candidates: []GeminiCandidate{
				{
					Content: GeminiContent{
						Parts: []GeminiPart{
							{Text: "I'm doing well, thank you for asking!"},
						},
						Role: "model",
					},
					FinishReason: "STOP",
					Index:        0,
				},
			},
			UsageMetadata: &GeminiUsageMetadata{
				PromptTokenCount:     10,
				CandidatesTokenCount: 20,
				TotalTokenCount:      30,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("test-api-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-123",
		ModelParams: models.ModelParameters{
			Model:       "gemini-pro",
			MaxTokens:   1000,
			Temperature: 0.7,
		},
		Prompt: "Hello, how are you?",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)

	assert.Equal(t, "gemini-api-test-req-123", resp.ID)
	assert.Equal(t, "test-req-123", resp.RequestID)
	assert.Equal(t, "gemini-api", resp.ProviderID)
	assert.Equal(t, "Gemini", resp.ProviderName)
	assert.Equal(t, "I'm doing well, thank you for asking!", resp.Content)
	assert.Greater(t, resp.Confidence, 0.8)
	assert.Equal(t, 30, resp.TokensUsed)
	assert.Equal(t, "STOP", resp.FinishReason)
	assert.Equal(t, "gemini-pro", resp.Metadata["model"])
}

func TestGeminiAPIProvider_Complete_WithMessages(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody GeminiAPIRequest
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		err = json.Unmarshal(body, &reqBody)
		require.NoError(t, err)

		assert.Len(t, reqBody.Contents, 3)
		assert.Equal(t, "user", reqBody.Contents[0].Role)
		assert.Equal(t, "You are a helpful assistant", reqBody.Contents[0].Parts[0].Text)
		assert.Equal(t, "user", reqBody.Contents[1].Role)
		assert.Equal(t, "Hello", reqBody.Contents[1].Parts[0].Text)
		assert.Equal(t, "model", reqBody.Contents[2].Role)
		assert.Equal(t, "Hi there!", reqBody.Contents[2].Parts[0].Text)

		response := GeminiResponse{
			Candidates: []GeminiCandidate{
				{
					Content: GeminiContent{
						Parts: []GeminiPart{
							{Text: "How can I help you today?"},
						},
						Role: "model",
					},
					FinishReason: "STOP",
					Index:        0,
				},
			},
			UsageMetadata: &GeminiUsageMetadata{
				PromptTokenCount:     15,
				CandidatesTokenCount: 25,
				TotalTokenCount:      40,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("test-api-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-messages",
		ModelParams: models.ModelParameters{
			Model: "gemini-pro",
		},
		Prompt: "You are a helpful assistant",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
		},
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "How can I help you today?", resp.Content)
	assert.Equal(t, 40, resp.TokensUsed)
}

func TestGeminiAPIProvider_Complete_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error": {"code": 401, "message": "API key not valid"}}`))
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("invalid-api-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-456",
		ModelParams: models.ModelParameters{
			Model: "gemini-pro",
		},
		Prompt: "Test prompt",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "Gemini API error: 401")
}

func TestGeminiAPIProvider_Complete_NoCandidates(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := GeminiResponse{
			Candidates: []GeminiCandidate{},
			UsageMetadata: &GeminiUsageMetadata{
				PromptTokenCount:     5,
				CandidatesTokenCount: 0,
				TotalTokenCount:      5,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("test-api-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-789",
		ModelParams: models.ModelParameters{
			Model: "gemini-pro",
		},
		Prompt: "Test prompt",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "", resp.Content)
	assert.Equal(t, 5, resp.TokensUsed)
}

func TestGeminiAPIProvider_Complete_NetworkError(t *testing.T) {
	provider := NewGeminiAPIProvider("test-api-key", "https://invalid-url-that-does-not-exist.example.com/v1beta/models/%s:generateContent", "gemini-pro")
	provider.httpClient = &http.Client{
		Timeout: 1 * time.Millisecond,
	}

	req := &models.LLMRequest{
		ID: "test-req-network",
		ModelParams: models.ModelParameters{
			Model: "gemini-pro",
		},
		Prompt: "Test prompt",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "Gemini API call failed")
}

func TestGeminiAPIProvider_Complete_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("invalid json"))
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("test-api-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-json",
		ModelParams: models.ModelParameters{
			Model: "gemini-pro",
		},
		Prompt: "Test prompt",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "failed to parse Gemini response")
}

func TestGeminiAPIProvider_CompleteStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		streamData := []string{
			`data: {"candidates":[{"content":{"parts":[{"text":"Hello "}],"role":"model"},"finishReason":"","index":0}]}`,
			`data: {"candidates":[{"content":{"parts":[{"text":"world"}],"role":"model"},"finishReason":"","index":0}]}`,
			`data: {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP","index":0}]}`,
			`data: [DONE]`,
		}

		for _, data := range streamData {
			_, _ = w.Write([]byte(data + "\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("test-api-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-stream",
		ModelParams: models.ModelParameters{
			Model: "gemini-pro",
		},
		Prompt: "Say hello",
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, ch)

	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	assert.GreaterOrEqual(t, len(responses), 3)
	assert.Equal(t, "gemini-api", responses[0].ProviderID)
	assert.Equal(t, "Gemini", responses[0].ProviderName)
}

func TestGeminiAPIProvider_CompleteStream_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error": "Invalid API key"}`))
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("invalid-api-key", server.URL+"/v1beta/models/%s:streamGenerateContent", "gemini-2.0-flash")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-stream-error",
		ModelParams: models.ModelParameters{
			Model: "gemini-2.0-flash",
		},
		Prompt: "Test prompt",
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.Error(t, err)
	require.Nil(t, ch)
	assert.Contains(t, err.Error(), "HTTP 401")
}

func TestGeminiAPIProvider_HealthCheck_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"models": []map[string]string{
				{"name": "gemini-pro"},
				{"name": "gemini-ultra"},
			},
		})
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("test-api-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.healthURL = server.URL + "/v1beta/models"
	provider.httpClient = server.Client()

	err := provider.HealthCheck()
	require.NoError(t, err)
}

func TestGeminiAPIProvider_HealthCheck_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error": {"code": 401, "message": "Invalid API key"}}`))
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("invalid-api-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.healthURL = server.URL + "/v1beta/models"
	provider.httpClient = server.Client()

	err := provider.HealthCheck()
	assert.Error(t, err)
}

func TestGeminiAPIProvider_CalculateConfidence(t *testing.T) {
	provider := NewGeminiAPIProvider("test-key", "", "")

	tests := []struct {
		name         string
		content      string
		finishReason string
		wantMin      float64
		wantMax      float64
	}{
		{
			name:         "STOP finish reason",
			content:      "This is a long response that should increase confidence",
			finishReason: "STOP",
			wantMin:      0.95,
			wantMax:      1.0,
		},
		{
			name:         "MAX_TOKENS finish reason",
			content:      "Short",
			finishReason: "MAX_TOKENS",
			wantMin:      0.75,
			wantMax:      0.85,
		},
		{
			name:         "SAFETY finish reason",
			content:      "Content",
			finishReason: "SAFETY",
			wantMin:      0.55,
			wantMax:      0.65,
		},
		{
			name:         "RECITATION finish reason",
			content:      "Content",
			finishReason: "RECITATION",
			wantMin:      0.64,
			wantMax:      0.75,
		},
		{
			name: "long content",
			content: "This is a very long response that exceeds 500 characters. " +
				"This is a very long response that exceeds 500 characters. " +
				"This is a very long response that exceeds 500 characters. " +
				"This is a very long response that exceeds 500 characters. " +
				"This is a very long response that exceeds 500 characters. " +
				"This is a very long response that exceeds 500 characters.",
			finishReason: "STOP",
			wantMin:      0.98,
			wantMax:      1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			confidence := provider.calculateConfidence(tt.content, tt.finishReason)
			assert.GreaterOrEqual(t, confidence, tt.wantMin)
			assert.LessOrEqual(t, confidence, tt.wantMax)
		})
	}
}

func TestGeminiAPIProvider_Complete_ContextTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		response := GeminiResponse{
			Candidates: []GeminiCandidate{
				{
					Content: GeminiContent{
						Parts: []GeminiPart{
							{Text: "Slow response"},
						},
						Role: "model",
					},
					FinishReason: "STOP",
					Index:        0,
				},
			},
			UsageMetadata: &GeminiUsageMetadata{
				PromptTokenCount:     5,
				CandidatesTokenCount: 10,
				TotalTokenCount:      15,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("test-api-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-timeout",
		ModelParams: models.ModelParameters{
			Model: "gemini-pro",
		},
		Prompt: "Test prompt",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	resp, err := provider.Complete(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "context deadline exceeded")
}

func TestGeminiAPIProvider_Complete_WithStopSequences(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody GeminiAPIRequest
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		err = json.Unmarshal(body, &reqBody)
		require.NoError(t, err)

		assert.Equal(t, []string{"STOP", "END"}, reqBody.GenerationConfig.StopSequences)

		response := GeminiResponse{
			Candidates: []GeminiCandidate{
				{
					Content: GeminiContent{
						Parts: []GeminiPart{
							{Text: "Response with stop sequences"},
						},
						Role: "model",
					},
					FinishReason: "STOP",
					Index:        0,
				},
			},
			UsageMetadata: &GeminiUsageMetadata{
				PromptTokenCount:     8,
				CandidatesTokenCount: 12,
				TotalTokenCount:      20,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("test-api-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-stop",
		ModelParams: models.ModelParameters{
			Model:         "gemini-pro",
			StopSequences: []string{"STOP", "END"},
			Temperature:   0.8,
			TopP:          0.9,
			MaxTokens:     500,
		},
		Prompt: "Generate text but stop when you see STOP or END",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "Response with stop sequences", resp.Content)
	assert.Equal(t, 20, resp.TokensUsed)
}

func TestGeminiAPIProvider_Complete_WithTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody GeminiAPIRequest
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		err = json.Unmarshal(body, &reqBody)
		require.NoError(t, err)

		// Verify tools are properly converted (user tools + googleSearch)
		require.GreaterOrEqual(t, len(reqBody.Tools), 1)

		response := GeminiResponse{
			Candidates: []GeminiCandidate{
				{
					Content: GeminiContent{
						Parts: []GeminiPart{
							{
								FunctionCall: map[string]any{
									"name": "get_weather",
									"args": map[string]interface{}{
										"location": "San Francisco",
										"unit":     "celsius",
									},
								},
							},
						},
						Role: "model",
					},
					FinishReason: "STOP",
					Index:        0,
				},
			},
			UsageMetadata: &GeminiUsageMetadata{
				PromptTokenCount:     20,
				CandidatesTokenCount: 30,
				TotalTokenCount:      50,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("test-api-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-tools",
		Prompt: "What's the weather in San Francisco?",
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.ToolFunction{
					Name:        "get_weather",
					Description: "Get weather for a location",
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{"type": "string"},
							"unit":     map[string]interface{}{"type": "string"},
						},
					},
				},
			},
			{
				Type: "function",
				Function: models.ToolFunction{
					Name:        "search_files",
					Description: "Search for files",
				},
			},
		},
		ToolChoice: "auto",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "tool_calls", resp.FinishReason)
	require.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, "get_weather", resp.ToolCalls[0].Function.Name)
}

func TestGeminiAPIProvider_ValidateConfig(t *testing.T) {
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
			baseURL:      "https://api.example.com",
			model:        "gemini-pro",
			expectValid:  true,
			expectErrLen: 0,
		},
		{
			name:         "missing api key",
			apiKey:       "",
			baseURL:      "https://api.example.com",
			model:        "gemini-pro",
			expectValid:  false,
			expectErrLen: 1,
		},
		{
			name:         "missing base url",
			apiKey:       "test-key",
			baseURL:      "",
			model:        "gemini-pro",
			expectValid:  false,
			expectErrLen: 1,
		},
		{
			name:         "missing model",
			apiKey:       "test-key",
			baseURL:      "https://api.example.com",
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
			provider := &GeminiAPIProvider{
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

func TestGeminiAPIProvider_GetCapabilities(t *testing.T) {
	provider := NewGeminiAPIProvider("", "", "")
	caps := provider.GetCapabilities()

	require.NotNil(t, caps)
	assert.Empty(t, caps.SupportedModels,
		"offline discovery must yield no models (CONST-036: no hardcoded fallback)")

	assert.True(t, caps.SupportsStreaming)
	assert.True(t, caps.SupportsFunctionCalling)
	assert.True(t, caps.SupportsVision)
	assert.True(t, caps.SupportsTools)
	assert.True(t, caps.SupportsSearch)
	assert.True(t, caps.SupportsReasoning)

	assert.Equal(t, "Google", caps.Metadata["provider"])
	assert.Equal(t, "Gemini", caps.Metadata["model_family"])
}

func TestGeminiAPIProvider_Retry_RateLimited(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		response := GeminiResponse{
			Candidates: []GeminiCandidate{
				{
					Content: GeminiContent{
						Parts: []GeminiPart{{Text: "Success after retry"}},
					},
					FinishReason: "STOP",
				},
			},
		}
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	retryConfig := RetryConfig{
		MaxRetries:   3,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     100 * time.Millisecond,
		Multiplier:   2.0,
	}

	provider := NewGeminiAPIProviderWithRetry("test-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro", retryConfig)
	provider.httpClient = server.Client()

	req := &models.LLMRequest{ID: "retry-test", Prompt: "Test"}
	resp, err := provider.Complete(context.Background(), req)

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "Success after retry", resp.Content)
	assert.Equal(t, 3, attempts)
}

func TestGeminiAPIProvider_Retry_ServerError(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		response := GeminiResponse{
			Candidates: []GeminiCandidate{
				{
					Content: GeminiContent{
						Parts: []GeminiPart{{Text: "OK"}},
					},
					FinishReason: "STOP",
				},
			},
		}
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	retryConfig := RetryConfig{
		MaxRetries:   3,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     100 * time.Millisecond,
		Multiplier:   2.0,
	}

	provider := NewGeminiAPIProviderWithRetry("test-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro", retryConfig)
	provider.httpClient = server.Client()

	req := &models.LLMRequest{ID: "retry-test", Prompt: "Test"}
	resp, err := provider.Complete(context.Background(), req)

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, 2, attempts)
}

func TestGeminiAPIProvider_Retry_AuthError(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 2 {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error": "Invalid API key"}`))
			return
		}
		response := GeminiResponse{
			Candidates: []GeminiCandidate{
				{Content: GeminiContent{Parts: []GeminiPart{{Text: "OK"}}}, FinishReason: "STOP"},
			},
		}
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	retryConfig := RetryConfig{
		MaxRetries:   3,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     100 * time.Millisecond,
		Multiplier:   2.0,
	}

	provider := NewGeminiAPIProviderWithRetry("test-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro", retryConfig)
	provider.httpClient = server.Client()

	req := &models.LLMRequest{ID: "auth-retry", Prompt: "Test"}
	resp, err := provider.Complete(context.Background(), req)

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, 2, attempts)
}

func TestIsRetryableStatus(t *testing.T) {
	tests := []struct {
		statusCode int
		retryable  bool
	}{
		{200, false},
		{400, false},
		{401, false},
		{403, false},
		{404, false},
		{429, true},
		{500, true},
		{502, true},
		{503, true},
		{504, true},
	}

	for _, tt := range tests {
		t.Run(http.StatusText(tt.statusCode), func(t *testing.T) {
			assert.Equal(t, tt.retryable, isRetryableStatus(tt.statusCode))
		})
	}
}

func TestIsAuthRetryableStatus(t *testing.T) {
	tests := []struct {
		statusCode int
		retryable  bool
	}{
		{401, true},
		{200, false},
		{403, false},
		{429, false},
	}

	for _, tt := range tests {
		t.Run(http.StatusText(tt.statusCode), func(t *testing.T) {
			assert.Equal(t, tt.retryable, isAuthRetryableStatus(tt.statusCode))
		})
	}
}

func TestGeminiAPIProvider_NextDelay(t *testing.T) {
	provider := NewGeminiAPIProviderWithRetry("test-key", "", "", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     10 * time.Second,
		Multiplier:   2.0,
	})

	next := provider.nextDelay(1 * time.Second)
	assert.Equal(t, 2*time.Second, next)

	next = provider.nextDelay(8 * time.Second)
	assert.Equal(t, 10*time.Second, next)
}

func TestGeminiAPIProvider_CompleteStream_MalformedJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		_, _ = w.Write([]byte("data: {invalid json}\n"))
		_, _ = w.Write([]byte("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Valid chunk\"}]},\"finishReason\":\"\"}]}\n"))
		_, _ = w.Write([]byte("data: [DONE]\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("test-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{ID: "malformed-test", Prompt: "Test"}
	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	assert.GreaterOrEqual(t, len(responses), 1)
}

func TestGeminiAPIProvider_CompleteStream_ArrayWrapper(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		_, _ = w.Write([]byte("[{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Array wrapped\"}]},\"finishReason\":\"STOP\"}]}]\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer server.Close()

	provider := NewGeminiAPIProvider("test-key", server.URL+"/v1beta/models/%s:generateContent", "gemini-pro")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{ID: "array-test", Prompt: "Test"}
	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	assert.GreaterOrEqual(t, len(responses), 1)
}

func TestGeminiAPIProvider_ConvertResponse_MultipleParts(t *testing.T) {
	provider := NewGeminiAPIProvider("test-key", "", "gemini-pro")
	req := &models.LLMRequest{ID: "multi-part"}

	geminiResp := &GeminiResponse{
		Candidates: []GeminiCandidate{
			{
				Content: GeminiContent{
					Parts: []GeminiPart{
						{Text: "Part one. "},
						{Text: "Part two."},
					},
				},
				FinishReason: "STOP",
			},
		},
		UsageMetadata: &GeminiUsageMetadata{
			TotalTokenCount: 10,
		},
	}

	resp := provider.convertResponse(req, geminiResp, time.Now())

	assert.Contains(t, resp.Content, "Part one")
	assert.Contains(t, resp.Content, "Part two")
}

func TestGeminiAPIProvider_HealthCheck_NetworkError(t *testing.T) {
	provider := NewGeminiAPIProvider("test-key", "", "gemini-pro")
	provider.healthURL = "http://localhost:9999/nonexistent"
	provider.httpClient = &http.Client{Timeout: 100 * time.Millisecond}

	err := provider.HealthCheck()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "health check request failed")
}

func TestNewGeminiAPIProviderWithRetry(t *testing.T) {
	retryConfig := RetryConfig{
		MaxRetries:   5,
		InitialDelay: 2 * time.Second,
		MaxDelay:     60 * time.Second,
		Multiplier:   3.0,
	}

	provider := NewGeminiAPIProviderWithRetry("test-key", "", "", retryConfig)

	assert.Equal(t, "test-key", provider.apiKey)
	assert.Equal(t, GeminiDefaultModel, provider.model)
	assert.Equal(t, 5, provider.retryConfig.MaxRetries)
	assert.Equal(t, 2*time.Second, provider.retryConfig.InitialDelay)
}

func TestDefaultRetryConfig(t *testing.T) {
	config := DefaultRetryConfig()

	assert.Equal(t, 3, config.MaxRetries)
	assert.Equal(t, 1*time.Second, config.InitialDelay)
	assert.Equal(t, 30*time.Second, config.MaxDelay)
	assert.Equal(t, 2.0, config.Multiplier)
}

func TestGeminiAPIProvider_WaitWithJitter(t *testing.T) {
	provider := NewGeminiAPIProvider("test-key", "", "")

	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	baseDelay := 100 * time.Millisecond
	provider.waitWithJitter(ctx, baseDelay)
	elapsed := time.Since(start)

	// Lower bound only: load-robust, because extra CPU pressure can only
	// delay a timer, never fire it early.
	assert.GreaterOrEqual(t, elapsed, baseDelay)

	// Upper bound: deliberately NOT asserted in wall-clock time. Elapsed time
	// also contains scheduler latency this package does not control, so a
	// wall-clock ceiling measures the host at least as much as it measures
	// waitWithJitter. Reproduced under deliberate CPU load as
	// `"158.920597ms" is not less than or equal to "150ms"`.
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

func TestGeminiAPIProvider_WaitWithJitter_ContextCancelled(t *testing.T) {
	provider := NewGeminiAPIProvider("test-key", "", "")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	start := time.Now()
	provider.waitWithJitter(ctx, 1*time.Second)
	elapsed := time.Since(start)

	assert.Less(t, elapsed, 100*time.Millisecond)
}

// ==============================================================================
// Power Feature Tests
// ==============================================================================

func TestGeminiAPIProvider_ExtendedThinking(t *testing.T) {
	t.Run("pro model includes thinkingConfig", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var reqBody GeminiAPIRequest
			body, err := io.ReadAll(r.Body)
			require.NoError(t, err)
			err = json.Unmarshal(body, &reqBody)
			require.NoError(t, err)

			// Verify thinkingConfig is present for pro model
			require.NotNil(t, reqBody.GenerationConfig.ThinkingConfig,
				"thinkingConfig should be present for gemini-2.5-pro")
			assert.Equal(t, 8192, reqBody.GenerationConfig.ThinkingConfig.ThinkingBudget,
				"thinking budget should be 8192")

			response := GeminiResponse{
				Candidates: []GeminiCandidate{
					{
						Content: GeminiContent{
							Parts: []GeminiPart{{Text: "Thought-through response"}},
							Role:  "model",
						},
						FinishReason: "STOP",
					},
				},
				UsageMetadata: &GeminiUsageMetadata{TotalTokenCount: 50},
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(response)
		}))
		defer server.Close()

		provider := NewGeminiAPIProvider("test-key",
			server.URL+"/v1beta/models/%s:generateContent", "gemini-2.5-pro")
		provider.httpClient = server.Client()

		req := &models.LLMRequest{
			ID:     "thinking-test",
			Prompt: "Think deeply about this",
			ModelParams: models.ModelParameters{
				MaxTokens: 1000,
			},
		}
		resp, err := provider.Complete(context.Background(), req)
		require.NoError(t, err)
		require.NotNil(t, resp)
		assert.Equal(t, "Thought-through response", resp.Content)
	})

	t.Run("flash model does not include thinkingConfig", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var reqBody GeminiAPIRequest
			body, err := io.ReadAll(r.Body)
			require.NoError(t, err)
			err = json.Unmarshal(body, &reqBody)
			require.NoError(t, err)

			// Verify thinkingConfig is NOT present for flash model
			assert.Nil(t, reqBody.GenerationConfig.ThinkingConfig,
				"thinkingConfig should NOT be present for gemini-2.0-flash")

			response := GeminiResponse{
				Candidates: []GeminiCandidate{
					{
						Content: GeminiContent{
							Parts: []GeminiPart{{Text: "Quick response"}},
							Role:  "model",
						},
						FinishReason: "STOP",
					},
				},
				UsageMetadata: &GeminiUsageMetadata{TotalTokenCount: 20},
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(response)
		}))
		defer server.Close()

		provider := NewGeminiAPIProvider("test-key",
			server.URL+"/v1beta/models/%s:generateContent", "gemini-2.0-flash")
		provider.httpClient = server.Client()

		req := &models.LLMRequest{
			ID:     "no-thinking-test",
			Prompt: "Quick question",
			ModelParams: models.ModelParameters{
				MaxTokens: 500,
			},
		}
		resp, err := provider.Complete(context.Background(), req)
		require.NoError(t, err)
		require.NotNil(t, resp)
		assert.Equal(t, "Quick response", resp.Content)
	})

	t.Run("gemini-3-pro-preview includes thinkingConfig", func(t *testing.T) {
		provider := NewGeminiAPIProvider("test-key", "", "gemini-3-pro-preview")
		geminiReq := provider.convertRequest(&models.LLMRequest{
			Prompt:      "Test",
			ModelParams: models.ModelParameters{MaxTokens: 100},
		})
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig,
			"thinkingConfig should be present for gemini-3-pro-preview")
		assert.Equal(t, 8192, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})

	t.Run("gemini-3.1-pro-preview includes thinkingConfig", func(t *testing.T) {
		provider := NewGeminiAPIProvider("test-key", "", "gemini-3.1-pro-preview")
		geminiReq := provider.convertRequest(&models.LLMRequest{
			Prompt:      "Test",
			ModelParams: models.ModelParameters{MaxTokens: 100},
		})
		require.NotNil(t, geminiReq.GenerationConfig.ThinkingConfig,
			"thinkingConfig should be present for gemini-3.1-pro-preview")
		assert.Equal(t, 8192, geminiReq.GenerationConfig.ThinkingConfig.ThinkingBudget)
	})
}

func TestGeminiAPIProvider_GoogleSearchGrounding(t *testing.T) {
	t.Run("always includes googleSearch in tools", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var reqBody GeminiAPIRequest
			body, err := io.ReadAll(r.Body)
			require.NoError(t, err)
			err = json.Unmarshal(body, &reqBody)
			require.NoError(t, err)

			// Verify tools array has at least one entry with googleSearch
			require.GreaterOrEqual(t, len(reqBody.Tools), 1,
				"tools array should contain at least googleSearch")

			foundGoogleSearch := false
			for _, tool := range reqBody.Tools {
				if tool.GoogleSearch != nil {
					foundGoogleSearch = true
					break
				}
			}
			assert.True(t, foundGoogleSearch,
				"tools array should contain a tool with googleSearch")

			response := GeminiResponse{
				Candidates: []GeminiCandidate{
					{
						Content: GeminiContent{
							Parts: []GeminiPart{{Text: "Grounded response"}},
							Role:  "model",
						},
						FinishReason: "STOP",
					},
				},
				UsageMetadata: &GeminiUsageMetadata{TotalTokenCount: 25},
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(response)
		}))
		defer server.Close()

		provider := NewGeminiAPIProvider("test-key",
			server.URL+"/v1beta/models/%s:generateContent", "gemini-2.5-flash")
		provider.httpClient = server.Client()

		// Request with NO user tools
		req := &models.LLMRequest{
			ID:     "search-grounding-test",
			Prompt: "What is the current weather?",
			ModelParams: models.ModelParameters{
				MaxTokens: 500,
			},
		}
		resp, err := provider.Complete(context.Background(), req)
		require.NoError(t, err)
		require.NotNil(t, resp)
		assert.Equal(t, "Grounded response", resp.Content)
	})

	t.Run("googleSearch present alongside user tools", func(t *testing.T) {
		provider := NewGeminiAPIProvider("test-key", "", "gemini-2.5-flash")

		req := &models.LLMRequest{
			Prompt: "Test",
			Tools: []models.Tool{
				{
					Type: "function",
					Function: models.ToolFunction{
						Name:        "test_func",
						Description: "A test function",
					},
				},
			},
			ModelParams: models.ModelParameters{MaxTokens: 100},
		}

		geminiReq := provider.convertRequest(req)

		// Should have 2 tool entries: one with function declarations, one with googleSearch
		require.Equal(t, 2, len(geminiReq.Tools),
			"should have function declarations tool + googleSearch tool")

		assert.NotNil(t, geminiReq.Tools[0].FunctionDeclarations,
			"first tool entry should contain function declarations")
		assert.NotNil(t, geminiReq.Tools[1].GoogleSearch,
			"second tool entry should be googleSearch")
	})

	t.Run("googleSearch only when no user tools", func(t *testing.T) {
		provider := NewGeminiAPIProvider("test-key", "", "gemini-2.5-flash")

		req := &models.LLMRequest{
			Prompt:      "Test",
			ModelParams: models.ModelParameters{MaxTokens: 100},
		}

		geminiReq := provider.convertRequest(req)

		// Should have exactly 1 tool entry: googleSearch only
		require.Equal(t, 1, len(geminiReq.Tools),
			"should have only googleSearch tool when no user tools")
		assert.NotNil(t, geminiReq.Tools[0].GoogleSearch,
			"the single tool entry should be googleSearch")
		assert.Nil(t, geminiReq.Tools[0].FunctionDeclarations,
			"should not have function declarations")
	})
}

func TestGeminiAPIProvider_ModelAwareMaxTokens(t *testing.T) {
	tests := []struct {
		name              string
		model             string
		requestedTokens   int
		expectedMaxTokens int
	}{
		{
			name:              "gemini-2.5-pro gets extended cap",
			model:             "gemini-2.5-pro",
			requestedTokens:   65536,
			expectedMaxTokens: 65536,
		},
		{
			name:              "gemini-2.0-flash gets legacy cap",
			model:             "gemini-2.0-flash",
			requestedTokens:   65536,
			expectedMaxTokens: 8192,
		},
		{
			name:              "gemini-3-pro-preview gets extended cap",
			model:             "gemini-3-pro-preview",
			requestedTokens:   65536,
			expectedMaxTokens: 65536,
		},
		{
			name:              "gemini-2.5-flash gets extended cap",
			model:             "gemini-2.5-flash",
			requestedTokens:   65536,
			expectedMaxTokens: 65536,
		},
		{
			name:              "gemini-3.1-pro-preview gets extended cap",
			model:             "gemini-3.1-pro-preview",
			requestedTokens:   65536,
			expectedMaxTokens: 65536,
		},
		{
			name:              "zero requested returns default 4096",
			model:             "gemini-2.5-pro",
			requestedTokens:   0,
			expectedMaxTokens: 4096,
		},
		{
			name:              "small requested tokens are preserved",
			model:             "gemini-2.5-pro",
			requestedTokens:   1000,
			expectedMaxTokens: 1000,
		},
		{
			name:              "legacy model caps at 8192",
			model:             "gemini-2.0-flash",
			requestedTokens:   10000,
			expectedMaxTokens: 8192,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := NewGeminiAPIProvider("test-key", "", tt.model)
			geminiReq := provider.convertRequest(&models.LLMRequest{
				Prompt: "Test",
				ModelParams: models.ModelParameters{
					MaxTokens: tt.requestedTokens,
				},
			})
			assert.Equal(t, tt.expectedMaxTokens,
				geminiReq.GenerationConfig.MaxOutputTokens,
				"MaxOutputTokens mismatch for model %s with requested %d",
				tt.model, tt.requestedTokens)
		})
	}
}

func TestGeminiAPIProvider_ThinkingContentExtraction(t *testing.T) {
	t.Run("extracts thinking content into metadata", func(t *testing.T) {
		provider := NewGeminiAPIProvider("test-key", "", "gemini-2.5-pro")

		geminiResp := &GeminiResponse{
			Candidates: []GeminiCandidate{
				{
					Content: GeminiContent{
						Parts: []GeminiPart{
							{Text: "Let me think about this step by step...", Thought: true},
							{Text: "The answer is 42."},
						},
						Role: "model",
					},
					FinishReason: "STOP",
				},
			},
			UsageMetadata: &GeminiUsageMetadata{TotalTokenCount: 30},
		}

		req := &models.LLMRequest{ID: "thinking-extract"}
		resp := provider.convertResponse(req, geminiResp, time.Now())

		// Regular content should NOT include thinking parts
		assert.Equal(t, "The answer is 42.", resp.Content,
			"content should only contain non-thought parts")

		// Thinking content should be in metadata
		require.NotNil(t, resp.Metadata)
		thinkingVal, ok := resp.Metadata["thinking"]
		require.True(t, ok, "metadata should contain 'thinking' key")
		assert.Equal(t, "Let me think about this step by step...", thinkingVal,
			"thinking metadata should contain the thought text")
	})

	t.Run("no thinking metadata when no thought parts", func(t *testing.T) {
		provider := NewGeminiAPIProvider("test-key", "", "gemini-2.0-flash")

		geminiResp := &GeminiResponse{
			Candidates: []GeminiCandidate{
				{
					Content: GeminiContent{
						Parts: []GeminiPart{
							{Text: "Direct answer without thinking."},
						},
						Role: "model",
					},
					FinishReason: "STOP",
				},
			},
			UsageMetadata: &GeminiUsageMetadata{TotalTokenCount: 15},
		}

		req := &models.LLMRequest{ID: "no-thinking"}
		resp := provider.convertResponse(req, geminiResp, time.Now())

		assert.Equal(t, "Direct answer without thinking.", resp.Content)
		require.NotNil(t, resp.Metadata)
		_, hasThinking := resp.Metadata["thinking"]
		assert.False(t, hasThinking,
			"metadata should NOT contain 'thinking' key when no thought parts")
	})

	t.Run("multiple thinking parts concatenated", func(t *testing.T) {
		provider := NewGeminiAPIProvider("test-key", "", "gemini-2.5-pro")

		geminiResp := &GeminiResponse{
			Candidates: []GeminiCandidate{
				{
					Content: GeminiContent{
						Parts: []GeminiPart{
							{Text: "First thought. ", Thought: true},
							{Text: "Second thought. ", Thought: true},
							{Text: "Final answer."},
						},
						Role: "model",
					},
					FinishReason: "STOP",
				},
			},
			UsageMetadata: &GeminiUsageMetadata{TotalTokenCount: 40},
		}

		req := &models.LLMRequest{ID: "multi-thought"}
		resp := provider.convertResponse(req, geminiResp, time.Now())

		assert.Equal(t, "Final answer.", resp.Content)
		thinkingVal := resp.Metadata["thinking"]
		assert.Equal(t, "First thought. Second thought. ", thinkingVal,
			"multiple thinking parts should be concatenated")
	})
}

// ==============================================================================
// API Provider Benchmarks
// ==============================================================================

func BenchmarkGeminiAPIProvider_ConvertRequest(b *testing.B) {
	provider := NewGeminiAPIProvider("test-key", "", "")
	req := &models.LLMRequest{
		ID:     "bench-request",
		Prompt: "Test prompt",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi"},
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

func BenchmarkGeminiAPIProvider_CalculateConfidence(b *testing.B) {
	provider := NewGeminiAPIProvider("test-key", "", "")
	content := "This is a sample response from the Gemini model."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.calculateConfidence(content, "STOP")
	}
}
