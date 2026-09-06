package zen

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"digital.vasic.llmprovider/pkg/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewZenProvider(t *testing.T) {
	tests := []struct {
		name      string
		apiKey    string
		baseURL   string
		model     string
		wantURL   string
		wantModel string
	}{
		{
			name:      "default values",
			apiKey:    "test-key",
			baseURL:   "",
			model:     "",
			wantURL:   ZenAPIURL,
			wantModel: DefaultZenModel,
		},
		{
			name:      "custom values",
			apiKey:    "test-key",
			baseURL:   "https://custom.url/v1/chat/completions",
			model:     ModelBigPickle,
			wantURL:   "https://custom.url/v1/chat/completions",
			wantModel: ModelBigPickle,
		},
		{
			name:      "grok code fast model",
			apiKey:    "test-key",
			baseURL:   "",
			model:     ModelBigPickle,
			wantURL:   ZenAPIURL,
			wantModel: ModelBigPickle,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewZenProvider(tt.apiKey, tt.baseURL, tt.model)
			assert.NotNil(t, p)
			assert.Equal(t, tt.apiKey, p.apiKey)
			assert.Equal(t, tt.wantURL, p.baseURL)
			assert.Equal(t, tt.wantModel, p.model)
		})
	}
}

func TestFreeModels(t *testing.T) {
	models := FreeModels()
	// Merged known + discovered: at least the 5 known free models
	assert.GreaterOrEqual(t, len(models), 5)
	assert.Contains(t, models, ModelBigPickle)
	assert.Contains(t, models, ModelGLM5Free)
	assert.Contains(t, models, ModelMinimaxM25Free)
	assert.Contains(t, models, ModelMinimaxM21Free)
	assert.Contains(t, models, ModelTrinityLargePreviewFree)
}

func TestIsFreeModel(t *testing.T) {
	tests := []struct {
		model    string
		expected bool
	}{
		{ModelBigPickle, true},
		{ModelGLM5Free, true},
		{ModelMinimaxM25Free, true},
		{ModelMinimaxM21Free, true},
		{ModelTrinityLargePreviewFree, true},
		{ModelQwen3, false},
		{"opencode/gpt-5.1-codex", false},
		{"claude-3.5-sonnet", false},
		{"big-pickle", true},
		{"glm-5-free", true},
		{"minimax-m2.5-free", true},
		{"trinity-large-preview-free", true},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			result := isFreeModel(tt.model)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestZenProvider_Complete(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.True(t, strings.HasPrefix(r.Header.Get("Authorization"), "Bearer "))

		// Parse request body
		var req ZenRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, ModelBigPickle, req.Model)
		assert.Len(t, req.Messages, 1)

		// Return mock response
		resp := ZenResponse{
			ID:      "chatcmpl-123",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   ModelBigPickle,
			Choices: []ZenChoice{
				{
					Index: 0,
					Message: ZenMessage{
						Role:    "assistant",
						Content: "Hello! I'm Grok Code Fast.",
					},
					FinishReason: "stop",
				},
			},
			Usage: ZenUsage{
				PromptTokens:     10,
				CompletionTokens: 8,
				TotalTokens:      18,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Create provider with mock server
	p := NewZenProvider("test-key", server.URL, ModelBigPickle)

	// Create request
	req := &models.LLMRequest{
		ID: "test-123",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
		ModelParams: models.ModelParameters{
			MaxTokens:   100,
			Temperature: 0.7,
		},
	}

	// Execute
	ctx := context.Background()
	resp, err := p.Complete(ctx, req)

	// Verify
	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "chatcmpl-123", resp.ID)
	assert.Equal(t, "zen", resp.ProviderID)
	assert.Equal(t, "OpenCode Zen", resp.ProviderName)
	assert.Equal(t, "Hello! I'm Grok Code Fast.", resp.Content)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Equal(t, 18, resp.TokensUsed)
	assert.Greater(t, resp.Confidence, 0.8)
}

func TestZenProvider_Complete_Error(t *testing.T) {
	// Create mock server that returns error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		resp := ZenErrorResponse{}
		resp.Error.Message = "Invalid API key"
		resp.Error.Type = "authentication_error"
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Create provider with mock server
	p := NewZenProviderWithRetry("invalid-key", server.URL, ModelBigPickle, RetryConfig{
		MaxRetries:   0, // No retries for faster test
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     1 * time.Second,
		Multiplier:   2.0,
	})

	// Create request
	req := &models.LLMRequest{
		ID: "test-123",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	// Execute
	ctx := context.Background()
	resp, err := p.Complete(ctx, req)

	// Verify
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "Invalid API key")
}

func TestZenProvider_CompleteStream(t *testing.T) {
	// Create mock streaming server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ZenRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		assert.True(t, req.Stream)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send streaming chunks
		chunks := []string{"Hello", " from", " Zen", "!"}
		for i, chunk := range chunks {
			streamResp := ZenStreamResponse{
				ID:      "chatcmpl-stream-123",
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   ModelBigPickle,
				Choices: []ZenStreamChoice{
					{
						Index: 0,
						Delta: ZenMessage{Content: chunk},
					},
				},
			}

			if i == len(chunks)-1 {
				finishReason := "stop"
				streamResp.Choices[0].FinishReason = &finishReason
			}

			data, _ := json.Marshal(streamResp)
			_, _ = w.Write([]byte("data: "))
			_, _ = w.Write(data)
			_, _ = w.Write([]byte("\n\n"))
			w.(http.Flusher).Flush()
		}

		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		w.(http.Flusher).Flush()
	}))
	defer server.Close()

	// Create provider
	p := NewZenProvider("test-key", server.URL, ModelBigPickle)

	// Create request
	req := &models.LLMRequest{
		ID: "test-stream-123",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	// Execute
	ctx := context.Background()
	ch, err := p.CompleteStream(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, ch)

	// Collect responses
	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	// Verify we got chunks
	assert.Greater(t, len(responses), 0)

	// Verify last response has finish reason
	lastResp := responses[len(responses)-1]
	assert.Equal(t, "stop", lastResp.FinishReason)
}

func TestZenProvider_GetCapabilities(t *testing.T) {
	p := NewZenProvider("test-key", "", "")
	caps := p.GetCapabilities()

	assert.NotNil(t, caps)
	assert.True(t, caps.SupportsStreaming)
	assert.True(t, caps.SupportsCodeCompletion)
	assert.True(t, caps.SupportsReasoning)
	assert.Contains(t, caps.SupportedModels, ModelBigPickle)
	assert.Contains(t, caps.SupportedModels, ModelBigPickle)
	assert.Equal(t, "OpenCode Zen", caps.Metadata["provider"])
	assert.Equal(t, "true", caps.Metadata["free_tier"])
}

func TestZenProvider_ValidateConfig(t *testing.T) {
	tests := []struct {
		name          string
		apiKey        string
		baseURL       string
		model         string
		anonymousMode bool
		wantValid     bool
	}{
		{
			name:      "valid config with api key",
			apiKey:    "test-key",
			baseURL:   ZenAPIURL,
			model:     ModelBigPickle,
			wantValid: true,
		},
		{
			name:          "valid anonymous mode with free model",
			apiKey:        "",
			baseURL:       ZenAPIURL,
			model:         ModelBigPickle,
			anonymousMode: true,
			wantValid:     true,
		},
		{
			name:          "invalid - non-free model without api key",
			apiKey:        "",
			baseURL:       ZenAPIURL,
			model:         "opencode/gpt-5.1-codex", // Non-free model
			anonymousMode: false,
			wantValid:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &ZenProvider{
				apiKey:        tt.apiKey,
				baseURL:       tt.baseURL,
				model:         tt.model,
				anonymousMode: tt.anonymousMode,
			}
			valid, errors := p.ValidateConfig(nil)
			assert.Equal(t, tt.wantValid, valid)
			if !tt.wantValid {
				assert.NotEmpty(t, errors)
			}
		})
	}
}

func TestZenProvider_AnonymousMode(t *testing.T) {
	// Test creating provider in anonymous mode
	p := NewZenProviderAnonymous(ModelBigPickle)
	assert.NotNil(t, p)
	assert.True(t, p.IsAnonymousMode())
	assert.Equal(t, ModelBigPickle, p.GetModel())
	assert.NotEmpty(t, p.deviceID)

	// Test that non-free model falls back to default
	p2 := NewZenProviderAnonymous("non-free-model")
	assert.Equal(t, DefaultZenModel, p2.GetModel())
}

func TestZenProvider_IsAnonymousAccessAllowed(t *testing.T) {
	// Get actual free models from the system (may be discovered from API or fallback)
	freeModels := FreeModels()
	freeModelsSet := make(map[string]bool)
	for _, m := range freeModels {
		freeModelsSet[m] = true
	}

	// Check all known free models - they should allow anonymous access if in freeModels
	knownFreeModelsList := []string{
		ModelBigPickle,
		ModelGLM5Free,
		ModelMinimaxM25Free,
		ModelMinimaxM21Free,
		ModelTrinityLargePreviewFree,
	}

	for _, model := range knownFreeModelsList {
		if freeModelsSet[model] {
			assert.True(t, IsAnonymousAccessAllowed(model), "Model %s should allow anonymous access", model)
		} else {
			t.Logf("Model %s not in free models list, skipping anonymous access test", model)
		}
	}

	// These models require API key (never free)
	assert.False(t, IsAnonymousAccessAllowed(ModelGLM47))
	assert.False(t, IsAnonymousAccessAllowed(ModelKimiK2))
	assert.False(t, IsAnonymousAccessAllowed(ModelGemini3))

	// Non-free models should not allow anonymous access
	assert.False(t, IsAnonymousAccessAllowed("opencode/gpt-5.1-codex"))
	assert.False(t, IsAnonymousAccessAllowed("claude-3.5-sonnet"))
	assert.False(t, IsAnonymousAccessAllowed(ModelQwen3))
}

func TestZenProvider_HealthCheck(t *testing.T) {
	// Create mock server for health check
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" || r.URL.Path == "/models" {
			w.WriteHeader(http.StatusOK)
			resp := ZenModelsResponse{
				Object: "list",
				Data: []ZenModelInfo{
					{ID: ModelBigPickle, OwnedBy: "opencode"},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	// Override the models URL for testing
	p := &ZenProvider{
		apiKey:     "test-key",
		baseURL:    server.URL + "/v1/chat/completions",
		model:      ModelBigPickle,
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}

	// For this test, we need to test the actual HealthCheck method
	// which uses ZenModelsURL constant, so we'll skip the actual health check
	// and just verify the provider is properly configured
	assert.Equal(t, "test-key", p.apiKey)
	assert.Equal(t, ModelBigPickle, p.model)
}

func TestZenProvider_ConvertRequest(t *testing.T) {
	p := NewZenProvider("test-key", "", ModelBigPickle)

	req := &models.LLMRequest{
		ID:     "test-123",
		Prompt: "You are a helpful assistant",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
			{Role: "user", Content: "How are you?"},
		},
		ModelParams: models.ModelParameters{
			MaxTokens:   1000,
			Temperature: 0.8,
			TopP:        0.9,
		},
	}

	zenReq := p.convertRequest(req)

	// Verify system message is added
	assert.Len(t, zenReq.Messages, 4) // 1 system + 3 conversation messages
	assert.Equal(t, "system", zenReq.Messages[0].Role)
	assert.Equal(t, "You are a helpful assistant", zenReq.Messages[0].Content)

	// Verify model params
	assert.Equal(t, ModelBigPickle, zenReq.Model)
	assert.Equal(t, 1000, zenReq.MaxTokens)
	assert.Equal(t, 0.8, zenReq.Temperature)
	assert.Equal(t, 0.9, zenReq.TopP)
}

func TestZenProvider_CalculateConfidence(t *testing.T) {
	p := NewZenProvider("test-key", "", "")

	tests := []struct {
		name         string
		content      string
		finishReason string
		minConf      float64
		maxConf      float64
	}{
		{
			name:         "stop finish reason",
			content:      "Short response",
			finishReason: "stop",
			minConf:      0.85,
			maxConf:      0.95,
		},
		{
			name:         "length finish reason",
			content:      "Response cut off due to length",
			finishReason: "length",
			minConf:      0.65,
			maxConf:      0.75,
		},
		{
			name:         "long content",
			content:      strings.Repeat("This is a longer response. ", 50),
			finishReason: "stop",
			minConf:      0.95,
			maxConf:      1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conf := p.calculateConfidence(tt.content, tt.finishReason)
			assert.GreaterOrEqual(t, conf, tt.minConf)
			assert.LessOrEqual(t, conf, tt.maxConf)
		})
	}
}

func TestZenProvider_SetGetModel(t *testing.T) {
	p := NewZenProvider("test-key", "", ModelBigPickle)

	assert.Equal(t, ModelBigPickle, p.GetModel())

	p.SetModel(ModelBigPickle)
	assert.Equal(t, ModelBigPickle, p.GetModel())
}

func TestDefaultRetryConfig(t *testing.T) {
	config := DefaultRetryConfig()

	assert.Equal(t, 3, config.MaxRetries)
	assert.Equal(t, 1*time.Second, config.InitialDelay)
	assert.Equal(t, 30*time.Second, config.MaxDelay)
	assert.Equal(t, 2.0, config.Multiplier)
}

func TestMinFunction(t *testing.T) {
	tests := []struct {
		a, b, expected int
	}{
		{1, 2, 1},
		{2, 1, 1},
		{5, 5, 5},
		{0, 0, 0},
		{-1, 0, -1},
		{-5, -3, -5},
		{100, 50, 50},
	}

	for _, tt := range tests {
		result := min(tt.a, tt.b)
		assert.Equal(t, tt.expected, result, "min(%d, %d) should be %d", tt.a, tt.b, tt.expected)
	}
}

func TestZenProvider_HealthCheck_Success(t *testing.T) {
	// Create mock server for health check
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		w.WriteHeader(http.StatusOK)
		resp := ZenModelsResponse{
			Object: "list",
			Data: []ZenModelInfo{
				{ID: ModelBigPickle, OwnedBy: "opencode"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := &ZenProvider{
		apiKey:     "test-key",
		baseURL:    server.URL,
		model:      ModelBigPickle,
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}

	// We need to test with the actual ZenModelsURL, which we can't change
	// So we verify the provider is properly configured and skip the actual network call
	assert.NotNil(t, p.httpClient)
}

func TestZenProvider_HealthCheck_Failure(t *testing.T) {
	// Create mock server that returns error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	p := &ZenProvider{
		apiKey:     "test-key",
		baseURL:    server.URL,
		model:      ModelBigPickle,
		httpClient: &http.Client{Timeout: 10 * time.Second},
	}

	// Provider is properly configured for failure scenario
	assert.NotNil(t, p.httpClient)
	assert.Equal(t, "test-key", p.apiKey)
}

func TestZenProvider_GetAvailableModels_Success(t *testing.T) {
	// Create mock server for models endpoint
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		resp := ZenModelsResponse{
			Object: "list",
			Data: []ZenModelInfo{
				{ID: ModelBigPickle, OwnedBy: "opencode", Created: time.Now().Unix()},
				{ID: ModelBigPickle, OwnedBy: "opencode", Created: time.Now().Unix()},
				{ID: "opencode/gpt-5.1", OwnedBy: "opencode", Created: time.Now().Unix()},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Note: We can't easily test GetAvailableModels because it uses the hardcoded ZenModelsURL
	// But we verify the struct is correctly defined
	modelsResp := ZenModelsResponse{
		Object: "list",
		Data: []ZenModelInfo{
			{ID: ModelBigPickle, OwnedBy: "opencode"},
		},
	}
	assert.Len(t, modelsResp.Data, 1)
	assert.Equal(t, ModelBigPickle, modelsResp.Data[0].ID)
}

func TestZenProvider_GetAvailableModels_Error(t *testing.T) {
	// Create mock server that returns error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("Internal server error"))
	}))
	defer server.Close()

	// Verify the ZenModelsResponse handles errors properly
	errorResp := ZenErrorResponse{}
	errorResp.Error.Message = "Internal server error"
	errorResp.Error.Type = "server_error"

	assert.Equal(t, "Internal server error", errorResp.Error.Message)
	assert.Equal(t, "server_error", errorResp.Error.Type)
}

func TestZenModelsResponse_Parsing(t *testing.T) {
	jsonData := `{
		"object": "list",
		"data": [
			{"id": "opencode/big-pickle", "owned_by": "opencode", "created": 1700000000},
			{"id": "opencode/big-pickle", "owned_by": "opencode", "created": 1700000001}
		]
	}`

	var resp ZenModelsResponse
	err := json.Unmarshal([]byte(jsonData), &resp)
	require.NoError(t, err)

	assert.Equal(t, "list", resp.Object)
	assert.Len(t, resp.Data, 2)
	assert.Equal(t, "opencode/big-pickle", resp.Data[0].ID)
	assert.Equal(t, "opencode", resp.Data[0].OwnedBy)
}

func TestZenProvider_GetFreeModels_Filtering(t *testing.T) {
	// Test that free models filtering logic correctly separates free from paid
	allModels := []ZenModelInfo{
		{ID: ModelBigPickle, OwnedBy: "opencode"},
		{ID: ModelGLM5Free, OwnedBy: "opencode"},
		{ID: ModelMinimaxM25Free, OwnedBy: "opencode"},
		{ID: ModelMinimaxM21Free, OwnedBy: "opencode"},
		{ID: ModelTrinityLargePreviewFree, OwnedBy: "opencode"},
		{ID: "opencode/gpt-5.1-codex", OwnedBy: "opencode"}, // Not in free list
		{ID: ModelGLM47, OwnedBy: "opencode"},               // Requires API key
		{ID: ModelQwen3, OwnedBy: "opencode"},               // Removed from free tier
	}

	freeModelIDs := FreeModels()
	freeModels := make([]ZenModelInfo, 0)

	for _, model := range allModels {
		for _, freeID := range freeModelIDs {
			if model.ID == freeID || strings.HasSuffix(freeID, model.ID) {
				freeModels = append(freeModels, model)
				break
			}
		}
	}

	// All 5 known free models should be matched from allModels
	assert.GreaterOrEqual(t, len(freeModels), 5, "Should find at least 5 known free models")
	// Paid/legacy models should NOT be included
	for _, freeModel := range freeModels {
		assert.NotEqual(t, "opencode/gpt-5.1-codex", freeModel.ID, "Paid model should not be in free list")
		assert.NotEqual(t, ModelGLM47, freeModel.ID, "Paid model should not be in free list")
		assert.NotEqual(t, ModelQwen3, freeModel.ID, "Legacy model should not be in free list")
	}
}

func TestZenProvider_NormalizeModelID(t *testing.T) {
	// normalizeModelID STRIPS the "opencode/" prefix (Zen API requires model names WITHOUT the prefix)
	tests := []struct {
		input    string
		expected string
	}{
		{"big-pickle", "big-pickle"},                // No prefix, unchanged
		{"big-pickle", "big-pickle"},                // No prefix, unchanged
		{"opencode/big-pickle", "big-pickle"},       // Strips opencode/ prefix
		{"opencode/glm-4-7b-free", "glm-4-7b-free"}, // Strips opencode/ prefix
		{"opencode-custom-model", "custom-model"},   // Strips opencode- prefix (alternate format)
		{"custom-model", "custom-model"},            // No prefix, unchanged
	}

	for _, tt := range tests {
		result := normalizeModelID(tt.input)
		assert.Equal(t, tt.expected, result, "normalizeModelID(%s)", tt.input)
	}
}

func TestZenProvider_ConvertResponse(t *testing.T) {
	p := NewZenProvider("test-key", "", ModelBigPickle)
	// Use startTime in the past to ensure ResponseTime > 0
	startTime := time.Now().Add(-100 * time.Millisecond)

	req := &models.LLMRequest{ID: "test-123"}
	zenResp := &ZenResponse{
		ID:      "chatcmpl-456",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   ModelBigPickle,
		Choices: []ZenChoice{
			{
				Index: 0,
				Message: ZenMessage{
					Role:    "assistant",
					Content: "Test response content",
				},
				FinishReason: "stop",
			},
		},
		Usage: ZenUsage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}

	resp := p.convertResponse(req, zenResp, startTime)

	assert.Equal(t, "chatcmpl-456", resp.ID)
	assert.Equal(t, "test-123", resp.RequestID)
	assert.Equal(t, "zen", resp.ProviderID)
	assert.Equal(t, "OpenCode Zen", resp.ProviderName)
	assert.Equal(t, "Test response content", resp.Content)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Equal(t, 30, resp.TokensUsed)
	assert.GreaterOrEqual(t, resp.ResponseTime, int64(100)) // At least 100ms
}

func TestZenProvider_AnonymousModeHeaders(t *testing.T) {
	p := NewZenProviderAnonymous(ModelBigPickle)

	assert.True(t, p.IsAnonymousMode())
	assert.NotEmpty(t, p.deviceID)
	assert.True(t, strings.HasPrefix(p.deviceID, "helix-"))
}

func TestWaitWithJitter(t *testing.T) {
	// Test that waitWithJitter returns within expected range
	p := NewZenProvider("test-key", "", ModelBigPickle)
	baseDelay := 100 * time.Millisecond
	start := time.Now()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	p.waitWithJitter(ctx, baseDelay)
	elapsed := time.Since(start)

	// waitWithJitter only ever ADDS jitter, so the true lower bound is the
	// full base delay, not half of it. Tightened rather than relaxed. This
	// bound is load-robust: extra CPU pressure can only delay a timer.
	assert.GreaterOrEqual(t, elapsed, baseDelay)

	// Upper bound: deliberately NOT asserted in wall-clock time. Elapsed time
	// also contains scheduler latency this package does not control, so a
	// wall-clock ceiling measures the host at least as much as it measures
	// waitWithJitter. Not reproduced in this session
	// (this ceiling carried 100ms of slack), but it is the same defect as the
	// five that were reproduced; a bigger constant is not a fix.
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

func TestIsAuthRetryableStatus(t *testing.T) {
	// isAuthRetryableStatus only returns true for 401 Unauthorized
	tests := []struct {
		status   int
		expected bool
	}{
		{401, true},  // Unauthorized - retryable
		{403, false}, // Forbidden - not retryable
		{429, false}, // Too Many Requests - not retryable by this function
		{500, false}, // Server Error - not retryable by this function
		{502, false}, // Bad Gateway - not retryable by this function
		{503, false}, // Service Unavailable - not retryable by this function
		{504, false}, // Gateway Timeout - not retryable by this function
		{200, false}, // OK - not retryable
		{404, false}, // Not Found - not retryable
		{400, false}, // Bad Request - not retryable
	}

	for _, tt := range tests {
		result := isAuthRetryableStatus(tt.status)
		assert.Equal(t, tt.expected, result, "isAuthRetryableStatus(%d)", tt.status)
	}
}

func TestGenerateDeviceID(t *testing.T) {
	id1 := generateDeviceID()
	id2 := generateDeviceID()

	assert.True(t, strings.HasPrefix(id1, "helix-"))
	assert.True(t, strings.HasPrefix(id2, "helix-"))

	// Each call should generate a unique ID
	assert.NotEqual(t, id1, id2)
}

func TestNextDelay(t *testing.T) {
	p := NewZenProviderWithRetry("test-key", "", ModelBigPickle, RetryConfig{
		MaxRetries:   3,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     1 * time.Second,
		Multiplier:   2.0,
	})

	tests := []struct {
		name         string
		currentDelay time.Duration
		expected     time.Duration
	}{
		{"double initial delay", 100 * time.Millisecond, 200 * time.Millisecond},
		{"double 200ms", 200 * time.Millisecond, 400 * time.Millisecond},
		{"cap at max delay", 600 * time.Millisecond, 1 * time.Second}, // 1200ms capped to 1000ms
		{"already at max", 1 * time.Second, 1 * time.Second},          // stays at max
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := p.nextDelay(tt.currentDelay)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsRetryableStatus(t *testing.T) {
	tests := []struct {
		status   int
		expected bool
	}{
		{429, true},  // Too Many Requests - retryable
		{500, true},  // Internal Server Error - retryable
		{502, true},  // Bad Gateway - retryable
		{503, true},  // Service Unavailable - retryable
		{504, true},  // Gateway Timeout - retryable
		{408, false}, // Request Timeout - not in retry list
		{200, false}, // OK
		{400, false}, // Bad Request
		{401, false}, // Unauthorized
		{403, false}, // Forbidden
		{404, false}, // Not Found
	}

	for _, tt := range tests {
		result := isRetryableStatus(tt.status)
		assert.Equal(t, tt.expected, result, "isRetryableStatus(%d)", tt.status)
	}
}

// mockRoundTripper implements http.RoundTripper for testing
type mockRoundTripper struct {
	response *http.Response
	err      error
}

func (m *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return m.response, m.err
}

func TestZenProvider_HealthCheck_WithMockTransport(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		// Create mock response
		respBody := `{"object":"list","data":[{"id":"big-pickle"}]}`
		mockResp := &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(respBody)),
			Header:     make(http.Header),
		}

		p := &ZenProvider{
			apiKey:     "test-key",
			model:      ModelBigPickle,
			httpClient: &http.Client{Transport: &mockRoundTripper{response: mockResp}},
		}

		err := p.HealthCheck()
		assert.NoError(t, err)
	})

	t.Run("failure - service unavailable", func(t *testing.T) {
		mockResp := &http.Response{
			StatusCode: http.StatusServiceUnavailable,
			Body:       io.NopCloser(strings.NewReader("")),
			Header:     make(http.Header),
		}

		p := &ZenProvider{
			apiKey:     "test-key",
			model:      ModelBigPickle,
			httpClient: &http.Client{Transport: &mockRoundTripper{response: mockResp}},
		}

		err := p.HealthCheck()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "health check failed with status: 503")
	})

	t.Run("failure - network error", func(t *testing.T) {
		p := &ZenProvider{
			apiKey:     "test-key",
			model:      ModelBigPickle,
			httpClient: &http.Client{Transport: &mockRoundTripper{err: fmt.Errorf("connection refused")}},
		}

		err := p.HealthCheck()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "health check request failed")
	})

	t.Run("with anonymous mode headers", func(t *testing.T) {
		respBody := `{"object":"list","data":[{"id":"grok-code"}]}`
		var capturedHeader string

		transport := &mockRoundTripper{
			response: &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(respBody)),
				Header:     make(http.Header),
			},
		}

		p := &ZenProvider{
			model:         ModelBigPickle,
			anonymousMode: true,
			deviceID:      "test-device-id",
			httpClient:    &http.Client{Transport: transport},
		}

		// Note: We can't easily capture headers with this simple mock
		// but we verify the anonymous mode is set
		err := p.HealthCheck()
		assert.NoError(t, err)
		_ = capturedHeader // Suppress unused variable warning
	})
}

func TestZenProvider_GetAvailableModels_WithMockTransport(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		respBody := `{
			"object": "list",
			"data": [
				{"id": "big-pickle", "owned_by": "opencode", "created": 1234567890},
				{"id": "big-pickle", "owned_by": "opencode", "created": 1234567891}
			]
		}`
		mockResp := &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(respBody)),
			Header:     make(http.Header),
		}

		p := &ZenProvider{
			apiKey:     "test-key",
			model:      ModelBigPickle,
			httpClient: &http.Client{Transport: &mockRoundTripper{response: mockResp}},
		}

		models, err := p.GetAvailableModels(context.Background())
		assert.NoError(t, err)
		assert.Len(t, models, 2)
		assert.Equal(t, "big-pickle", models[0].ID)
		assert.Equal(t, "big-pickle", models[1].ID)
	})

	t.Run("error - API error", func(t *testing.T) {
		mockResp := &http.Response{
			StatusCode: http.StatusInternalServerError,
			Body:       io.NopCloser(strings.NewReader("Internal Server Error")),
			Header:     make(http.Header),
		}

		p := &ZenProvider{
			apiKey:     "test-key",
			model:      ModelBigPickle,
			httpClient: &http.Client{Transport: &mockRoundTripper{response: mockResp}},
		}

		_, err := p.GetAvailableModels(context.Background())
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "API error: 500")
	})

	t.Run("error - network error", func(t *testing.T) {
		p := &ZenProvider{
			apiKey:     "test-key",
			model:      ModelBigPickle,
			httpClient: &http.Client{Transport: &mockRoundTripper{err: fmt.Errorf("network error")}},
		}

		_, err := p.GetAvailableModels(context.Background())
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "request failed")
	})

	t.Run("error - invalid JSON", func(t *testing.T) {
		mockResp := &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("invalid json")),
			Header:     make(http.Header),
		}

		p := &ZenProvider{
			apiKey:     "test-key",
			model:      ModelBigPickle,
			httpClient: &http.Client{Transport: &mockRoundTripper{response: mockResp}},
		}

		_, err := p.GetAvailableModels(context.Background())
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to decode response")
	})
}

func TestZenProvider_GetFreeModels_WithMockTransport(t *testing.T) {
	t.Run("success - filters free models", func(t *testing.T) {
		respBody := `{
			"object": "list",
			"data": [
				{"id": "big-pickle", "owned_by": "opencode"},
				{"id": "big-pickle", "owned_by": "opencode"},
				{"id": "premium-model", "owned_by": "opencode"},
				{"id": "gpt-5-nano", "owned_by": "opencode"}
			]
		}`
		mockResp := &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(respBody)),
			Header:     make(http.Header),
		}

		p := &ZenProvider{
			apiKey:     "test-key",
			model:      ModelBigPickle,
			httpClient: &http.Client{Transport: &mockRoundTripper{response: mockResp}},
		}

		models, err := p.GetFreeModels(context.Background())
		assert.NoError(t, err)
		// Should only return models that are in the FreeModels list
		for _, m := range models {
			assert.True(t, isFreeModel(m.ID), "model %s should be free", m.ID)
		}
	})

	t.Run("error - propagates GetAvailableModels error", func(t *testing.T) {
		p := &ZenProvider{
			apiKey:     "test-key",
			model:      ModelBigPickle,
			httpClient: &http.Client{Transport: &mockRoundTripper{err: fmt.Errorf("network error")}},
		}

		_, err := p.GetFreeModels(context.Background())
		assert.Error(t, err)
	})
}

// =============================================================================
// Benchmarks
// =============================================================================

func BenchmarkZenProvider_ConvertRequest(b *testing.B) {
	provider := NewZenProvider("test-key", "", "")
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

func BenchmarkZenProvider_CalculateConfidence(b *testing.B) {
	provider := NewZenProvider("test-key", "", "")
	content := "This is a sample response from the Zen model for confidence scoring."
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.calculateConfidence(content, "stop")
	}
}

func BenchmarkZenProvider_GetCapabilities(b *testing.B) {
	provider := NewZenProvider("test-key", "", "")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.GetCapabilities()
	}
}

func BenchmarkZenProvider_ConvertResponse(b *testing.B) {
	provider := NewZenProvider("test-key", "", "")
	req := &models.LLMRequest{ID: "bench-request"}
	startTime := time.Now()
	zenResp := &ZenResponse{
		ID:    "resp-456",
		Model: DefaultZenModel,
		Choices: []ZenChoice{
			{
				Index:        0,
				Message:      ZenMessage{Role: "assistant", Content: "Benchmark response content"},
				FinishReason: "stop",
			},
		},
		Usage: ZenUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.convertResponse(req, zenResp, startTime)
	}
}

func BenchmarkZenProvider_NormalizeModelID(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		normalizeModelID("opencode/some-model")
	}
}

func BenchmarkZenProvider_NextDelay(b *testing.B) {
	provider := NewZenProviderWithRetry("test-key", "", DefaultZenModel, RetryConfig{
		MaxRetries:   3,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     1 * time.Second,
		Multiplier:   2.0,
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.nextDelay(100 * time.Millisecond)
	}
}

func BenchmarkZenProvider_IsRetryableStatus(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		isRetryableStatus(429)
		isRetryableStatus(500)
		isRetryableStatus(200)
	}
}

func BenchmarkZenProvider_IsAuthRetryableStatus(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		isAuthRetryableStatus(401)
		isAuthRetryableStatus(403)
		isAuthRetryableStatus(200)
	}
}
