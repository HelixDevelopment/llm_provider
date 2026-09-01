package ai21

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"digital.vasic.llmprovider/pkg/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewProvider(t *testing.T) {
	provider := NewProvider("test-api-key", "", "")
	assert.NotNil(t, provider)
	assert.Equal(t, "test-api-key", provider.apiKey)
	assert.Equal(t, AI21APIURL, provider.baseURL)
	assert.Equal(t, DefaultModel, provider.model)
}

func TestNewProviderWithCustomURL(t *testing.T) {
	customURL := "https://custom.ai21.com/studio/v1/chat/completions"
	provider := NewProvider("test-api-key", customURL, "jamba-1.5-mini")
	assert.Equal(t, customURL, provider.baseURL)
	assert.Equal(t, "jamba-1.5-mini", provider.model)
}

func TestNewProviderWithRetry(t *testing.T) {
	retryConfig := RetryConfig{
		MaxRetries:   5,
		InitialDelay: 2 * time.Second,
		MaxDelay:     60 * time.Second,
		Multiplier:   3.0,
	}
	provider := NewProviderWithRetry("test-key", "", "jamba-instruct", retryConfig)
	assert.Equal(t, 5, provider.retryConfig.MaxRetries)
	assert.Equal(t, 2*time.Second, provider.retryConfig.InitialDelay)
	assert.Equal(t, "jamba-instruct", provider.model)
}

func TestDefaultRetryConfig(t *testing.T) {
	config := DefaultRetryConfig()
	assert.Equal(t, 3, config.MaxRetries)
	assert.Equal(t, time.Second, config.InitialDelay)
	assert.Equal(t, 30*time.Second, config.MaxDelay)
	assert.Equal(t, 2.0, config.Multiplier)
}

func TestComplete(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer ")

		var req Request
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, "jamba-1.5-large", req.Model)

		resp := Response{
			ID:      "chatcmpl-123",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "jamba-1.5-large",
			Choices: []Choice{
				{
					Index:        0,
					Message:      Message{Role: "assistant", Content: "Hello! I'm Jamba, how can I help?"},
					FinishReason: "stop",
				},
			},
			Usage: Usage{
				PromptTokens:     15,
				CompletionTokens: 10,
				TotalTokens:      25,
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewProvider("test-api-key", server.URL, "jamba-1.5-large")
	req := &models.LLMRequest{
		ID:     "req-1",
		Prompt: "You are a helpful assistant.",
		Messages: []models.Message{
			{Role: "user", Content: "Hello!"},
		},
		ModelParams: models.ModelParameters{
			Temperature: 0.7,
			MaxTokens:   1000,
		},
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "chatcmpl-123", resp.ID)
	assert.Contains(t, resp.Content, "Jamba")
	assert.Equal(t, "ai21", resp.ProviderID)
	assert.Equal(t, "AI21 Labs", resp.ProviderName)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Equal(t, 25, resp.TokensUsed)
}

func TestCompleteWithTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req Request
		_ = json.NewDecoder(r.Body).Decode(&req)
		assert.Len(t, req.Tools, 1)
		assert.Equal(t, "function", req.Tools[0].Type)
		assert.Equal(t, "calculate", req.Tools[0].Function.Name)

		resp := Response{
			ID:    "chatcmpl-tools",
			Model: "jamba-1.5-large",
			Choices: []Choice{
				{
					Index: 0,
					Message: Message{
						Role: "assistant",
						ToolCalls: []ToolCall{
							{
								ID:   "call-1",
								Type: "function",
								Function: FunctionCall{
									Name:      "calculate",
									Arguments: `{"expression": "2+2"}`,
								},
							},
						},
					},
					FinishReason: "tool_calls",
				},
			},
			Usage: Usage{TotalTokens: 30},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewProvider("test-api-key", server.URL, "")
	req := &models.LLMRequest{
		ID: "req-tools",
		Messages: []models.Message{
			{Role: "user", Content: "Calculate 2+2"},
		},
		ModelParams: models.ModelParameters{
			Temperature: 0.5,
		},
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.ToolFunction{
					Name:        "calculate",
					Description: "Calculate math expressions",
					Parameters:  map[string]any{"type": "object"},
				},
			},
		},
		ToolChoice: "auto",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, "call-1", resp.ToolCalls[0].ID)
	assert.Equal(t, "calculate", resp.ToolCalls[0].Function.Name)
	assert.Equal(t, "tool_calls", resp.FinishReason)
}

func TestCompleteAPIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"detail": "Invalid API key"}`))
	}))
	defer server.Close()

	provider := NewProviderWithRetry("invalid-key", server.URL, "", RetryConfig{MaxRetries: 0})
	req := &models.LLMRequest{
		ID:       "req-error",
		Messages: []models.Message{{Role: "user", Content: "Test"}},
	}

	_, err := provider.Complete(context.Background(), req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "401")
}

func TestCompleteStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req Request
		_ = json.NewDecoder(r.Body).Decode(&req)
		assert.True(t, req.Stream)

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		events := []string{
			`data: {"id":"chunk-1","choices":[{"delta":{"content":"AI21"}}]}`,
			`data: {"id":"chunk-2","choices":[{"delta":{"content":" Labs"}}]}`,
			`data: {"id":"chunk-3","choices":[{"delta":{"content":" response"}}]}`,
			`data: [DONE]`,
		}

		for _, event := range events {
			_, _ = w.Write([]byte(event + "\n\n"))
			flusher.Flush()
		}
	}))
	defer server.Close()

	provider := NewProvider("test-api-key", server.URL, "")
	req := &models.LLMRequest{
		ID:       "req-stream",
		Messages: []models.Message{{Role: "user", Content: "Say hello"}},
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	require.GreaterOrEqual(t, len(responses), 3)
	lastResp := responses[len(responses)-1]
	assert.Equal(t, "AI21 Labs response", lastResp.Content)
	assert.Equal(t, "stop", lastResp.FinishReason)
}

func TestCompleteStreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error": "Service unavailable"}`))
	}))
	defer server.Close()

	provider := NewProviderWithRetry("test-key", server.URL, "", RetryConfig{MaxRetries: 0})
	req := &models.LLMRequest{
		ID:       "req-stream-error",
		Messages: []models.Message{{Role: "user", Content: "Test"}},
	}

	_, err := provider.CompleteStream(context.Background(), req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "503")
}

// modelsFixture serves an OpenAI-compatible /models response, which is the
// shape pkg/discovery parses when no custom ResponseParser is configured. The
// returned URL is a chat-completions base URL, so the provider derives its
// models endpoint from it exactly as it does in production.
func modelsFixture(t *testing.T, ids ...string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/models") {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		// The adapter must authenticate discovery, not just completions.
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		data := make([]map[string]any, 0, len(ids))
		for _, id := range ids {
			data = append(data, map[string]any{"id": id, "object": "model"})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestGetCapabilities pins the ADAPTER'S BEHAVIOUR against a controlled
// fixture.
//
// It used to construct a provider with a synthetic key and assert
// `len(SupportedModels) >= 1` against LIVE discovery. That is not a unit test:
// it asserts that a third party is up and that whoever runs it holds a valid
// AI21 credential, so it fails on any machine without both — which is how it
// was found, red on a pristine checkout. Worse, the thing it demanded is the
// OPPOSITE of the documented contract: pkg/discovery returns nil when live
// discovery is unreachable (CONST-036, no hardcoded fallback), so the old
// assertion required a correct implementation to fail.
func TestGetCapabilities(t *testing.T) {
	srv := modelsFixture(t, "jamba-1.5-large", "jamba-1.5-mini", "text-embedding-ada-002")
	provider := NewProvider("test-api-key", srv.URL+"/studio/v1/chat/completions", "")

	caps := provider.GetCapabilities()

	require.NotNil(t, caps)
	// Discovery reached the fixture and its result was plumbed into the
	// capabilities — the behaviour the old assertion was reaching for.
	assert.Contains(t, caps.SupportedModels, "jamba-1.5-large")
	assert.Contains(t, caps.SupportedModels, "jamba-1.5-mini")
	// And the chat-model filter still applies to whatever the endpoint returns.
	assert.NotContains(t, caps.SupportedModels, "text-embedding-ada-002")

	// The static half of the capability record, which was always deterministic
	// and never needed a network at all.
	assert.Contains(t, caps.SupportedFeatures, "chat")
	assert.Contains(t, caps.SupportedFeatures, "streaming")
	assert.Contains(t, caps.SupportedFeatures, "tools")
	assert.True(t, caps.SupportsStreaming)
	assert.True(t, caps.SupportsTools)
	assert.True(t, caps.SupportsFunctionCalling)
	assert.Equal(t, 256000, caps.Limits.MaxTokens)
	assert.Equal(t, "ai21", caps.Metadata["provider"])
}

// TestGetCapabilitiesDiscoveryUnavailable is the honest-unavailability half.
//
// When discovery cannot answer, SupportedModels MUST be empty — never the
// deprecated FallbackModels catalogue this provider still passes to the
// discoverer. Serving that list would hand a caller model IDs it may not be
// able to invoke, which CONST-036 calls a structural bluff. The names are
// asserted individually because "empty" alone would also pass if the list
// leaked through under a different key.
func TestGetCapabilitiesDiscoveryUnavailable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	provider := NewProvider("test-api-key", srv.URL+"/studio/v1/chat/completions", "")
	caps := provider.GetCapabilities()

	require.NotNil(t, caps)
	assert.Empty(t, caps.SupportedModels,
		"unreachable discovery must report NO models, not a stale hardcoded catalogue (CONST-036)")
	for _, fallback := range []string{"jamba-1.5-large", "jamba-instruct", "j2-ultra"} {
		assert.NotContains(t, caps.SupportedModels, fallback,
			"the deprecated FallbackModels list must never reach a caller")
	}
	// The rest of the record is still served: unavailable models do not make
	// the provider's fixed capabilities unknown.
	assert.True(t, caps.SupportsStreaming)
	assert.Equal(t, "ai21", caps.Metadata["provider"])
}

// TestModelsURLMatchesConstant guards the refactor that made the discovery
// endpoint config-injectable: with the DEFAULT base URL it must still resolve
// to the production constant, byte for byte. Without this, deriving the
// endpoint could silently change where production traffic goes.
func TestModelsURLMatchesConstant(t *testing.T) {
	assert.Equal(t, AI21ModelsURL, NewProvider("k", "", "").modelsURL())
}

// TestGetCapabilitiesLiveDiscovery is the live probe, kept but made OPT-IN.
//
// It is genuinely useful — it is the only thing here that would notice AI21
// changing its models endpoint — but it depends on a third party and a real
// credential, so it must never be part of the default suite. It SKIPS with a
// stated reason rather than failing, because "nobody exported a key" is not a
// defect in this adapter.
//
//	AI21_LIVE_DISCOVERY_TEST=1 AI21_API_KEY=<real key> go test ./pkg/providers/ai21/
func TestGetCapabilitiesLiveDiscovery(t *testing.T) {
	if os.Getenv("AI21_LIVE_DISCOVERY_TEST") == "" {
		t.Skip("live discovery probe is opt-in: set AI21_LIVE_DISCOVERY_TEST=1 " +
			"together with a real AI21_API_KEY to run it")
	}
	apiKey := os.Getenv("AI21_API_KEY")
	if apiKey == "" {
		t.Skip("AI21_LIVE_DISCOVERY_TEST is set but AI21_API_KEY is empty; a live " +
			"probe without a credential would measure the credential, not the adapter")
	}

	caps := NewProvider(apiKey, "", "").GetCapabilities()
	require.NotNil(t, caps)
	assert.NotEmpty(t, caps.SupportedModels,
		"live discovery with a real credential returned no models — either the "+
			"AI21 models endpoint moved or the credential is not valid")
}

func TestValidateConfig(t *testing.T) {
	tests := []struct {
		name     string
		apiKey   string
		expected bool
	}{
		{"valid key", "test-api-key", true},
		{"empty key", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := NewProvider(tt.apiKey, "", "")
			valid, errors := provider.ValidateConfig(nil)
			assert.Equal(t, tt.expected, valid)
			if !tt.expected {
				assert.NotEmpty(t, errors)
			}
		})
	}
}

func TestConvertRequest(t *testing.T) {
	provider := NewProvider("test-api-key", "", "jamba-1.5-large")
	req := &models.LLMRequest{
		ID:     "test-id",
		Prompt: "You are a coding assistant.",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi!"},
			{Role: "user", Content: "Help me"},
		},
		ModelParams: models.ModelParameters{
			Model:         "jamba-1.5-mini",
			Temperature:   0.8,
			MaxTokens:     2000,
			TopP:          0.9,
			StopSequences: []string{"END"},
		},
	}

	apiReq := provider.convertRequest(req)
	assert.Equal(t, "jamba-1.5-mini", apiReq.Model)
	assert.Len(t, apiReq.Messages, 4) // system + 3 messages
	assert.Equal(t, "system", apiReq.Messages[0].Role)
	assert.Equal(t, 0.8, apiReq.Temperature)
	assert.Equal(t, 2000, apiReq.MaxTokens)
	assert.Equal(t, []string{"END"}, apiReq.Stop)
}

func TestConvertRequestDefaultMaxTokens(t *testing.T) {
	provider := NewProvider("test-api-key", "", "")
	req := &models.LLMRequest{
		Messages:    []models.Message{{Role: "user", Content: "Test"}},
		ModelParams: models.ModelParameters{},
	}

	apiReq := provider.convertRequest(req)
	assert.Equal(t, 4096, apiReq.MaxTokens)
}

func TestConvertResponse(t *testing.T) {
	provider := NewProvider("test-api-key", "", "")
	req := &models.LLMRequest{ID: "req-123"}
	startTime := time.Now()

	apiResp := &Response{
		ID:    "resp-456",
		Model: "jamba-1.5-large",
		Choices: []Choice{
			{
				Index:        0,
				Message:      Message{Role: "assistant", Content: "AI21 response"},
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}

	resp := provider.convertResponse(req, apiResp, startTime)
	assert.Equal(t, "resp-456", resp.ID)
	assert.Equal(t, "req-123", resp.RequestID)
	assert.Equal(t, "AI21 response", resp.Content)
	assert.Equal(t, "ai21", resp.ProviderID)
	assert.Equal(t, "AI21 Labs", resp.ProviderName)
	assert.Equal(t, 150, resp.TokensUsed)
	assert.Equal(t, "stop", resp.FinishReason)
}

func TestConvertResponseWithToolCalls(t *testing.T) {
	provider := NewProvider("test-api-key", "", "")
	req := &models.LLMRequest{ID: "req-tools"}
	startTime := time.Now()

	apiResp := &Response{
		ID:    "resp-tools",
		Model: "jamba-1.5-large",
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role: "assistant",
					ToolCalls: []ToolCall{
						{
							ID:   "tc-1",
							Type: "function",
							Function: FunctionCall{
								Name:      "search",
								Arguments: `{"query": "test"}`,
							},
						},
					},
				},
				FinishReason: "stop",
			},
		},
		Usage: Usage{TotalTokens: 30},
	}

	resp := provider.convertResponse(req, apiResp, startTime)
	assert.Equal(t, "tool_calls", resp.FinishReason)
	require.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, "tc-1", resp.ToolCalls[0].ID)
	assert.Equal(t, "search", resp.ToolCalls[0].Function.Name)
}

func TestCalculateConfidence(t *testing.T) {
	provider := NewProvider("test-api-key", "", "")

	tests := []struct {
		content      string
		finishReason string
		minConf      float64
		maxConf      float64
	}{
		{"Short", "stop", 0.9, 1.0},
		{strings.Repeat("Long content ", 20), "stop", 0.95, 1.0},
		{"Short", "length", 0.7, 0.8},
		{"Short", "content_filter", 0.5, 0.6},
	}

	for _, tt := range tests {
		conf := provider.calculateConfidence(tt.content, tt.finishReason)
		assert.GreaterOrEqual(t, conf, tt.minConf)
		assert.LessOrEqual(t, conf, tt.maxConf)
	}
}

func TestCalculateBackoff(t *testing.T) {
	provider := NewProvider("test-api-key", "", "")

	delay1 := provider.calculateBackoff(1)
	delay2 := provider.calculateBackoff(2)

	assert.LessOrEqual(t, delay1, 2*time.Second)
	assert.LessOrEqual(t, delay1, delay2+time.Second)

	delay10 := provider.calculateBackoff(10)
	assert.LessOrEqual(t, delay10, 35*time.Second)
}

func TestGetModel(t *testing.T) {
	provider := NewProvider("test-api-key", "", "jamba-1.5-large")
	assert.Equal(t, "jamba-1.5-large", provider.GetModel())
}

func TestSetModel(t *testing.T) {
	provider := NewProvider("test-api-key", "", "jamba-1.5-large")
	provider.SetModel("jamba-1.5-mini")
	assert.Equal(t, "jamba-1.5-mini", provider.GetModel())
}

func TestGetName(t *testing.T) {
	provider := NewProvider("test-api-key", "", "")
	assert.Equal(t, "ai21", provider.GetName())
}

func TestRetryOnServerError(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		resp := Response{
			ID:      "success",
			Choices: []Choice{{Message: Message{Content: "Success"}, FinishReason: "stop"}},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewProviderWithRetry("test-key", server.URL, "", RetryConfig{
		MaxRetries:   5,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     100 * time.Millisecond,
		Multiplier:   2.0,
	})

	req := &models.LLMRequest{
		Messages: []models.Message{{Role: "user", Content: "Test"}},
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "success", resp.ID)
	assert.Equal(t, 3, attempts)
}

func TestContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
	}))
	defer server.Close()

	provider := NewProvider("test-key", server.URL, "")
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	req := &models.LLMRequest{
		Messages: []models.Message{{Role: "user", Content: "Test"}},
	}

	_, err := provider.Complete(ctx, req)
	require.Error(t, err)
}

func TestMultipleModels(t *testing.T) {
	testModels := []string{
		"jamba-1.5-large",
		"jamba-1.5-mini",
		"jamba-instruct",
	}

	for _, model := range testModels {
		t.Run(model, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req Request
				_ = json.NewDecoder(r.Body).Decode(&req)
				assert.Equal(t, model, req.Model)

				resp := Response{
					ID:      "test-" + model,
					Model:   model,
					Choices: []Choice{{Message: Message{Content: "Response from " + model}, FinishReason: "stop"}},
				}
				_ = json.NewEncoder(w).Encode(resp)
			}))
			defer server.Close()

			provider := NewProvider("test-key", server.URL, model)
			req := &models.LLMRequest{
				Messages:    []models.Message{{Role: "user", Content: "Test"}},
				ModelParams: models.ModelParameters{},
			}

			resp, err := provider.Complete(context.Background(), req)
			require.NoError(t, err)
			assert.Contains(t, resp.Content, model)
		})
	}
}

// =============================================================================
// Benchmarks
// =============================================================================

func BenchmarkAI21Provider_ConvertRequest(b *testing.B) {
	provider := NewProvider("test-key", "", "")
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

func BenchmarkAI21Provider_CalculateConfidence(b *testing.B) {
	provider := NewProvider("test-key", "", "")
	content := "This is a sample response from the AI21 model for confidence scoring."
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.calculateConfidence(content, "stop")
	}
}

func BenchmarkAI21Provider_GetCapabilities(b *testing.B) {
	provider := NewProvider("test-key", "", "")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.GetCapabilities()
	}
}

func BenchmarkAI21Provider_ConvertResponse(b *testing.B) {
	provider := NewProvider("test-key", "", "jamba-1.5-large")
	req := &models.LLMRequest{
		ID: "bench-request",
	}
	resp := &Response{
		ID:    "resp-1",
		Model: "jamba-1.5-large",
		Choices: []Choice{
			{
				Index:        0,
				Message:      Message{Role: "assistant", Content: "This is a benchmark response with enough content."},
				FinishReason: "stop",
			},
		},
		Usage: Usage{
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

// TestProvider_HealthCheck_UsesInjectedBaseURL proves HealthCheck derives its
// endpoint from the configured baseURL (CONST-051(B) / HXC-085) rather than a
// hardcoded production URL. The test fails if HealthCheck ignores the injected
// baseURL (it would never reach this server and the path assertion would never run).
func TestProvider_HealthCheck_UsesInjectedBaseURL(t *testing.T) {
	hit := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/models", r.URL.Path)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"data": []}`))
	}))
	defer server.Close()

	// baseURL mirrors the default shape so modelsURL() resolves onto this server.
	p := NewProvider("test-key", server.URL+"/chat/completions", "")
	require.NoError(t, p.HealthCheck())
	assert.True(t, hit, "HealthCheck must hit the injected baseURL, not a hardcoded URL")
}
