package openrouter

import (
	"context"
	"encoding/json"
	"io"
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

func TestNewSimpleOpenRouterProvider(t *testing.T) {
	tests := []struct {
		name   string
		apiKey string
		want   *SimpleOpenRouterProvider
	}{
		{
			name:   "valid api key",
			apiKey: "test-api-key-123",
			want: &SimpleOpenRouterProvider{
				apiKey:  "test-api-key-123",
				baseURL: defaultBaseURL,
				client: &http.Client{
					Timeout: 60 * time.Second,
				},
			},
		},
		{
			name:   "empty api key",
			apiKey: "",
			want: &SimpleOpenRouterProvider{
				apiKey:  "",
				baseURL: defaultBaseURL,
				client: &http.Client{
					Timeout: 60 * time.Second,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewSimpleOpenRouterProvider(tt.apiKey)
			assert.Equal(t, tt.want.apiKey, got.apiKey)
			assert.Equal(t, tt.want.baseURL, got.baseURL)
			assert.NotNil(t, got.client)
			assert.Equal(t, 60*time.Second, got.client.Timeout)
		})
	}
}

func TestNewSimpleOpenRouterProviderWithBaseURL(t *testing.T) {
	tests := []struct {
		name    string
		apiKey  string
		baseURL string
		want    *SimpleOpenRouterProvider
	}{
		{
			name:    "custom base URL",
			apiKey:  "test-key",
			baseURL: "https://custom.example.com",
			want: &SimpleOpenRouterProvider{
				apiKey:  "test-key",
				baseURL: "https://custom.example.com",
				client: &http.Client{
					Timeout: 60 * time.Second,
				},
			},
		},
		{
			name:    "empty base URL defaults to standard URL",
			apiKey:  "test-key",
			baseURL: "",
			want: &SimpleOpenRouterProvider{
				apiKey:  "test-key",
				baseURL: "https://openrouter.ai/api/v1", // Defaults to standard URL when empty
				client: &http.Client{
					Timeout: 60 * time.Second,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewSimpleOpenRouterProviderWithBaseURL(tt.apiKey, tt.baseURL)
			assert.Equal(t, tt.want.apiKey, got.apiKey)
			assert.Equal(t, tt.want.baseURL, got.baseURL)
			assert.NotNil(t, got.client)
			assert.Equal(t, 60*time.Second, got.client.Timeout)
		})
	}
}

func TestSimpleOpenRouterProvider_Complete_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/chat/completions", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		assert.Equal(t, "llm_provider", r.Header.Get("HTTP-Referer"))

		var reqBody struct {
			Model       string           `json:"model"`
			Messages    []models.Message `json:"messages"`
			MaxTokens   int              `json:"max_tokens,omitempty"`
			Temperature float64          `json:"temperature,omitempty"`
		}

		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		err = json.Unmarshal(body, &reqBody)
		require.NoError(t, err)

		assert.Equal(t, "openrouter/anthropic/claude-3.5-sonnet", reqBody.Model)
		// Prompt is converted to a system message for compatibility
		assert.NotEmpty(t, reqBody.Messages, "Messages should not be empty")
		assert.Equal(t, "system", reqBody.Messages[0].Role)
		assert.Equal(t, "Hello, how are you?", reqBody.Messages[0].Content)
		assert.Equal(t, 1000, reqBody.MaxTokens)
		assert.Equal(t, 0.7, reqBody.Temperature)

		response := map[string]interface{}{
			"id": "chatcmpl-123",
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "I'm doing well, thank you for asking!",
					},
				},
			},
			"created": 1677858242,
			"model":   "openrouter/anthropic/claude-3.5-sonnet",
			"usage": map[string]interface{}{
				"prompt_tokens":     10,
				"completion_tokens": 20,
				"total_tokens":      30,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-api-key", server.URL)
	provider.client = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-123",
		ModelParams: models.ModelParameters{
			Model:       "openrouter/anthropic/claude-3.5-sonnet",
			MaxTokens:   1000,
			Temperature: 0.7,
		},
		Prompt: "Hello, how are you?",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)

	assert.Equal(t, "chatcmpl-123", resp.ID)
	assert.Equal(t, "test-req-123", resp.RequestID)
	assert.Equal(t, "openrouter", resp.ProviderID)
	assert.Equal(t, "OpenRouter", resp.ProviderName)
	assert.Equal(t, "I'm doing well, thank you for asking!", resp.Content)
	assert.Equal(t, 0.85, resp.Confidence) // Default confidence for OpenRouter
	assert.Equal(t, 30, resp.TokensUsed)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Equal(t, "openrouter/anthropic/claude-3.5-sonnet", resp.Metadata["model"])
	assert.Equal(t, "openrouter", resp.Metadata["provider"])
}

func TestSimpleOpenRouterProvider_Complete_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"error": map[string]interface{}{
				"message": "Invalid API key",
				"type":    "authentication_error",
				"code":    401,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("invalid-api-key", server.URL)
	provider.client = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-456",
		ModelParams: models.ModelParameters{
			Model: "openrouter/anthropic/claude-3.5-sonnet",
		},
		Prompt: "Test prompt",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "OpenRouter API error: Invalid API key")
}

func TestSimpleOpenRouterProvider_Complete_NoChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"id":      "chatcmpl-789",
			"choices": []map[string]interface{}{},
			"created": 1677858242,
			"model":   "openrouter/anthropic/claude-3.5-sonnet",
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-api-key", server.URL)
	provider.client = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-789",
		ModelParams: models.ModelParameters{
			Model: "openrouter/anthropic/claude-3.5-sonnet",
		},
		Prompt: "Test prompt",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "no choices in OpenRouter response")
}

func TestSimpleOpenRouterProvider_Complete_NetworkError(t *testing.T) {
	provider := NewSimpleOpenRouterProvider("test-api-key")
	// Create a client that will fail
	provider.client = &http.Client{
		Transport: &http.Transport{
			DisableKeepAlives: true,
		},
	}

	req := &models.LLMRequest{
		ID: "test-req-network",
		ModelParams: models.ModelParameters{
			Model: "openrouter/anthropic/claude-3.5-sonnet",
		},
		Prompt: "Test prompt",
	}

	// Use a context that will timeout immediately
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Microsecond)
	defer cancel()

	resp, err := provider.Complete(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "context deadline exceeded")
}

func TestSimpleOpenRouterProvider_Complete_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("invalid json"))
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-api-key", server.URL)
	provider.client = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-json",
		ModelParams: models.ModelParameters{
			Model: "openrouter/anthropic/claude-3.5-sonnet",
		},
		Prompt: "Test prompt",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "failed to decode OpenRouter response")
}

func TestSimpleOpenRouterProvider_CompleteStream(t *testing.T) {
	// Create a mock server that simulates SSE streaming
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		// Send SSE data chunks
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("server doesn't support flushing")
		}

		// Send a single chunk and the done message
		_, _ = w.Write([]byte("data: {\"id\":\"stream-1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"))
		flusher.Flush()
		time.Sleep(50 * time.Millisecond) // Give client time to process
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
		time.Sleep(50 * time.Millisecond) // Keep connection open briefly
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProvider("test-api-key")
	provider.baseURL = server.URL

	req := &models.LLMRequest{
		ID: "test-req-stream",
		ModelParams: models.ModelParameters{
			Model: "openrouter/anthropic/claude-3.5-sonnet",
		},
		Prompt: "Test streaming prompt",
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, ch)

	// Read from channel with timeout
	responses := make([]*models.LLMResponse, 0)
	timeout := time.After(2 * time.Second)
	done := false

	for !done {
		select {
		case resp, ok := <-ch:
			if !ok {
				done = true
				break
			}
			if resp != nil {
				responses = append(responses, resp)
			}
		case <-timeout:
			done = true
		}
	}

	assert.NotEmpty(t, responses, "expected at least one response")
}

// modelsFixture serves an OpenAI-compatible /models response, which is the
// shape pkg/discovery parses when no custom ResponseParser is configured. The
// returned URL is an OpenRouter-style API base URL, so the provider appends
// "/models" to it exactly as it does in production.
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

// TestSimpleOpenRouterProvider_HealthCheck exercises both branches of the
// health check against controlled endpoints.
//
// The "valid api key" case used to call NewSimpleOpenRouterProvider, which
// pins the production base URL, so the assertion `NoError(HealthCheck())` was
// really "openrouter.ai is up, reachable, and accepts the literal string
// test-api-key". It passed here only because this host has egress and
// OpenRouter answers /models unauthenticated; with HTTP(S)_PROXY pointed at a
// closed port it failed with a dial error. The credential-shape branch below
// never needed a network at all — it returns before any request is made.
func TestSimpleOpenRouterProvider_HealthCheck(t *testing.T) {
	t.Run("reachable endpoint", func(t *testing.T) {
		srv := modelsFixture(t, "anthropic/claude-3.5-sonnet")
		provider := NewSimpleOpenRouterProviderWithBaseURL("test-api-key", srv.URL)
		assert.NoError(t, provider.HealthCheck())
	})

	t.Run("empty api key", func(t *testing.T) {
		// Short-circuits before any request: no endpoint is needed, and none
		// must be contacted.
		provider := NewSimpleOpenRouterProvider("")
		err := provider.HealthCheck()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "API key is required")
	})

	t.Run("endpoint rejects the credential", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
		}))
		defer srv.Close()

		provider := NewSimpleOpenRouterProviderWithBaseURL("test-api-key", srv.URL)
		err := provider.HealthCheck()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "invalid or expired")
	})
}

// TestSimpleOpenRouterProvider_GetCapabilities pins the ADAPTER'S BEHAVIOUR
// against a controlled fixture.
//
// It used to construct a provider on the production base URL and assert
// `NotEmpty(SupportedModels)` plus `len >= 5` against LIVE discovery. That is
// not a unit test: it asserts that OpenRouter is up and still lists at least
// five chat models. Worse, the thing it demanded is the OPPOSITE of the
// documented contract — pkg/discovery returns nil when live discovery is
// unreachable (CONST-036, no hardcoded fallback), so the old assertion
// required a CORRECT implementation to fail.
// SYNTHETIC fixture model ids. They were real vendor ids, which made this
// fixture a frozen environment assumption in its own right — a test that names
// a live vendor's catalogue carries the same defect as the code it tests, and
// scripts/audit-environment-assumptions.sh reported exactly that.
//
// Two properties are load-bearing and both are preserved: the ids keep the
// "<vendor>/<model>" shape OpenRouter uses, and the chat filter in
// pkg/discovery rejects any id containing "embedding", so fixtureEmbedding must
// contain that substring and the other five must not. Six in, five kept out of
// the filter, one dropped — which is what `assert.Len(..., 5)` measures.
const (
	fixtureChatA     = "vendor-a/synthetic-chat-a"
	fixtureChatE     = "vendor-e/synthetic-chat-e"
	fixtureEmbedding = "vendor-f/synthetic-text-embedding"
)

func TestSimpleOpenRouterProvider_GetCapabilities(t *testing.T) {
	srv := modelsFixture(t,
		fixtureChatA,
		"vendor-b/synthetic-chat-b",
		"vendor-c/synthetic-chat-c",
		"vendor-d/synthetic-chat-d",
		fixtureChatE,
		fixtureEmbedding,
	)
	provider := NewSimpleOpenRouterProviderWithBaseURL("test-api-key", srv.URL)
	caps := provider.GetCapabilities()

	assert.NotNil(t, caps)
	// Discovery reached the fixture and its result was plumbed into the
	// capabilities — the behaviour the old assertion was reaching for.
	assert.Contains(t, caps.SupportedModels, fixtureChatA)
	assert.Contains(t, caps.SupportedModels, fixtureChatE)
	assert.Len(t, caps.SupportedModels, 5)
	// And the chat-model filter still applies to whatever the endpoint returns.
	assert.NotContains(t, caps.SupportedModels, fixtureEmbedding)

	assert.Contains(t, caps.SupportedFeatures, "text_completion")
	assert.Contains(t, caps.SupportedFeatures, "chat")
	assert.Contains(t, caps.SupportedFeatures, "multi_model_routing")
	assert.Contains(t, caps.SupportedFeatures, "cost_optimization")

	assert.Contains(t, caps.SupportedRequestTypes, "text_completion")
	assert.Contains(t, caps.SupportedRequestTypes, "chat")

	assert.True(t, caps.SupportsStreaming)
	assert.False(t, caps.SupportsFunctionCalling)
	assert.False(t, caps.SupportsVision)
	assert.True(t, caps.SupportsTools)
	assert.True(t, caps.SupportsSearch)
	assert.True(t, caps.SupportsReasoning)
	assert.True(t, caps.SupportsCodeCompletion)
	assert.True(t, caps.SupportsCodeAnalysis)
	assert.True(t, caps.SupportsRefactoring)

	assert.Equal(t, 200000, caps.Limits.MaxTokens)
	assert.Equal(t, 200000, caps.Limits.MaxInputLength)
	assert.Equal(t, 8192, caps.Limits.MaxOutputLength)
	assert.Equal(t, 10, caps.Limits.MaxConcurrentRequests)

	assert.Equal(t, "OpenRouter", caps.Metadata["provider"])
	assert.Equal(t, "v1", caps.Metadata["api_version"])
	assert.Equal(t, "basic", caps.Metadata["routing"])
	assert.Equal(t, "true", caps.Metadata["multi_tenancy"])
}

// TestSimpleOpenRouterProvider_GetCapabilitiesDiscoveryUnavailable is the
// honest-unavailability half.
//
// When discovery cannot answer, SupportedModels MUST be empty — never the
// deprecated FallbackModels catalogue this provider still passes to the
// discoverer. Serving that list would hand a caller model IDs it may not be
// able to invoke, which CONST-036 calls a structural bluff. The names are
// asserted individually because "empty" alone would also pass if the list
// leaked through under a different key.
func TestSimpleOpenRouterProvider_GetCapabilitiesDiscoveryUnavailable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-api-key", srv.URL)
	caps := provider.GetCapabilities()

	require.NotNil(t, caps)
	assert.Empty(t, caps.SupportedModels,
		"unreachable discovery must report NO models, not a stale hardcoded catalogue (CONST-036)")
	for _, fallback := range []string{
		"anthropic/claude-3.5-sonnet",
		"openai/gpt-4o",
		"google/gemini-pro",
	} {
		assert.NotContains(t, caps.SupportedModels, fallback,
			"the deprecated FallbackModels list must never reach a caller")
	}
	// The rest of the record is still served: unavailable models do not make
	// the provider's fixed capabilities unknown.
	assert.True(t, caps.SupportsStreaming)
	assert.Equal(t, "OpenRouter", caps.Metadata["provider"])
}

// TestSimpleOpenRouterProviderModelsEndpointMatchesDefault guards the
// config-injection path: with the DEFAULT base URL, discovery and the health
// check must still address the production models endpoint, byte for byte.
func TestSimpleOpenRouterProviderModelsEndpointMatchesDefault(t *testing.T) {
	assert.Equal(t, "https://openrouter.ai/api/v1",
		NewSimpleOpenRouterProvider("k").baseURL)
	assert.Equal(t, "https://openrouter.ai/api/v1/models",
		NewSimpleOpenRouterProvider("k").baseURL+"/models")
}

// TestSimpleOpenRouterProviderLiveDiscovery is the live probe, kept but made
// OPT-IN.
//
// It is genuinely useful — it is the only thing here that would notice
// OpenRouter moving its models endpoint — but it depends on a third party and a
// real credential, so it must never be part of the default suite. It SKIPS with
// a stated reason rather than failing, because "nobody exported a key" is not a
// defect in this adapter.
//
//	OPENROUTER_LIVE_DISCOVERY_TEST=1 OPENROUTER_API_KEY=<real key> go test ./pkg/providers/openrouter/
func TestSimpleOpenRouterProviderLiveDiscovery(t *testing.T) {
	if os.Getenv("OPENROUTER_LIVE_DISCOVERY_TEST") == "" {
		t.Skip("live discovery probe is opt-in: set OPENROUTER_LIVE_DISCOVERY_TEST=1 " + // SKIP-OK: #opt-in-live-probe
			"together with a real OPENROUTER_API_KEY to run it")
	}
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENROUTER_LIVE_DISCOVERY_TEST is set but OPENROUTER_API_KEY is empty; a " + // SKIP-OK: #opt-in-live-probe
			"live probe without a credential would measure the credential, not the adapter")
	}

	caps := NewSimpleOpenRouterProvider(apiKey).GetCapabilities()
	require.NotNil(t, caps)
	assert.GreaterOrEqual(t, len(caps.SupportedModels), 5,
		"live discovery with a real credential returned fewer than five models — "+
			"either the OpenRouter models endpoint moved or the credential is not valid")
}

func TestSimpleOpenRouterProvider_ValidateConfig(t *testing.T) {
	tests := []struct {
		name      string
		apiKey    string
		config    map[string]interface{}
		wantValid bool
		wantErrs  []string
	}{
		{
			name:      "valid config with api key",
			apiKey:    "test-api-key",
			config:    map[string]interface{}{},
			wantValid: true,
			wantErrs:  nil,
		},
		{
			name:      "empty api key",
			apiKey:    "",
			config:    map[string]interface{}{},
			wantValid: false,
			wantErrs:  []string{"api_key is required"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := NewSimpleOpenRouterProvider(tt.apiKey)
			valid, errs := provider.ValidateConfig(tt.config)
			assert.Equal(t, tt.wantValid, valid)
			assert.Equal(t, tt.wantErrs, errs)
		})
	}
}

func TestSimpleOpenRouterProvider_Complete_WithMessages(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody struct {
			Messages []models.Message `json:"messages"`
		}

		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		err = json.Unmarshal(body, &reqBody)
		require.NoError(t, err)

		assert.Len(t, reqBody.Messages, 2)
		assert.Equal(t, "user", reqBody.Messages[0].Role)
		assert.Equal(t, "Hello", reqBody.Messages[0].Content)
		assert.Equal(t, "assistant", reqBody.Messages[1].Role)
		assert.Equal(t, "Hi there!", reqBody.Messages[1].Content)

		response := map[string]interface{}{
			"id": "chatcmpl-messages",
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "How can I help you today?",
					},
				},
			},
			"created": 1677858242,
			"model":   "openrouter/anthropic/claude-3.5-sonnet",
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-api-key", server.URL)
	provider.client = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-messages",
		ModelParams: models.ModelParameters{
			Model: "openrouter/anthropic/claude-3.5-sonnet",
		},
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
		},
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "How can I help you today?", resp.Content)
}

func TestSimpleOpenRouterProvider_Complete_ContextTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate slow response
		time.Sleep(100 * time.Millisecond)
		response := map[string]interface{}{
			"id": "chatcmpl-slow",
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Slow response",
					},
				},
			},
			"created": 1677858242,
			"model":   "openrouter/anthropic/claude-3.5-sonnet",
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-api-key", server.URL)
	provider.client = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-timeout",
		ModelParams: models.ModelParameters{
			Model: "openrouter/anthropic/claude-3.5-sonnet",
		},
		Prompt: "Test prompt",
	}

	// Create context with short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	resp, err := provider.Complete(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "context deadline exceeded")
}

// Test isRetryableStatus function
func TestIsRetryableStatus(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		expected   bool
	}{
		{"429 Too Many Requests", http.StatusTooManyRequests, true},
		{"500 Internal Server Error", http.StatusInternalServerError, true},
		{"502 Bad Gateway", http.StatusBadGateway, true},
		{"503 Service Unavailable", http.StatusServiceUnavailable, true},
		{"504 Gateway Timeout", http.StatusGatewayTimeout, true},
		{"200 OK", http.StatusOK, false},
		{"400 Bad Request", http.StatusBadRequest, false},
		{"401 Unauthorized", http.StatusUnauthorized, false},
		{"403 Forbidden", http.StatusForbidden, false},
		{"404 Not Found", http.StatusNotFound, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRetryableStatus(tt.statusCode)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// Test isAuthRetryableStatus function
func TestIsAuthRetryableStatus(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		expected   bool
	}{
		{"401 Unauthorized", http.StatusUnauthorized, true},
		{"200 OK", http.StatusOK, false},
		{"403 Forbidden", http.StatusForbidden, false},
		{"429 Too Many Requests", http.StatusTooManyRequests, false},
		{"500 Internal Server Error", http.StatusInternalServerError, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isAuthRetryableStatus(tt.statusCode)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// Test Complete with retry on server error
func TestSimpleOpenRouterProvider_Complete_RetryOnServerError(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		response := map[string]interface{}{
			"id": "chatcmpl-retry",
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Success after retry",
					},
				},
			},
			"created": 1677858242,
			"model":   "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithRetry("test-key", server.URL, RetryConfig{
		MaxRetries:   5,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     100 * time.Millisecond,
		Multiplier:   2.0,
	})

	req := &models.LLMRequest{
		ID: "test-retry",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "Success after retry", resp.Content)
	assert.Equal(t, 3, attempts)
}

// Test Complete with all retries exhausted (network error path)
func TestSimpleOpenRouterProvider_Complete_RetryExhausted(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		// Return 503 with an error body (so decode succeeds but API error is returned)
		response := map[string]interface{}{
			"error": map[string]interface{}{
				"message": "Service unavailable",
				"type":    "server_error",
			},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithRetry("test-key", server.URL, RetryConfig{
		MaxRetries:   2,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     50 * time.Millisecond,
		Multiplier:   2.0,
	})

	req := &models.LLMRequest{
		ID: "test-exhaust",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	// After retries exhausted, the final 503 response is decoded and returns API error
	assert.Contains(t, err.Error(), "Service unavailable")
	assert.Equal(t, 3, attempts) // Initial + 2 retries
}

// Test Complete with network failure during retry
func TestSimpleOpenRouterProvider_Complete_RetryNetworkFailure(t *testing.T) {
	// Use an invalid address to simulate network failure
	provider := NewSimpleOpenRouterProviderWithRetry("test-key", "http://localhost:1", RetryConfig{
		MaxRetries:   2,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     50 * time.Millisecond,
		Multiplier:   2.0,
	})
	provider.client = &http.Client{Timeout: 50 * time.Millisecond}

	req := &models.LLMRequest{
		ID: "test-net-fail",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	// Network failures return "OpenRouter API request failed" error after retries exhausted
	assert.Contains(t, err.Error(), "OpenRouter API request failed")
}

// Test Complete with rate limiting (429)
func TestSimpleOpenRouterProvider_Complete_RateLimited429(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		response := map[string]interface{}{
			"id": "chatcmpl-rate",
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "OK after rate limit",
					},
				},
			},
			"created": 1677858242,
			"model":   "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithRetry("test-key", server.URL, RetryConfig{
		MaxRetries:   3,
		InitialDelay: 10 * time.Millisecond,
		MaxDelay:     100 * time.Millisecond,
		Multiplier:   2.0,
	})

	req := &models.LLMRequest{
		ID: "test-rate",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "OK after rate limit", resp.Content)
	assert.Equal(t, 2, attempts)
}

// Test Complete with nil context (should use default)
func TestSimpleOpenRouterProvider_Complete_NilContext(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"id": "chatcmpl-nil-ctx",
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Response with nil context",
					},
				},
			},
			"created": 1677858242,
			"model":   "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", server.URL)

	req := &models.LLMRequest{
		ID: "test-nil-ctx",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	// Pass nil context - the function should handle this
	resp, err := provider.Complete(nil, req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "Response with nil context", resp.Content)
}

// Test Complete with max tokens capping
func TestSimpleOpenRouterProvider_Complete_MaxTokensCapping(t *testing.T) {
	tests := []struct {
		name              string
		inputMaxTokens    int
		expectedMaxTokens int
	}{
		{"zero defaults to 4096", 0, 4096},
		{"negative defaults to 4096", -100, 4096},
		{"normal value unchanged", 1000, 1000},
		{"exceeds limit capped to 16384", 50000, 16384},
		{"at limit unchanged", 16384, 16384},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var reqBody struct {
					MaxTokens int `json:"max_tokens"`
				}
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &reqBody)

				assert.Equal(t, tt.expectedMaxTokens, reqBody.MaxTokens)

				response := map[string]interface{}{
					"id": "chatcmpl-tokens",
					"choices": []map[string]interface{}{
						{
							"message": map[string]interface{}{
								"role":    "assistant",
								"content": "OK",
							},
						},
					},
					"model": "test-model",
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(response)
			}))
			defer server.Close()

			provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", server.URL)

			req := &models.LLMRequest{
				ID: "test-tokens",
				ModelParams: models.ModelParameters{
					Model:     "test-model",
					MaxTokens: tt.inputMaxTokens,
				},
				Prompt: "Test",
			}

			resp, err := provider.Complete(context.Background(), req)
			require.NoError(t, err)
			require.NotNil(t, resp)
		})
	}
}

// Test Complete with tools
func TestSimpleOpenRouterProvider_Complete_WithTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody struct {
			Tools []struct {
				Type     string `json:"type"`
				Function struct {
					Name        string                 `json:"name"`
					Description string                 `json:"description"`
					Parameters  map[string]interface{} `json:"parameters"`
				} `json:"function"`
			} `json:"tools"`
			ToolChoice interface{} `json:"tool_choice"`
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &reqBody)

		// Verify tools are passed correctly
		assert.Len(t, reqBody.Tools, 1)
		assert.Equal(t, "function", reqBody.Tools[0].Type)
		assert.Equal(t, "get_weather", reqBody.Tools[0].Function.Name)
		assert.Equal(t, "Get the weather for a location", reqBody.Tools[0].Function.Description)
		assert.Equal(t, "auto", reqBody.ToolChoice)

		response := map[string]interface{}{
			"id": "chatcmpl-tools",
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "",
						"tool_calls": []map[string]interface{}{
							{
								"id":   "call_123",
								"type": "function",
								"function": map[string]interface{}{
									"name":      "get_weather",
									"arguments": `{"location":"New York"}`,
								},
							},
						},
					},
					"finish_reason": "tool_calls",
				},
			},
			"model": "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", server.URL)

	req := &models.LLMRequest{
		ID: "test-tools",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "What's the weather in New York?",
		Tools: []models.Tool{
			{
				Type: "function",
				Function: models.ToolFunction{
					Name:        "get_weather",
					Description: "Get the weather for a location",
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The city name",
							},
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
	assert.Equal(t, "tool_calls", resp.FinishReason)
	require.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, "call_123", resp.ToolCalls[0].ID)
	assert.Equal(t, "function", resp.ToolCalls[0].Type)
	assert.Equal(t, "get_weather", resp.ToolCalls[0].Function.Name)
	assert.Equal(t, `{"location":"New York"}`, resp.ToolCalls[0].Function.Arguments)
}

// Test Complete with tools - non-function tool type is skipped
func TestSimpleOpenRouterProvider_Complete_WithNonFunctionTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody struct {
			Tools []interface{} `json:"tools"`
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &reqBody)

		// Non-function tools should not be included
		assert.Len(t, reqBody.Tools, 0)

		response := map[string]interface{}{
			"id": "chatcmpl-nonfunction",
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "OK",
					},
				},
			},
			"model": "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", server.URL)

	req := &models.LLMRequest{
		ID: "test-nonfunc",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
		Tools: []models.Tool{
			{
				Type: "other_type", // Not a function type
				Function: models.ToolFunction{
					Name: "some_tool",
				},
			},
		},
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
}

// Test Complete with numeric ID in response
func TestSimpleOpenRouterProvider_Complete_NumericIDInResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"id": 12345, // Numeric ID
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "OK",
					},
				},
			},
			"model": "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", server.URL)

	req := &models.LLMRequest{
		ID: "test-numid",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "12345", resp.ID) // Should be converted to string
}

// Test Complete with finish_reason in response
func TestSimpleOpenRouterProvider_Complete_WithFinishReason(t *testing.T) {
	tests := []struct {
		name           string
		finishReason   string
		expectedReason string
	}{
		{"stop", "stop", "stop"},
		{"length", "length", "length"},
		{"tool_calls", "tool_calls", "tool_calls"},
		{"empty defaults to stop", "", "stop"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				response := map[string]interface{}{
					"id": "chatcmpl-finish",
					"choices": []map[string]interface{}{
						{
							"message": map[string]interface{}{
								"role":    "assistant",
								"content": "OK",
							},
							"finish_reason": tt.finishReason,
						},
					},
					"model": "test-model",
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(response)
			}))
			defer server.Close()

			provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", server.URL)

			req := &models.LLMRequest{
				ID: "test-finish",
				ModelParams: models.ModelParameters{
					Model: "test-model",
				},
				Prompt: "Test",
			}

			resp, err := provider.Complete(context.Background(), req)
			require.NoError(t, err)
			require.NotNil(t, resp)
			assert.Equal(t, tt.expectedReason, resp.FinishReason)
		})
	}
}

// Test CompleteStream with HTTP error
func TestSimpleOpenRouterProvider_CompleteStream_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error": "Invalid API key"}`))
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("invalid-key", server.URL)

	req := &models.LLMRequest{
		ID: "test-stream-error",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, ch)
	assert.Contains(t, err.Error(), "HTTP 401")
}

// Test CompleteStream with context cancellation
func TestSimpleOpenRouterProvider_CompleteStream_ContextCancelled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)

		// Send some data
		_, _ = w.Write([]byte("data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"))
		flusher.Flush()

		// Wait for context cancellation
		time.Sleep(500 * time.Millisecond)

		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", server.URL)

	req := &models.LLMRequest{
		ID: "test-cancel",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	ctx, cancel := context.WithCancel(context.Background())
	ch, err := provider.CompleteStream(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, ch)

	// Get first response
	select {
	case resp := <-ch:
		assert.NotNil(t, resp)
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for first response")
	}

	// Cancel context
	cancel()

	// Channel should close eventually
	select {
	case _, ok := <-ch:
		if ok {
			// Drain any remaining responses
			for range ch {
			}
		}
	case <-time.After(2 * time.Second):
		// Channel might already be closed
	}
}

// Test CompleteStream with max tokens capping
func TestSimpleOpenRouterProvider_CompleteStream_MaxTokensCapping(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody struct {
			MaxTokens int `json:"max_tokens"`
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &reqBody)

		// Verify max tokens capped to 16384
		assert.Equal(t, 16384, reqBody.MaxTokens)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", server.URL)

	req := &models.LLMRequest{
		ID: "test-stream-tokens",
		ModelParams: models.ModelParameters{
			Model:     "test-model",
			MaxTokens: 100000, // Should be capped
		},
		Prompt: "Test",
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, ch)

	// Drain channel
	for range ch {
	}
}

// Test HealthCheck with mock server
func TestSimpleOpenRouterProvider_HealthCheck_WithServer(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		wantErr    bool
		errMsg     string
	}{
		{"success", http.StatusOK, false, ""},
		{"unauthorized", http.StatusUnauthorized, true, "invalid or expired"},
		{"internal error", http.StatusInternalServerError, true, "status 500"},
		{"service unavailable", http.StatusServiceUnavailable, true, "status 503"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "GET", r.Method)
				assert.Equal(t, "/models", r.URL.Path)
				assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
				w.WriteHeader(tt.statusCode)
			}))
			defer server.Close()

			provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", server.URL)

			err := provider.HealthCheck()
			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errMsg)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// Test HealthCheck with network error
func TestSimpleOpenRouterProvider_HealthCheck_NetworkError(t *testing.T) {
	provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", "http://localhost:1")
	provider.client = &http.Client{Timeout: 100 * time.Millisecond}

	err := provider.HealthCheck()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "health check failed")
}

// Test waitWithJitter
func TestSimpleOpenRouterProvider_WaitWithJitter(t *testing.T) {
	provider := NewSimpleOpenRouterProvider("test-key")

	// Test normal wait
	start := time.Now()
	provider.waitWithJitter(context.Background(), 50*time.Millisecond)
	elapsed := time.Since(start)

	// Should wait at least the delay (50ms) but not more than delay + 10% jitter + margin
	assert.GreaterOrEqual(t, elapsed.Milliseconds(), int64(50))
	assert.Less(t, elapsed.Milliseconds(), int64(70)) // 50ms + 10% jitter + margin
}

// Test waitWithJitter with cancelled context
func TestSimpleOpenRouterProvider_WaitWithJitter_ContextCancelled(t *testing.T) {
	provider := NewSimpleOpenRouterProvider("test-key")

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	start := time.Now()
	provider.waitWithJitter(ctx, 1*time.Second) // Should return immediately
	elapsed := time.Since(start)

	// Should return almost immediately
	assert.Less(t, elapsed.Milliseconds(), int64(50))
}

// Test nextDelay
func TestSimpleOpenRouterProvider_NextDelay(t *testing.T) {
	provider := NewSimpleOpenRouterProviderWithRetry("test-key", "", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     1 * time.Second,
		Multiplier:   2.0,
	})

	// First delay: 100ms * 2 = 200ms
	next := provider.nextDelay(100 * time.Millisecond)
	assert.Equal(t, 200*time.Millisecond, next)

	// Second delay: 200ms * 2 = 400ms
	next = provider.nextDelay(200 * time.Millisecond)
	assert.Equal(t, 400*time.Millisecond, next)

	// Third delay: 800ms * 2 = 1600ms, capped to MaxDelay (1s)
	next = provider.nextDelay(800 * time.Millisecond)
	assert.Equal(t, 1*time.Second, next)
}

// Test DefaultRetryConfig
func TestDefaultRetryConfig(t *testing.T) {
	config := DefaultRetryConfig()

	assert.Equal(t, 3, config.MaxRetries)
	assert.Equal(t, 1*time.Second, config.InitialDelay)
	assert.Equal(t, 30*time.Second, config.MaxDelay)
	assert.Equal(t, 2.0, config.Multiplier)
}

// Test Complete with context cancelled before request
func TestSimpleOpenRouterProvider_Complete_ContextCancelledBeforeRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("should not reach server")
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", server.URL)

	req := &models.LLMRequest{
		ID: "test-cancel",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	resp, err := provider.Complete(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "context cancelled")
}

// Test Complete with nil usage in response
func TestSimpleOpenRouterProvider_Complete_NilUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"id": "chatcmpl-nil-usage",
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "OK",
					},
				},
			},
			"model": "test-model",
			// No usage field
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", server.URL)

	req := &models.LLMRequest{
		ID: "test-nil-usage",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, 0, resp.TokensUsed) // Should be 0 when usage is nil
}

// Test Complete with nil ID in response
func TestSimpleOpenRouterProvider_Complete_NilID(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			// No id field
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "OK",
					},
				},
			},
			"model": "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewSimpleOpenRouterProviderWithBaseURL("test-key", server.URL)

	req := &models.LLMRequest{
		ID: "test-nil-id",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "", resp.ID) // Should be empty when id is nil
}

// Test CompleteStream with SSE processing comments (keep-alive)
func TestSimpleOpenRouterProvider_CompleteStream_ProcessingComments(
	t *testing.T,
) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.WriteHeader(http.StatusOK)

			flusher, ok := w.(http.Flusher)
			if !ok {
				t.Fatal("server doesn't support flushing")
			}

			// Send processing comment (should be skipped)
			_, _ = w.Write(
				[]byte(": OPENROUTER PROCESSING\n\n"),
			)
			flusher.Flush()
			time.Sleep(50 * time.Millisecond)

			// Send actual data
			_, _ = w.Write([]byte(
				"data: {\"choices\":[{\"delta\":" +
					"{\"content\":\"Hello\"}}]}\n\n",
			))
			flusher.Flush()
			time.Sleep(50 * time.Millisecond)

			// Another processing comment
			_, _ = w.Write(
				[]byte(": OPENROUTER PROCESSING\n\n"),
			)
			flusher.Flush()
			time.Sleep(50 * time.Millisecond)

			// More actual data
			_, _ = w.Write([]byte(
				"data: {\"choices\":[{\"delta\":" +
					"{\"content\":\" world\"}}]}\n\n",
			))
			flusher.Flush()
			time.Sleep(50 * time.Millisecond)

			// Done
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			flusher.Flush()
			time.Sleep(50 * time.Millisecond)
		},
	))
	defer server.Close()

	provider := NewSimpleOpenRouterProvider("test-key")
	provider.baseURL = server.URL

	req := &models.LLMRequest{
		ID: "test-processing-comments",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, ch)

	// Read from channel with timeout (same pattern as existing test)
	var responses []*models.LLMResponse
	timeout := time.After(5 * time.Second)
	done := false

	for !done {
		select {
		case resp, ok := <-ch:
			if !ok {
				done = true
				break
			}
			if resp != nil {
				responses = append(responses, resp)
			}
		case <-timeout:
			done = true
		}
	}

	// Should have chunk responses plus final
	require.GreaterOrEqual(t, len(responses), 1)

	// Last response should be the final with accumulated content
	lastResp := responses[len(responses)-1]
	assert.Equal(t, "stop", lastResp.FinishReason)
	assert.Contains(t, lastResp.Content, "Hello")
	assert.Contains(t, lastResp.Content, "world")
}

// Test CompleteStream with mid-stream error
func TestSimpleOpenRouterProvider_CompleteStream_MidStreamError(
	t *testing.T,
) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.WriteHeader(http.StatusOK)

			flusher, ok := w.(http.Flusher)
			if !ok {
				t.Fatal("server doesn't support flushing")
			}

			// Send some valid data first
			_, _ = w.Write([]byte(
				"data: {\"choices\":[{\"delta\":" +
					"{\"content\":\"Partial\"}}]}\n\n",
			))
			flusher.Flush()
			time.Sleep(50 * time.Millisecond)

			// Send mid-stream error
			_, _ = w.Write([]byte(
				"data: {\"error\":{\"message\":" +
					"\"rate limit exceeded\"," +
					"\"code\":429}}\n\n",
			))
			flusher.Flush()
			time.Sleep(50 * time.Millisecond)
		},
	))
	defer server.Close()

	provider := NewSimpleOpenRouterProvider("test-key")
	provider.baseURL = server.URL

	req := &models.LLMRequest{
		ID: "test-midstream-error",
		ModelParams: models.ModelParameters{
			Model: "test-model",
		},
		Prompt: "Test",
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, ch)

	// Read from channel with timeout
	var responses []*models.LLMResponse
	timeout := time.After(5 * time.Second)
	done := false

	for !done {
		select {
		case resp, ok := <-ch:
			if !ok {
				done = true
				break
			}
			if resp != nil {
				responses = append(responses, resp)
			}
		case <-timeout:
			done = true
		}
	}

	// Should have the content chunk plus the error response
	require.GreaterOrEqual(t, len(responses), 1)

	// Find the error response
	lastResp := responses[len(responses)-1]
	assert.Equal(t, "error", lastResp.FinishReason)
	assert.NotNil(t, lastResp.Metadata["error"])
	assert.Equal(t,
		"rate limit exceeded",
		lastResp.Metadata["error"],
	)
}

// =============================================================================
// Benchmarks
// =============================================================================

func BenchmarkOpenRouterProvider_GetCapabilities(b *testing.B) {
	provider := NewSimpleOpenRouterProvider("test-key")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.GetCapabilities()
	}
}

func BenchmarkOpenRouterProvider_ValidateConfig(b *testing.B) {
	provider := NewSimpleOpenRouterProvider("test-key")
	config := map[string]interface{}{
		"api_key": "test-key",
		"model":   "openai/gpt-4",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.ValidateConfig(config)
	}
}

func BenchmarkOpenRouterProvider_NextDelay(b *testing.B) {
	provider := NewSimpleOpenRouterProvider("test-key")
	initialDelay := 1 * time.Second

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.nextDelay(initialDelay)
	}
}

func BenchmarkOpenRouterProvider_NewProvider(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewSimpleOpenRouterProvider("test-key")
	}
}

func BenchmarkOpenRouterProvider_IsRetryableStatus(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		isRetryableStatus(429)
		isRetryableStatus(500)
		isRetryableStatus(200)
	}
}

func BenchmarkOpenRouterProvider_IsAuthRetryableStatus(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		isAuthRetryableStatus(401)
		isAuthRetryableStatus(403)
		isAuthRetryableStatus(200)
	}
}
