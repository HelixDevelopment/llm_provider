package novita

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
	provider := NewNovitaProvider("test-api-key", "", "")
	assert.NotNil(t, provider)
	assert.Equal(t, "test-api-key", provider.apiKey)
	assert.Equal(t, "https://api.novita.ai/v3/openai/chat/completions", provider.baseURL)
	assert.Equal(t, "meta-llama/llama-3-8b-instruct", provider.model)
}

func TestNewProviderWithCustomURL(t *testing.T) {
	customURL := "https://custom.api.com/v1/chat/completions"
	provider := NewNovitaProvider("test-key", customURL, "custom-model")
	assert.Equal(t, customURL, provider.baseURL)
	assert.Equal(t, "custom-model", provider.model)
}

func TestNewProviderWithRetry(t *testing.T) {
	retryConfig := RetryConfig{
		MaxRetries:   5,
		InitialDelay: 2 * time.Second,
		MaxDelay:     60 * time.Second,
		Multiplier:   3.0,
	}
	provider := NewNovitaProviderWithRetry("test-key", "", "", retryConfig)
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

func TestComplete(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Contains(t, r.Header.Get("Authorization"), "Bearer")

		resp := map[string]interface{}{
			"id":      "resp_123",
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   "test-model",
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"message":       map[string]string{"role": "assistant", "content": "Hello!"},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]int{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"total_tokens":      15,
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewNovitaProvider("test-api-key", server.URL, "")
	req := &models.LLMRequest{
		ID:       "req-1",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
		ModelParams: models.ModelParameters{
			Temperature: 0.7,
			MaxTokens:   1000,
		},
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "resp_123", resp.ID)
	assert.Contains(t, resp.Content, "Hello")
	assert.Equal(t, "novita", resp.ProviderID)
}

func TestCompleteWithError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		resp := map[string]interface{}{
			"error": map[string]string{"message": "Invalid API key"},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewNovitaProvider("invalid-key", server.URL, "")
	req := &models.LLMRequest{ID: "req-1", Messages: []models.Message{{Role: "user", Content: "Hi"}}}

	_, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
}

func TestCompleteStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		chunks := []string{
			"data: {\"id\":\"stream-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
			"data: {\"id\":\"stream-2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\n",
			"data: [DONE]\n\n",
		}

		for _, chunk := range chunks {
			_, _ = w.Write([]byte(chunk))
		}
	}))
	defer server.Close()

	provider := NewNovitaProvider("test-key", server.URL, "")
	req := &models.LLMRequest{ID: "stream-req", Messages: []models.Message{{Role: "user", Content: "Hi"}}}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var responses int
	for range ch {
		responses++
	}
	assert.GreaterOrEqual(t, responses, 1)
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
		assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
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
// `NotEmpty(SupportedModels)` against LIVE discovery. That is not a unit test:
// it asserts that Novita is up and reachable from whatever machine runs it, so
// it passed here only because this host has egress. With HTTP(S)_PROXY pointed
// at a closed port it failed with "Should NOT be empty, but was []".
//
// Worse, the thing it demanded is the OPPOSITE of the documented contract:
// pkg/discovery returns nil when live discovery is unreachable (CONST-036, no
// hardcoded fallback), so the old assertion required a CORRECT implementation
// to fail.
func TestGetCapabilities(t *testing.T) {
	// SYNTHETIC fixture ids. They were real vendor ids, which made this
	// fixture a frozen environment assumption in its own right. Only one
	// property is load-bearing and it is preserved: pkg/discovery's chat filter
	// rejects any id containing "embedding". (Line 23 and the FallbackModels
	// assertions below DO name real ids on purpose -- they pin this adapter's
	// own compiled default and its last-resort catalogue, which are the
	// subject, not a fixture.)
	srv := modelsFixture(t,
		"vendor-a/synthetic-chat-a",
		"vendor-b/synthetic-chat-b",
		"vendor-c/synthetic-embedding",
	)
	provider := NewNovitaProvider("test-key", srv.URL+"/v3/openai/chat/completions", "")

	caps := provider.GetCapabilities()

	assert.NotNil(t, caps)
	// Discovery reached the fixture and its result was plumbed into the
	// capabilities — the behaviour the old assertion was reaching for.
	assert.Contains(t, caps.SupportedModels, "vendor-a/synthetic-chat-a")
	assert.Contains(t, caps.SupportedModels, "vendor-b/synthetic-chat-b")
	// And the chat-model filter still applies to whatever the endpoint returns.
	assert.NotContains(t, caps.SupportedModels, "vendor-c/synthetic-embedding")

	// The static half of the capability record, which was always deterministic
	// and never needed a network at all.
	assert.Contains(t, caps.SupportedFeatures, "text_completion")
	assert.True(t, caps.SupportsStreaming)
	assert.Equal(t, 4096, caps.Limits.MaxTokens)
	assert.Equal(t, 8192, caps.Limits.MaxInputLength)
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

	provider := NewNovitaProvider("test-key", srv.URL+"/v3/openai/chat/completions", "")
	caps := provider.GetCapabilities()

	require.NotNil(t, caps)
	assert.Empty(t, caps.SupportedModels,
		"unreachable discovery must report NO models, not a stale hardcoded catalogue (CONST-036)")
	for _, fallback := range []string{
		"meta-llama/llama-3-8b-instruct",
		"meta-llama/llama-3-70b-instruct",
		"mistralai/mistral-7b-instruct",
	} {
		assert.NotContains(t, caps.SupportedModels, fallback,
			"the deprecated FallbackModels list must never reach a caller")
	}
	// The rest of the record is still served: unavailable models do not make
	// the provider's fixed capabilities unknown.
	assert.True(t, caps.SupportsStreaming)
	assert.Equal(t, "Novita", caps.Metadata["provider"])
}

// TestModelsURLMatchesConstant guards the refactor that made the discovery
// endpoint config-injectable: with the DEFAULT base URL it must still resolve
// to the production constant, byte for byte. Without this, deriving the
// endpoint could silently change where production traffic goes.
func TestModelsURLMatchesConstant(t *testing.T) {
	assert.Equal(t, NovitaModelsURL, NewNovitaProvider("k", "", "").modelsURL())
}

// TestGetCapabilitiesLiveDiscovery is the live probe, kept but made OPT-IN.
//
// It is genuinely useful — it is the only thing here that would notice Novita
// moving its models endpoint — but it depends on a third party and a real
// credential, so it must never be part of the default suite. It SKIPS with a
// stated reason rather than failing, because "nobody exported a key" is not a
// defect in this adapter.
//
//	NOVITA_LIVE_DISCOVERY_TEST=1 NOVITA_API_KEY=<real key> go test ./pkg/providers/novita/
func TestGetCapabilitiesLiveDiscovery(t *testing.T) {
	if os.Getenv("NOVITA_LIVE_DISCOVERY_TEST") == "" {
		t.Skip("live discovery probe is opt-in: set NOVITA_LIVE_DISCOVERY_TEST=1 " + // SKIP-OK: #opt-in-live-probe
			"together with a real NOVITA_API_KEY to run it")
	}
	apiKey := os.Getenv("NOVITA_API_KEY")
	if apiKey == "" {
		t.Skip("NOVITA_LIVE_DISCOVERY_TEST is set but NOVITA_API_KEY is empty; a live " + // SKIP-OK: #opt-in-live-probe
			"probe without a credential would measure the credential, not the adapter")
	}

	caps := NewNovitaProvider(apiKey, "", "").GetCapabilities()
	require.NotNil(t, caps)
	assert.NotEmpty(t, caps.SupportedModels,
		"live discovery with a real credential returned no models — either the "+
			"Novita models endpoint moved or the credential is not valid")
}

func TestValidateConfig(t *testing.T) {
	tests := []struct {
		name      string
		apiKey    string
		wantValid bool
	}{
		{name: "valid config", apiKey: "test-key", wantValid: true},
		{name: "missing api key", apiKey: "", wantValid: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := NewNovitaProvider(tt.apiKey, "", "")
			valid, errs := provider.ValidateConfig(nil)
			assert.Equal(t, tt.wantValid, valid)
			if !tt.wantValid {
				assert.NotEmpty(t, errs)
			}
		})
	}
}

func TestHealthCheck(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping health check test in short mode") // SKIP-OK: #short-mode
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"object": "list",
			"data":   []interface{}{},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewNovitaProvider("test-key", server.URL, "")
	err := provider.HealthCheck()
	assert.NoError(t, err)
}

func TestHealthCheckWithError(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping health check test in short mode") // SKIP-OK: #short-mode
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	provider := NewNovitaProvider("test-key", server.URL, "")
	err := provider.HealthCheck()
	assert.Error(t, err)
}

// Benchmarks

func BenchmarkNovitaProvider_ConvertRequest(b *testing.B) {
	provider := NewNovitaProvider("test-key", "", "")
	req := &models.LLMRequest{
		ID:     "bench-request",
		Prompt: "You are a helpful assistant.",
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

func BenchmarkNovitaProvider_ConvertResponse(b *testing.B) {
	provider := NewNovitaProvider("test-key", "", "")
	req := &models.LLMRequest{ID: "bench-request"}
	startTime := time.Now()
	nResp := &NovitaResponse{
		ID:    "resp-456",
		Model: "meta-llama/llama-3-8b-instruct",
		Choices: []NovitaChoice{
			{
				Index:        0,
				Message:      NovitaMessage{Role: "assistant", Content: "Benchmark response content"},
				FinishReason: "stop",
			},
		},
		Usage: NovitaUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.convertResponse(req, nResp, startTime)
	}
}
