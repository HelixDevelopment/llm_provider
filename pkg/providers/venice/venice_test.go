package venice

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"digital.vasic.llmprovider/pkg/models"
)

func TestNewVeniceProvider(t *testing.T) {
	p := NewProvider("venice-test-key", "", "")
	assert.NotNil(t, p)
	assert.Equal(t, "venice-test-key", p.apiKey)
	assert.Equal(t, VeniceAPIURL, p.baseURL)
	assert.Equal(t, VeniceDefault, p.model)
}

func TestNewVeniceProvider_Custom(t *testing.T) {
	retryConfig := RetryConfig{
		MaxRetries:   5,
		InitialDelay: 2 * time.Second,
		MaxDelay:     60 * time.Second,
		Multiplier:   3.0,
	}

	p := NewProviderWithRetry(
		"venice-test-key",
		"https://custom.venice.ai/api/v1/chat/completions",
		"deepseek-r1-671b",
		retryConfig,
	)

	assert.Equal(t, "venice-test-key", p.apiKey)
	assert.Equal(t,
		"https://custom.venice.ai/api/v1/chat/completions", p.baseURL,
	)
	assert.Equal(t, "deepseek-r1-671b", p.model)
	assert.Equal(t, 5, p.retryConfig.MaxRetries)
	assert.Equal(t, 3.0, p.retryConfig.Multiplier)
}

func TestComplete_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "POST", r.Method)
			assert.Equal(t,
				"Bearer venice-test-key",
				r.Header.Get("Authorization"),
			)
			assert.Equal(t,
				"application/json",
				r.Header.Get("Content-Type"),
			)

			var reqBody Request
			err := json.NewDecoder(r.Body).Decode(&reqBody)
			require.NoError(t, err)
			assert.Equal(t, "llama-3.3-70b", reqBody.Model)
			assert.Len(t, reqBody.Messages, 2) // system + user
			assert.Equal(t, 0.7, reqBody.Temperature)
			assert.Equal(t, 100, reqBody.MaxTokens)

			resp := Response{
				ID:    "chatcmpl-venice-123",
				Model: "llama-3.3-70b",
				Choices: []Choice{{
					Index: 0,
					Message: Message{
						Role:    "assistant",
						Content: "Hello from Venice AI!",
					},
					FinishReason: "stop",
				}},
				Usage: Usage{
					PromptTokens:     10,
					CompletionTokens: 5,
					TotalTokens:      15,
				},
			}

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		},
	))
	defer server.Close()

	p := NewProvider("venice-test-key", server.URL, "llama-3.3-70b")

	req := &models.LLMRequest{
		ID:     "req-123",
		Prompt: "You are a helpful assistant.",
		Messages: []models.Message{
			{Role: "user", Content: "Hello!"},
		},
		ModelParams: models.ModelParameters{
			Temperature: 0.7,
			MaxTokens:   100,
		},
	}

	ctx := context.Background()
	resp, err := p.Complete(ctx, req)

	require.NoError(t, err)
	assert.Equal(t, "chatcmpl-venice-123", resp.ID)
	assert.Equal(t, "req-123", resp.RequestID)
	assert.Equal(t, "venice", resp.ProviderID)
	assert.Equal(t, "Venice", resp.ProviderName)
	assert.Equal(t, "Hello from Venice AI!", resp.Content)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Equal(t, 15, resp.TokensUsed)
	assert.NotNil(t, resp.Metadata["model"])
	assert.GreaterOrEqual(t, resp.ResponseTime, int64(0))
}

func TestComplete_WithMessages(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			var reqBody Request
			_ = json.NewDecoder(r.Body).Decode(&reqBody)

			// system + 3 conversation messages
			assert.Len(t, reqBody.Messages, 4)
			assert.Equal(t, "system", reqBody.Messages[0].Role)
			assert.Equal(t, "user", reqBody.Messages[1].Role)
			assert.Equal(t, "assistant", reqBody.Messages[2].Role)
			assert.Equal(t, "user", reqBody.Messages[3].Role)

			resp := Response{
				ID:    "chatcmpl-venice-multi",
				Model: "llama-3.3-70b",
				Choices: []Choice{{
					Message: Message{
						Content: "Multi-turn reply",
					},
					FinishReason: "stop",
				}},
				Usage: Usage{TotalTokens: 25},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		},
	))
	defer server.Close()

	p := NewProvider("venice-test-key", server.URL, "llama-3.3-70b")

	req := &models.LLMRequest{
		ID:     "req-multi",
		Prompt: "You are helpful.",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi!"},
			{Role: "user", Content: "How are you?"},
		},
		ModelParams: models.ModelParameters{Temperature: 0.5},
	}

	resp, err := p.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "Multi-turn reply", resp.Content)
	assert.Equal(t, 25, resp.TokensUsed)
}

func TestComplete_ErrorResponse_401(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(
				`{"error": {"message": "Invalid API key"}}`,
			))
		},
	))
	defer server.Close()

	p := NewProviderWithRetry(
		"bad-key", server.URL, "llama-3.3-70b",
		RetryConfig{MaxRetries: 0},
	)

	req := &models.LLMRequest{
		ID:       "req-401",
		Messages: []models.Message{{Role: "user", Content: "Hi"}},
	}

	_, err := p.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "401")
}

func TestComplete_ErrorResponse_500(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			attempts++
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(
				`{"error": "Internal server error"}`,
			))
		},
	))
	defer server.Close()

	p := NewProviderWithRetry(
		"venice-test-key", server.URL, "llama-3.3-70b",
		RetryConfig{
			MaxRetries:   2,
			InitialDelay: 5 * time.Millisecond,
			MaxDelay:     20 * time.Millisecond,
			Multiplier:   2.0,
		},
	)

	req := &models.LLMRequest{
		ID:       "req-500",
		Messages: []models.Message{{Role: "user", Content: "Hi"}},
	}

	_, err := p.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "max retries exceeded")
	assert.Equal(t, 3, attempts) // initial + 2 retries
}

func TestComplete_WithTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			var reqBody Request
			_ = json.NewDecoder(r.Body).Decode(&reqBody)

			assert.Len(t, reqBody.Tools, 1)
			assert.Equal(t, "function", reqBody.Tools[0].Type)
			assert.Equal(t,
				"search_web", reqBody.Tools[0].Function.Name,
			)

			resp := Response{
				ID:    "chatcmpl-venice-tools",
				Model: "llama-3.3-70b",
				Choices: []Choice{{
					Index: 0,
					Message: Message{
						Role:    "assistant",
						Content: "",
						ToolCalls: []ToolCall{{
							ID:   "call-venice-tc1",
							Type: "function",
							Function: FunctionCall{
								Name:      "search_web",
								Arguments: `{"query": "latest AI news"}`,
							},
						}},
					},
					FinishReason: "tool_calls",
				}},
				Usage: Usage{
					PromptTokens:     20,
					CompletionTokens: 15,
					TotalTokens:      35,
				},
			}

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		},
	))
	defer server.Close()

	p := NewProvider("venice-test-key", server.URL, "llama-3.3-70b")

	req := &models.LLMRequest{
		ID:     "req-tools",
		Prompt: "You are a search assistant.",
		Messages: []models.Message{
			{Role: "user", Content: "Search for latest AI news"},
		},
		ModelParams: models.ModelParameters{Temperature: 0.5},
		Tools: []models.Tool{{
			Type: "function",
			Function: models.ToolFunction{
				Name:        "search_web",
				Description: "Search the web",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query": map[string]interface{}{
							"type": "string",
						},
					},
				},
			},
		}},
	}

	resp, err := p.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.Len(t, resp.ToolCalls, 1)
	assert.Equal(t, "call-venice-tc1", resp.ToolCalls[0].ID)
	assert.Equal(t, "search_web", resp.ToolCalls[0].Function.Name)
	assert.Equal(t, "tool_calls", resp.FinishReason)
}

func TestCompleteStream_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			var reqBody Request
			_ = json.NewDecoder(r.Body).Decode(&reqBody)
			assert.True(t, reqBody.Stream)

			w.Header().Set("Content-Type", "text/event-stream")
			flusher, _ := w.(http.Flusher)

			chunks := []string{
				`{"id":"chatcmpl-stream-v","choices":[{"delta":{"content":"Hello"}}]}`,
				`{"id":"chatcmpl-stream-v","choices":[{"delta":{"content":" from"}}]}`,
				`{"id":"chatcmpl-stream-v","choices":[{"delta":{"content":" Venice!"}}]}`,
				"[DONE]",
			}

			for _, chunk := range chunks {
				_, _ = w.Write([]byte("data: " + chunk + "\n\n"))
				flusher.Flush()
			}
		},
	))
	defer server.Close()

	p := NewProvider("venice-test-key", server.URL, "llama-3.3-70b")

	req := &models.LLMRequest{
		ID:       "req-stream",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	ch, err := p.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	assert.GreaterOrEqual(t, len(responses), 1)
	lastResp := responses[len(responses)-1]
	assert.Equal(t, "stop", lastResp.FinishReason)
	assert.Equal(t, "Hello from Venice!", lastResp.Content)
	assert.Equal(t, "venice", lastResp.ProviderID)
}

func TestCompleteStream_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error": "Bad request"}`))
		},
	))
	defer server.Close()

	p := NewProviderWithRetry(
		"venice-test-key", server.URL, "llama-3.3-70b",
		RetryConfig{MaxRetries: 0},
	)

	req := &models.LLMRequest{
		ID:       "req-stream-err",
		Messages: []models.Message{{Role: "user", Content: "Hi"}},
	}

	_, err := p.CompleteStream(context.Background(), req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "400")
}

func TestHealthCheck_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, "GET", r.Method)
			assert.Equal(t,
				"Bearer venice-test-key",
				r.Header.Get("Authorization"),
			)
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"data": []}`))
		},
	))
	defer server.Close()

	p := &Provider{
		apiKey:    "venice-test-key",
		baseURL:   server.URL,
		modelsURL: server.URL,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}

	err := p.HealthCheck()
	assert.NoError(t, err)
}

func TestHealthCheck_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
		},
	))
	defer server.Close()

	p := &Provider{
		apiKey:    "bad-key",
		baseURL:   server.URL,
		modelsURL: server.URL,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}

	err := p.HealthCheck()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "status 401")
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
		assert.Equal(t, "Bearer venice-test-key", r.Header.Get("Authorization"))
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
// The flavour-matching loop below used to run against the LIVE Venice
// catalogue. Its comment was right that pre-baked exact model IDs are a
// CONST-036 violation, but the remedy it chose swapped one bluff for another:
// a green run then meant "Venice is up, reachable from this machine, and still
// publishes something llama-shaped" — a statement about a third party's
// inventory, asserted by a unit test. It passed here only because this host has
// egress; with HTTP(S)_PROXY pointed at a closed port it failed three
// assertions at once (NotEmpty, hasLlama, hasUncensored).
//
// The loop is kept verbatim because the ADAPTER logic it exercises is real —
// it just runs against a fixture whose contents this test controls, so what is
// asserted is the plumbing, not Venice's uptime. The live catalogue is still
// checked, by the opt-in probe at the bottom of this file.
func TestGetCapabilities(t *testing.T) {
	srv := modelsFixture(t,
		"llama-3.3-70b",
		"venice-uncensored-1-2",
		"text-embedding-bge-m3",
	)
	p := NewProvider("venice-test-key", srv.URL+"/api/v1/chat/completions", "")
	caps := p.GetCapabilities()

	assert.True(t, caps.SupportsStreaming)
	assert.True(t, caps.SupportsTools)
	assert.True(t, caps.SupportsVision)
	assert.True(t, caps.SupportsFunctionCalling)
	assert.True(t, caps.SupportsReasoning)
	assert.True(t, caps.SupportsSearch)
	assert.True(t, caps.SupportsCodeCompletion)
	assert.True(t, caps.SupportsCodeAnalysis)
	assert.NotEmpty(t, caps.SupportedModels)
	// Model IDs drift as Venice renames its catalogue (e.g.
	// `venice-uncensored` → `venice-uncensored-1-2` /
	// `venice-uncensored-role-play`; bare `llama-3.3-70b` got
	// reissued as a versioned variant on some catalogues). The flavour
	// match — rather than exact equality — is therefore still the right
	// shape for this assertion, and it is what the OPT-IN live probe
	// applies to the real catalogue.
	hasLlama := false
	hasUncensored := false
	for _, m := range caps.SupportedModels {
		if strings.Contains(m, "llama-3") {
			hasLlama = true
		}
		if strings.Contains(m, "uncensored") {
			hasUncensored = true
		}
	}
	assert.True(t, hasLlama, "expected the discovered catalogue to expose at least one llama-3* model; got %v", caps.SupportedModels)
	assert.True(t, hasUncensored, "expected the discovered catalogue to expose at least one uncensored-flavoured model; got %v", caps.SupportedModels)
	// And the chat-model filter still applies to whatever the endpoint returns.
	assert.NotContains(t, caps.SupportedModels, "text-embedding-bge-m3")
	assert.Contains(t, caps.SupportedFeatures, "web_search")
	assert.Contains(t, caps.SupportedFeatures, "reasoning")
	assert.Contains(t, caps.SupportedFeatures, "streaming")
	assert.Equal(t, 131072, caps.Limits.MaxTokens)
	assert.Equal(t, "true", caps.Metadata["web_search"])
	assert.Equal(t, "venice", caps.Metadata["provider"])
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

	p := NewProvider("venice-test-key", srv.URL+"/api/v1/chat/completions", "")
	caps := p.GetCapabilities()

	require.NotNil(t, caps)
	assert.Empty(t, caps.SupportedModels,
		"unreachable discovery must report NO models, not a stale hardcoded catalogue (CONST-036)")
	for _, fallback := range []string{"llama-3.3-70b", "venice-uncensored", "zai-org-glm-4.7"} {
		assert.NotContains(t, caps.SupportedModels, fallback,
			"the deprecated FallbackModels list must never reach a caller")
	}
	// The rest of the record is still served: unavailable models do not make
	// the provider's fixed capabilities unknown.
	assert.True(t, caps.SupportsStreaming)
	assert.Equal(t, "venice", caps.Metadata["provider"])
}

// TestModelsURLMatchesConstant guards the derivation that makes the discovery
// endpoint config-injectable: with the DEFAULT base URL it must still resolve
// to the production constant, byte for byte. Without this, deriving the
// endpoint could silently change where production traffic goes.
func TestModelsURLMatchesConstant(t *testing.T) {
	assert.Equal(t, VeniceModelsURL, NewProvider("k", "", "").modelsURL)
}

// TestGetCapabilitiesLiveDiscovery is the live probe, kept but made OPT-IN.
//
// It is genuinely useful — it is the only thing here that would notice Venice
// moving its models endpoint or retiring a whole model flavour — but it depends
// on a third party and a real credential, so it must never be part of the
// default suite. It SKIPS with a stated reason rather than failing, because
// "nobody exported a key" is not a defect in this adapter.
//
//	VENICE_LIVE_DISCOVERY_TEST=1 VENICE_API_KEY=<real key> go test ./pkg/providers/venice/
func TestGetCapabilitiesLiveDiscovery(t *testing.T) {
	if os.Getenv("VENICE_LIVE_DISCOVERY_TEST") == "" {
		t.Skip("live discovery probe is opt-in: set VENICE_LIVE_DISCOVERY_TEST=1 " + // SKIP-OK: #opt-in-live-probe
			"together with a real VENICE_API_KEY to run it")
	}
	apiKey := os.Getenv("VENICE_API_KEY")
	if apiKey == "" {
		t.Skip("VENICE_LIVE_DISCOVERY_TEST is set but VENICE_API_KEY is empty; a live " + // SKIP-OK: #opt-in-live-probe
			"probe without a credential would measure the credential, not the adapter")
	}

	caps := NewProvider(apiKey, "", "").GetCapabilities()
	require.NotNil(t, caps)
	assert.NotEmpty(t, caps.SupportedModels,
		"live discovery with a real credential returned no models — either the "+
			"Venice models endpoint moved or the credential is not valid")
}

func TestValidateConfig(t *testing.T) {
	tests := []struct {
		name       string
		apiKey     string
		wantValid  bool
		wantErrLen int
	}{
		{
			name:       "Valid key",
			apiKey:     "venice-valid-key-12345",
			wantValid:  true,
			wantErrLen: 0,
		},
		{
			name:       "Empty key",
			apiKey:     "",
			wantValid:  false,
			wantErrLen: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewProvider(tt.apiKey, "", "")
			valid, errs := p.ValidateConfig(nil)
			assert.Equal(t, tt.wantValid, valid)
			assert.Len(t, errs, tt.wantErrLen)
		})
	}
}

func TestRetry_RateLimited(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			attempts++
			if attempts < 3 {
				w.WriteHeader(http.StatusTooManyRequests)
				return
			}
			resp := Response{
				ID: "chatcmpl-retry-rl",
				Choices: []Choice{{
					Message: Message{
						Content: "Success after rate limit",
					},
					FinishReason: "stop",
				}},
				Usage: Usage{TotalTokens: 10},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		},
	))
	defer server.Close()

	p := NewProviderWithRetry(
		"venice-test-key", server.URL, "llama-3.3-70b",
		RetryConfig{
			MaxRetries:   5,
			InitialDelay: 5 * time.Millisecond,
			MaxDelay:     50 * time.Millisecond,
			Multiplier:   2.0,
		},
	)

	req := &models.LLMRequest{
		ID:       "req-rl",
		Messages: []models.Message{{Role: "user", Content: "Hi"}},
	}

	resp, err := p.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "Success after rate limit", resp.Content)
	assert.Equal(t, 3, attempts)
}

func TestRetry_ServerError(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			attempts++
			if attempts < 2 {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			resp := Response{
				ID: "chatcmpl-retry-503",
				Choices: []Choice{{
					Message: Message{
						Content: "Recovered from 503",
					},
					FinishReason: "stop",
				}},
				Usage: Usage{TotalTokens: 8},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		},
	))
	defer server.Close()

	p := NewProviderWithRetry(
		"venice-test-key", server.URL, "llama-3.3-70b",
		RetryConfig{
			MaxRetries:   3,
			InitialDelay: 5 * time.Millisecond,
			MaxDelay:     50 * time.Millisecond,
			Multiplier:   2.0,
		},
	)

	req := &models.LLMRequest{
		ID:       "req-503",
		Messages: []models.Message{{Role: "user", Content: "Hi"}},
	}

	resp, err := p.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "Recovered from 503", resp.Content)
	assert.Equal(t, 2, attempts)
}

func TestContextTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(5 * time.Second)
			w.WriteHeader(http.StatusOK)
		},
	))
	defer server.Close()

	p := NewProvider("venice-test-key", server.URL, "llama-3.3-70b")

	req := &models.LLMRequest{
		ID:       "req-timeout",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	ctx, cancel := context.WithTimeout(
		context.Background(), 100*time.Millisecond,
	)
	defer cancel()

	_, err := p.Complete(ctx, req)
	assert.Error(t, err)
}

func TestGetName(t *testing.T) {
	p := NewProvider("venice-test-key", "", "")
	assert.Equal(t, "venice", p.GetName())
}

func TestGetProviderType(t *testing.T) {
	p := NewProvider("venice-test-key", "", "")
	assert.Equal(t, "venice", p.GetProviderType())
}

func TestGetModel(t *testing.T) {
	p := NewProvider("venice-test-key", "", "llama-3.3-70b")
	assert.Equal(t, "llama-3.3-70b", p.GetModel())
}

func TestSetModel(t *testing.T) {
	p := NewProvider("venice-test-key", "", "llama-3.3-70b")
	assert.Equal(t, "llama-3.3-70b", p.GetModel())

	p.SetModel("deepseek-r1-671b")
	assert.Equal(t, "deepseek-r1-671b", p.GetModel())
}

func TestDefaultRetryConfig(t *testing.T) {
	cfg := DefaultRetryConfig()
	assert.Equal(t, 3, cfg.MaxRetries)
	assert.Equal(t, 1*time.Second, cfg.InitialDelay)
	assert.Equal(t, 30*time.Second, cfg.MaxDelay)
	assert.Equal(t, 2.0, cfg.Multiplier)
}

func TestVeniceParameters(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			var reqBody Request
			err := json.NewDecoder(r.Body).Decode(&reqBody)
			require.NoError(t, err)

			// Verify venice_parameters are present
			require.NotNil(t, reqBody.VeniceParameters)
			assert.Equal(t,
				"on", reqBody.VeniceParameters.EnableWebSearch,
			)
			assert.True(t,
				reqBody.VeniceParameters.EnableWebCitations,
			)
			assert.True(t,
				reqBody.VeniceParameters.StripThinkingResponse,
			)
			assert.Equal(t, "high", reqBody.Reasoning)

			resp := Response{
				ID:    "chatcmpl-venice-params",
				Model: "llama-3.3-70b",
				Choices: []Choice{{
					Message: Message{
						Content: "Response with Venice params",
					},
					FinishReason: "stop",
				}},
				Usage: Usage{TotalTokens: 20},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		},
	))
	defer server.Close()

	p := NewProvider("venice-test-key", server.URL, "llama-3.3-70b")

	req := &models.LLMRequest{
		ID:       "req-params",
		Messages: []models.Message{{Role: "user", Content: "Search test"}},
		ModelParams: models.ModelParameters{
			Temperature: 0.7,
			ProviderSpecific: map[string]interface{}{
				"enable_web_search":       "on",
				"enable_web_citations":    true,
				"strip_thinking_response": true,
				"reasoning":               "high",
			},
		},
	}

	resp, err := p.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "Response with Venice params", resp.Content)
}

func TestVeniceParameters_NoProviderSpecific(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			var reqBody Request
			_ = json.NewDecoder(r.Body).Decode(&reqBody)

			// No venice_parameters when ProviderSpecific is nil
			assert.Nil(t, reqBody.VeniceParameters)
			assert.Empty(t, reqBody.Reasoning)

			resp := Response{
				ID:    "chatcmpl-no-params",
				Model: "llama-3.3-70b",
				Choices: []Choice{{
					Message: Message{
						Content: "Plain response",
					},
					FinishReason: "stop",
				}},
				Usage: Usage{TotalTokens: 10},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		},
	))
	defer server.Close()

	p := NewProvider("venice-test-key", server.URL, "llama-3.3-70b")

	req := &models.LLMRequest{
		ID:       "req-no-params",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := p.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "Plain response", resp.Content)
}

func TestConvertRequest(t *testing.T) {
	p := NewProvider("venice-test-key", "", "llama-3.3-70b")

	req := &models.LLMRequest{
		ID:     "test-req",
		Prompt: "You are a helpful assistant.",
		Messages: []models.Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
			{Role: "user", Content: "What can Venice AI do?"},
		},
		ModelParams: models.ModelParameters{
			Model:         "deepseek-r1-671b",
			Temperature:   0.5,
			MaxTokens:     1000,
			TopP:          0.95,
			StopSequences: []string{"END"},
		},
	}

	apiReq := p.convertRequest(req)

	assert.Equal(t, "deepseek-r1-671b", apiReq.Model) // Model override
	assert.Len(t, apiReq.Messages, 4)                 // System + 3
	assert.Equal(t, "system", apiReq.Messages[0].Role)
	assert.Equal(t,
		"You are a helpful assistant.",
		apiReq.Messages[0].Content,
	)
	assert.Equal(t, 0.5, apiReq.Temperature)
	assert.Equal(t, 1000, apiReq.MaxTokens)
	assert.Equal(t, 0.95, apiReq.TopP)
	assert.Equal(t, []string{"END"}, apiReq.Stop)
}

func TestConvertRequest_DefaultMaxTokens(t *testing.T) {
	p := NewProvider("venice-test-key", "", "")

	req := &models.LLMRequest{
		ID:       "test-req",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
		ModelParams: models.ModelParameters{
			MaxTokens: 0, // Not set
		},
	}

	apiReq := p.convertRequest(req)
	assert.Equal(t, 4096, apiReq.MaxTokens) // Default value
}

func TestCalculateConfidence(t *testing.T) {
	p := NewProvider("venice-test-key", "", "")

	tests := []struct {
		name         string
		content      string
		finishReason string
		minConf      float64
		maxConf      float64
	}{
		{
			name:         "Stop finish",
			content:      "Short response",
			finishReason: "stop",
			minConf:      0.9,
			maxConf:      1.0,
		},
		{
			name:         "Length finish",
			content:      "Short",
			finishReason: "length",
			minConf:      0.7,
			maxConf:      0.8,
		},
		{
			name:         "Content filter",
			content:      "Filtered",
			finishReason: "content_filter",
			minConf:      0.5,
			maxConf:      0.6,
		},
		{
			name: "Long content bonus",
			content: "Venice AI provides uncensored, private " +
				"inference with multiple LLM models including " +
				"Llama, DeepSeek, and Qwen.",
			finishReason: "stop",
			minConf:      0.95,
			maxConf:      1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conf := p.calculateConfidence(
				tt.content, tt.finishReason,
			)
			assert.GreaterOrEqual(t, conf, tt.minConf)
			assert.LessOrEqual(t, conf, tt.maxConf)
		})
	}
}

func TestCalculateBackoff(t *testing.T) {
	p := NewProvider("venice-test-key", "", "")

	// First attempt should be around initial delay (1s)
	delay1 := p.calculateBackoff(1)
	assert.GreaterOrEqual(t, delay1, 900*time.Millisecond)
	assert.LessOrEqual(t, delay1, 1200*time.Millisecond)

	// Second attempt should be roughly doubled
	delay2 := p.calculateBackoff(2)
	assert.GreaterOrEqual(t, delay2, 1800*time.Millisecond)
	assert.LessOrEqual(t, delay2, 2400*time.Millisecond)
}

func TestRetry_502_504(t *testing.T) {
	statusCodes := []int{502, 504}
	for _, code := range statusCodes {
		t.Run(
			http.StatusText(code),
			func(t *testing.T) {
				attempts := 0
				server := httptest.NewServer(http.HandlerFunc(
					func(w http.ResponseWriter, r *http.Request) {
						attempts++
						if attempts < 2 {
							w.WriteHeader(code)
							return
						}
						resp := Response{
							ID: "chatcmpl-retry-code",
							Choices: []Choice{{
								Message: Message{
									Content: "Recovered",
								},
								FinishReason: "stop",
							}},
							Usage: Usage{TotalTokens: 5},
						}
						w.Header().Set(
							"Content-Type",
							"application/json",
						)
						_ = json.NewEncoder(w).Encode(resp)
					},
				))
				defer server.Close()

				p := NewProviderWithRetry(
					"venice-test-key",
					server.URL,
					"llama-3.3-70b",
					RetryConfig{
						MaxRetries:   3,
						InitialDelay: 5 * time.Millisecond,
						MaxDelay:     50 * time.Millisecond,
						Multiplier:   2.0,
					},
				)

				req := &models.LLMRequest{
					ID: "req-code",
					Messages: []models.Message{
						{Role: "user", Content: "Hi"},
					},
				}

				resp, err := p.Complete(
					context.Background(), req,
				)
				require.NoError(t, err)
				assert.Equal(t, "Recovered", resp.Content)
				assert.Equal(t, 2, attempts)
			},
		)
	}
}

func TestModelsURLDerivation(t *testing.T) {
	// Default URL
	p1 := NewProvider("key", "", "")
	assert.Equal(t, VeniceModelsURL, p1.modelsURL)

	// Custom base URL
	p2 := NewProvider(
		"key",
		"https://custom.venice.ai/api/v1/chat/completions",
		"",
	)
	assert.Equal(t,
		"https://custom.venice.ai/api/v1/models",
		p2.modelsURL,
	)
}

func TestEmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			resp := Response{
				ID:      "chatcmpl-empty",
				Model:   "llama-3.3-70b",
				Choices: []Choice{},
				Usage:   Usage{TotalTokens: 0},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		},
	))
	defer server.Close()

	p := NewProvider("venice-test-key", server.URL, "llama-3.3-70b")
	req := &models.LLMRequest{
		ID:       "req-empty",
		Messages: []models.Message{{Role: "user", Content: "Hello"}},
	}

	resp, err := p.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.Empty(t, resp.Content)
	assert.Empty(t, resp.FinishReason)
}

func TestVeniceProvider_GetSupportedEndpoints(t *testing.T) {
	p := NewProvider("key", "", "")
	endpoints := p.GetSupportedEndpoints()
	assert.Contains(t, endpoints, "chat/completions")
	assert.Contains(t, endpoints, "embeddings")
	assert.Contains(t, endpoints, "image/generate")
	assert.Contains(t, endpoints, "audio/speech")
	assert.Contains(t, endpoints, "audio/transcriptions")
	assert.Contains(t, endpoints, "models")
	assert.Contains(t, endpoints, "image/edit")
	assert.Contains(t, endpoints, "image/upscale")
	assert.GreaterOrEqual(t, len(endpoints), 7)
}

func TestVeniceProvider_Capabilities_MultiModal(t *testing.T) {
	p := NewProvider("key", "", "")
	caps := p.GetCapabilities()
	assert.Contains(t, caps.SupportedFeatures, "embeddings")
	assert.Contains(t, caps.SupportedFeatures, "image_generation")
	assert.Contains(t, caps.SupportedFeatures, "text_to_speech")
	assert.Contains(t, caps.SupportedFeatures, "speech_to_text")
	assert.True(t, caps.SupportsVision)
	assert.True(t, caps.SupportsReasoning)
	assert.True(t, caps.SupportsSearch)
}

func TestVeniceProvider_Capabilities_Metadata(t *testing.T) {
	p := NewProvider("key", "", "")
	caps := p.GetCapabilities()
	assert.Equal(t, "true", caps.Metadata["e2ee"])
	assert.Equal(t, "true", caps.Metadata["uncensored_models"])
	assert.Equal(t,
		"chat,embeddings,image,audio,models",
		caps.Metadata["supported_endpoints"],
	)
}

func TestVeniceProvider_EndpointConstants(t *testing.T) {
	assert.Equal(t,
		"https://api.venice.ai/api/v1/embeddings",
		VeniceEmbeddingsURL,
	)
	assert.Equal(t,
		"https://api.venice.ai/api/v1/image/generate",
		VeniceImageURL,
	)
	assert.Equal(t,
		"https://api.venice.ai/api/v1/audio/speech",
		VeniceAudioURL,
	)
	assert.Equal(t,
		"https://api.venice.ai/api/v1/audio/transcriptions",
		VeniceTranscribeURL,
	)
}

// Benchmarks

func BenchmarkProvider_ConvertRequest(b *testing.B) {
	provider := NewProvider(
		"test-key", VeniceAPIURL, "llama-3.3-70b",
	)
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

func BenchmarkProvider_CalculateConfidence(b *testing.B) {
	provider := NewProvider("test-key", "", "llama-3.3-70b")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.calculateConfidence(
			"This is a detailed response with sufficient "+
				"content to test confidence calculation.",
			"stop",
		)
	}
}

func BenchmarkProvider_ConvertResponse(b *testing.B) {
	provider := NewProvider("test-key", "", "llama-3.3-70b")
	req := &models.LLMRequest{
		ID: "bench-request",
	}
	resp := &Response{
		ID:    "resp-1",
		Model: "llama-3.3-70b",
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: "This is a benchmark response.",
				},
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
