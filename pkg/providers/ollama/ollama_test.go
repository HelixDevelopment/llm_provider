package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"digital.vasic.llmprovider/pkg/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewOllamaProvider(t *testing.T) {
	// The constructor now consults the environment when an argument is empty,
	// so this test states the environment it assumes instead of inheriting
	// whatever the machine happens to export. Without this it would pass on a
	// clean workstation and fail on any host where an operator has legitimately
	// exported OLLAMA_HOST — a test that depends on an unstated ambient
	// condition is not measuring the constructor.
	clearCanonical(t)
	t.Setenv(EnvBaseURL, "")
	t.Setenv(EnvModel, "")
	t.Setenv(EnvTimeout, "")

	tests := []struct {
		name     string
		baseURL  string
		model    string
		expected *OllamaProvider
	}{
		{
			name:    "default values",
			baseURL: "",
			model:   "",
			expected: &OllamaProvider{
				baseURL: "http://localhost:11434",
				model:   "llama2",
				httpClient: &http.Client{
					Timeout: 120 * time.Second,
				},
			},
		},
		{
			name:    "custom values",
			baseURL: "http://custom:7061",
			model:   "custom-model",
			expected: &OllamaProvider{
				baseURL: "http://custom:7061",
				model:   "custom-model",
				httpClient: &http.Client{
					Timeout: 120 * time.Second,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := NewOllamaProvider(tt.baseURL, tt.model)
			assert.Equal(t, tt.expected.baseURL, provider.baseURL)
			assert.Equal(t, tt.expected.model, provider.model)
			assert.Equal(t, tt.expected.httpClient.Timeout, provider.httpClient.Timeout)
		})
	}
}

func TestOllamaProvider_Complete(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/api/generate", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var req OllamaRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		assert.Equal(t, "llama2", req.Model)
		assert.Equal(t, "test prompt", req.Prompt)
		assert.False(t, req.Stream)
		assert.Equal(t, 0.7, req.Options.Temperature)

		response := OllamaResponse{
			Model:    "llama2",
			Response: "Test response",
			Done:     true,
			Context:  []int{1, 2, 3, 4, 5},
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewOllamaProvider(server.URL, "llama2")

	req := &models.LLMRequest{
		ID:     "test-123",
		Prompt: "test prompt",
		ModelParams: models.ModelParameters{
			Temperature: 0.7,
		},
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "test-123", resp.RequestID)
	assert.Equal(t, "ollama", resp.ProviderID)
	assert.Equal(t, "Ollama", resp.ProviderName)
	assert.Equal(t, "Test response", resp.Content)
	assert.Equal(t, 0.8, resp.Confidence)
	assert.Equal(t, "stop", resp.FinishReason)
}

func TestOllamaProvider_Complete_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("Internal Server Error"))
	}))
	defer server.Close()

	provider := NewOllamaProvider(server.URL, "llama2")

	req := &models.LLMRequest{
		ID:     "test-123",
		Prompt: "test prompt",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "failed to complete request")
}

func TestOllamaProvider_CompleteStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/api/generate", r.URL.Path)

		var req OllamaRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		assert.True(t, req.Stream)

		// Send streaming responses
		responses := []OllamaResponse{
			{Model: "llama2", Response: "Hello", Done: false},
			{Model: "llama2", Response: " world", Done: false},
			{Model: "llama2", Response: "!", Done: true},
		}

		flusher, _ := w.(http.Flusher)
		for _, resp := range responses {
			_ = json.NewEncoder(w).Encode(resp)
			flusher.Flush()
		}
	}))
	defer server.Close()

	provider := NewOllamaProvider(server.URL, "llama2")

	req := &models.LLMRequest{
		ID:     "test-123",
		Prompt: "test prompt",
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)
	assert.NotNil(t, ch)

	var responses []*models.LLMResponse
	for resp := range ch {
		responses = append(responses, resp)
	}

	// Should have 3 streaming chunks + 1 final empty chunk
	assert.Len(t, responses, 4)
	assert.Equal(t, "Hello", responses[0].Content)
	assert.Equal(t, " world", responses[1].Content)
	assert.Equal(t, "!", responses[2].Content)
	assert.Equal(t, "", responses[3].Content) // Final empty response
	assert.Equal(t, "stop", responses[3].FinishReason)
}

func TestOllamaProvider_CompleteStream_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return invalid JSON to trigger error in streaming
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte("invalid json response"))
	}))
	defer server.Close()

	provider := NewOllamaProvider(server.URL, "llama2")

	req := &models.LLMRequest{
		ID:     "test-123",
		Prompt: "test prompt",
	}

	ch, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)
	assert.NotNil(t, ch)

	// Read first response (should be error)
	resp := <-ch
	// CONST-046 round-441: the decode-error response Content is routed
	// through the i18n seam. With no Translator wired the NoopTranslator
	// echoes the message ID verbatim. Localized-string coverage lives in
	// ollama_responseerror_i18n_test.go.
	assert.Equal(t, "llmprovider_ollama_response_decode_error", resp.Content)
	assert.Equal(t, "error", resp.FinishReason)
}

// TestOllamaProvider_CompleteStream_ContextCancellation asserts the BEHAVIOUR —
// cancelling the context terminates the stream and closes the channel — rather
// than asserting that it happens inside an arbitrary wall-clock window.
//
// WHAT WAS WRONG BEFORE. The test raced three frozen durations against each
// other: the handler slept 100ms, the context expired after 10ms, and the
// assertion gave up after 50ms. Nothing about cancellation is being measured
// there; what is measured is whether this machine scheduled a goroutine within
// 50ms. On a loaded host it does not, and the test failed with cancellation
// working perfectly — measured at 2 failures in 10 runs. A time-based threshold
// is not a test of behaviour, it is a test of the machine.
//
// Two of the three durations are now gone. The handler blocks until the test
// releases it, so the response cannot win the race by accident; the context is
// cancelled explicitly, so cancellation is an event the test CAUSES rather than
// one it waits out. The single remaining bound is a deadlock guard, orders of
// magnitude larger than the propagation it protects, and it is not the
// assertion — see the comment on it below.
//
// A NOTE ON WHAT "CLOSED" MEANS HERE. The channel is unbuffered and the
// producer sends an error response BEFORE its `defer close(ch)` runs, so a
// cancelled stream yields one value and THEN closes. The old `case <-ch:`
// therefore accepted that value and never observed a closure at all — its own
// comment ("Channel should be closed") described something it did not check.
// Draining to closure is what makes this an assertion about termination.
func TestOllamaProvider_CompleteStream_ContextCancellation(t *testing.T) {
	// The handler blocks until the test releases it. A fixed sleep only makes
	// the response LIKELY to lose the race against cancellation; blocking makes
	// it impossible for it to win, which is what the test actually requires.
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-release
		w.WriteHeader(http.StatusOK)
	}))
	// LIFO: release the handler first, so Close() is not left waiting on an
	// in-flight request that is parked on the channel above.
	defer server.Close()
	defer close(release)

	provider := NewOllamaProvider(server.URL, "llama2")

	ctx, cancel := context.WithCancel(context.Background())

	req := &models.LLMRequest{
		ID:     "test-123",
		Prompt: "test prompt",
	}

	ch, err := provider.CompleteStream(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, ch)

	// The cause, applied deterministically rather than waited for.
	cancel()

	// A bound is still needed so a genuine hang fails THIS test instead of
	// hanging the whole package, but it is a DEADLOCK GUARD, not a threshold:
	// it is derived from the test's own deadline and is ~3 orders of magnitude
	// larger than the sub-millisecond propagation it guards, so no amount of
	// machine load can turn it into a false failure.
	guard := 30 * time.Second
	if d, ok := t.Deadline(); ok {
		if half := time.Until(d) / 2; half > 0 && half < guard {
			guard = half
		}
	}
	timer := time.NewTimer(guard)
	defer timer.Stop()

	sawResponse := false
	for {
		select {
		case resp, open := <-ch:
			if !open {
				// THE ASSERTION: the producer goroutine ran its deferred
				// close, so cancellation propagated and the stream terminated.
				assert.True(t, sawResponse,
					"cancelled stream closed without reporting why; expected a terminal error response first")
				return
			}
			sawResponse = true
			assert.Equal(t, "error", resp.FinishReason,
				"a cancelled stream should terminate with an error response")
		case <-timer.C:
			t.Fatalf("context was cancelled but the response channel never closed within %s: "+
				"cancellation did not propagate to the streaming goroutine", guard)
		}
	}
}

func TestOllamaProvider_HealthCheck(t *testing.T) {
	tests := []struct {
		name           string
		responseStatus int
		expectError    bool
	}{
		{
			name:           "healthy",
			responseStatus: http.StatusOK,
			expectError:    false,
		},
		{
			name:           "unhealthy",
			responseStatus: http.StatusServiceUnavailable,
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "GET", r.Method)
				assert.Equal(t, "/api/tags", r.URL.Path)
				w.WriteHeader(tt.responseStatus)
			}))
			defer server.Close()

			provider := NewOllamaProvider(server.URL, "llama2")
			err := provider.HealthCheck()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestOllamaProvider_HealthCheck_NetworkError(t *testing.T) {
	provider := NewOllamaProvider("http://invalid-url:1234", "llama2")
	err := provider.HealthCheck()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "health check request failed")
}

func TestOllamaProvider_GetCapabilities(t *testing.T) {
	// CONST-035 anti-bluff: this test previously instantiated the provider
	// with NewOllamaProvider("", "") which defaults to
	// http://localhost:11434 — meaning the assertion accidentally
	// exercised whichever local Ollama daemon the operator happened to
	// have running, drifting whenever the daemon's installed-model list
	// changed (or asserting against an offline fallback when no daemon
	// was running). The bluff: a green test depended on the operator's
	// daemon state, not on the provider's wiring. Replace with an
	// httptest server returning a known catalogue so we verify the
	// discovery pipeline plumbs models from API → caps.SupportedModels.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":[{"name":"llama2"},{"name":"mistral"},{"name":"codellama"}]}`))
	}))
	defer server.Close()

	provider := NewOllamaProvider(server.URL, "llama2")
	caps := provider.GetCapabilities()

	assert.NotNil(t, caps)
	// Positive evidence: discovery pulled the known catalogue from the
	// controlled endpoint. If the discovery wiring breaks, this list
	// will be empty or missing entries.
	assert.Contains(t, caps.SupportedModels, "llama2")
	assert.Contains(t, caps.SupportedModels, "mistral")
	assert.Contains(t, caps.SupportedModels, "codellama")
	assert.Contains(t, caps.SupportedFeatures, "text_completion")
	assert.Contains(t, caps.SupportedFeatures, "streaming")
	assert.True(t, caps.SupportsStreaming)
	assert.False(t, caps.SupportsFunctionCalling)
	assert.False(t, caps.SupportsVision)
	assert.Equal(t, 4096, caps.Limits.MaxTokens)
	assert.Equal(t, 1, caps.Limits.MaxConcurrentRequests)
	assert.Equal(t, "Ollama", caps.Metadata["provider"])
	assert.Equal(t, "true", caps.Metadata["local"])
}

func TestOllamaProvider_ValidateConfig(t *testing.T) {
	// Note: NewOllamaProvider sets defaults for baseURL and model,
	// so validation will always pass when using the constructor.
	// The validator is useful for checking manually created providers.
	tests := []struct {
		name        string
		baseURL     string
		model       string
		expectValid bool
		expectedErr []string
	}{
		{
			name:        "valid config",
			baseURL:     "http://localhost:11434",
			model:       "llama2",
			expectValid: true,
			expectedErr: []string{},
		},
		{
			name:        "empty base URL uses default",
			baseURL:     "",
			model:       "llama2",
			expectValid: true, // NewOllamaProvider sets default baseURL
			expectedErr: []string{},
		},
		{
			name:        "empty model uses default",
			baseURL:     "http://localhost:11434",
			model:       "",
			expectValid: true, // NewOllamaProvider sets default model
			expectedErr: []string{},
		},
		{
			name:        "empty both uses defaults",
			baseURL:     "",
			model:       "",
			expectValid: true, // NewOllamaProvider sets both defaults
			expectedErr: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := NewOllamaProvider(tt.baseURL, tt.model)
			valid, errs := provider.ValidateConfig(nil)

			assert.Equal(t, tt.expectValid, valid)
			assert.Equal(t, len(tt.expectedErr), len(errs))
			for i, expectedErr := range tt.expectedErr {
				if i < len(errs) {
					assert.Contains(t, errs[i], expectedErr)
				}
			}
		})
	}
}

func TestOllamaProvider_convertResponse(t *testing.T) {
	provider := NewOllamaProvider("", "")

	ollamaResp := &OllamaResponse{
		Model:    "llama2",
		Response: "Test response",
		Done:     true,
		Context:  []int{1, 2, 3, 4, 5},
	}

	resp, err := provider.convertResponse(ollamaResp, "test-123")
	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "test-123", resp.RequestID)
	assert.Equal(t, "ollama", resp.ProviderID)
	assert.Equal(t, "Ollama", resp.ProviderName)
	assert.Equal(t, "Test response", resp.Content)
	assert.Equal(t, 0.8, resp.Confidence)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Contains(t, resp.Metadata["model"], "llama2")
	assert.Equal(t, 5, resp.Metadata["context"])
}

func TestOllamaProvider_makeRequest(t *testing.T) {
	tests := []struct {
		name           string
		request        OllamaRequest
		response       OllamaResponse
		responseStatus int
		expectError    bool
	}{
		{
			name: "successful request",
			request: OllamaRequest{
				Model:  "llama2",
				Prompt: "test",
				Stream: false,
			},
			response: OllamaResponse{
				Model:    "llama2",
				Response: "test response",
				Done:     true,
			},
			responseStatus: http.StatusOK,
			expectError:    false,
		},
		{
			name: "API error",
			request: OllamaRequest{
				Model:  "llama2",
				Prompt: "test",
				Stream: false,
			},
			response:       OllamaResponse{},
			responseStatus: http.StatusBadRequest,
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "POST", r.Method)
				assert.Equal(t, "/api/generate", r.URL.Path)
				assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

				var req OllamaRequest
				err := json.NewDecoder(r.Body).Decode(&req)
				require.NoError(t, err)

				assert.Equal(t, tt.request.Model, req.Model)
				assert.Equal(t, tt.request.Prompt, req.Prompt)
				assert.Equal(t, tt.request.Stream, req.Stream)

				w.WriteHeader(tt.responseStatus)
				if tt.responseStatus == http.StatusOK {
					_ = json.NewEncoder(w).Encode(tt.response)
				} else {
					_, _ = w.Write([]byte("Bad Request"))
				}
			}))
			defer server.Close()

			provider := NewOllamaProvider(server.URL, "llama2")
			resp, err := provider.makeRequest(context.Background(), tt.request)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, resp)
			} else {
				require.NoError(t, err)
				assert.NotNil(t, resp)
				assert.Equal(t, tt.response.Model, resp.Model)
				assert.Equal(t, tt.response.Response, resp.Response)
			}
		})
	}
}

func TestOllamaProvider_makeRequest_InvalidJSON(t *testing.T) {
	// Use an invalid URL to ensure we get a network error, not a server response
	provider := NewOllamaProvider("http://invalid-ollama-host-12345:11434", "llama2")

	req := OllamaRequest{
		Model:  "test",
		Prompt: "test",
	}

	// We'll test the JSON marshaling directly
	_, err := json.Marshal(req)
	assert.NoError(t, err) // Valid request should marshal fine

	// Test with invalid data
	invalidData := make(chan int)
	_, err = json.Marshal(invalidData)
	assert.Error(t, err) // Channel should fail to marshal

	// Since we can't easily create an invalid OllamaRequest struct,
	// we'll test the error path by using an invalid URL
	_, err = provider.makeRequest(context.Background(), req)
	assert.Error(t, err)
	// The error should be either a network error or an API error
	assert.True(t, err != nil, "Expected an error from invalid request")
}

func TestOllamaProvider_makeRequest_NetworkError(t *testing.T) {
	provider := NewOllamaProvider("http://invalid-url:1234", "llama2")

	req := OllamaRequest{
		Model:  "llama2",
		Prompt: "test",
	}

	_, err := provider.makeRequest(context.Background(), req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "HTTP request failed")
}

func TestOllamaProvider_makeRequest_InvalidResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte("invalid json"))
	}))
	defer server.Close()

	provider := NewOllamaProvider(server.URL, "llama2")

	req := OllamaRequest{
		Model:  "llama2",
		Prompt: "test",
	}

	_, err := provider.makeRequest(context.Background(), req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to unmarshal response")
}

// Benchmark tests
func BenchmarkOllamaProvider_Complete(b *testing.B) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := OllamaResponse{
			Model:    "llama2",
			Response: "Test response",
			Done:     true,
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewOllamaProvider(server.URL, "llama2")
	req := &models.LLMRequest{
		ID:     "test-123",
		Prompt: "test prompt",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = provider.Complete(context.Background(), req)
	}
}

func BenchmarkOllamaProvider_GetCapabilities(b *testing.B) {
	provider := NewOllamaProvider("", "")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = provider.GetCapabilities()
	}
}
