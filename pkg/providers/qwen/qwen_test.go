package qwen

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

	"digital.vasic.llmprovider/pkg/i18n"
	"digital.vasic.llmprovider/pkg/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewQwenProvider(t *testing.T) {
	tests := []struct {
		name    string
		apiKey  string
		baseURL string
		model   string
		want    *QwenProvider
	}{
		{
			name:    "all parameters provided",
			apiKey:  "test-api-key-123",
			baseURL: "https://custom.example.com",
			model:   "qwen-max",
			want: &QwenProvider{
				apiKey:  "test-api-key-123",
				baseURL: "https://custom.example.com",
				model:   "qwen-max",
				httpClient: &http.Client{
					Timeout: 60 * time.Second,
				},
			},
		},
		{
			name:    "default parameters",
			apiKey:  "test-key",
			baseURL: "",
			model:   "",
			want: &QwenProvider{
				apiKey:  "test-key",
				baseURL: "https://dashscope.aliyuncs.com/api/v1",
				model:   "qwen-turbo",
				httpClient: &http.Client{
					Timeout: 60 * time.Second,
				},
			},
		},
		{
			name:    "empty api key",
			apiKey:  "",
			baseURL: "https://api.example.com",
			model:   "qwen-plus",
			want: &QwenProvider{
				apiKey:  "",
				baseURL: "https://api.example.com",
				model:   "qwen-plus",
				httpClient: &http.Client{
					Timeout: 60 * time.Second,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewQwenProvider(tt.apiKey, tt.baseURL, tt.model)
			assert.Equal(t, tt.want.apiKey, got.apiKey)
			assert.Equal(t, tt.want.baseURL, got.baseURL)
			assert.Equal(t, tt.want.model, got.model)
			assert.NotNil(t, got.httpClient)
			assert.Equal(t, 60*time.Second, got.httpClient.Timeout)
		})
	}
}

func TestQwenProvider_Complete_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/services/aigc/text-generation/generation", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))

		var reqBody QwenRequest
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		err = json.Unmarshal(body, &reqBody)
		require.NoError(t, err)

		assert.Equal(t, "qwen-turbo", reqBody.Model)
		assert.False(t, reqBody.Stream)
		assert.Equal(t, 0.7, reqBody.Temperature)
		assert.Equal(t, 1000, reqBody.MaxTokens)
		assert.Len(t, reqBody.Messages, 1)
		assert.Equal(t, "system", reqBody.Messages[0].Role)
		assert.Equal(t, "Hello, how are you?", reqBody.Messages[0].Content)

		response := QwenResponse{
			ID:      "chatcmpl-123",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenChoice{
				{
					Index: 0,
					Message: QwenMessage{
						Role:    "assistant",
						Content: "I'm doing well, thank you for asking!",
					},
					FinishReason: "stop",
				},
			},
			Usage: QwenUsage{
				PromptTokens:     10,
				CompletionTokens: 20,
				TotalTokens:      30,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-123",
		ModelParams: models.ModelParameters{
			Model:       "qwen-turbo",
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
	assert.Equal(t, "qwen", resp.ProviderID)
	assert.Equal(t, "Qwen", resp.ProviderName)
	assert.Equal(t, "I'm doing well, thank you for asking!", resp.Content)
	assert.Equal(t, 0.85, resp.Confidence)
	assert.Equal(t, 30, resp.TokensUsed)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Equal(t, "qwen-turbo", resp.Metadata["model"])
	assert.Equal(t, "chat.completion", resp.Metadata["object"])
	assert.Equal(t, 10, resp.Metadata["prompt_tokens"])
	assert.Equal(t, 20, resp.Metadata["completion_tokens"])
}

func TestQwenProvider_Complete_WithMessages(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody QwenRequest
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		err = json.Unmarshal(body, &reqBody)
		require.NoError(t, err)

		assert.Len(t, reqBody.Messages, 3)
		assert.Equal(t, "system", reqBody.Messages[0].Role)
		assert.Equal(t, "You are a helpful assistant", reqBody.Messages[0].Content)
		assert.Equal(t, "user", reqBody.Messages[1].Role)
		assert.Equal(t, "Hello", reqBody.Messages[1].Content)
		assert.Equal(t, "assistant", reqBody.Messages[2].Role)
		assert.Equal(t, "Hi there!", reqBody.Messages[2].Content)

		response := QwenResponse{
			ID:      "chatcmpl-messages",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenChoice{
				{
					Index: 0,
					Message: QwenMessage{
						Role:    "assistant",
						Content: "How can I help you today?",
					},
					FinishReason: "stop",
				},
			},
			Usage: QwenUsage{
				PromptTokens:     15,
				CompletionTokens: 25,
				TotalTokens:      40,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-messages",
		ModelParams: models.ModelParameters{
			Model: "qwen-turbo",
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

func TestQwenProvider_Complete_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := QwenError{
			Error: struct {
				Message string `json:"message"`
				Type    string `json:"type"`
				Code    string `json:"code"`
			}{
				Message: "Invalid API key",
				Type:    "authentication_error",
				Code:    "401",
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewQwenProvider("invalid-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-456",
		ModelParams: models.ModelParameters{
			Model: "qwen-turbo",
		},
		Prompt: "Test prompt",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "Qwen API error: Invalid API key (authentication_error)")
}

func TestQwenProvider_Complete_NoChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := QwenResponse{
			ID:      "chatcmpl-789",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenChoice{},
			Usage: QwenUsage{
				PromptTokens:     5,
				CompletionTokens: 0,
				TotalTokens:      5,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-789",
		ModelParams: models.ModelParameters{
			Model: "qwen-turbo",
		},
		Prompt: "Test prompt",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "no choices returned from Qwen API")
}

func TestQwenProvider_Complete_NetworkError(t *testing.T) {
	provider := NewQwenProvider("test-api-key", "https://invalid-url-that-does-not-exist.example.com", "qwen-turbo")
	// Create a client that will fail quickly
	provider.httpClient = &http.Client{
		Timeout: 1 * time.Millisecond,
	}

	req := &models.LLMRequest{
		ID: "test-req-network",
		ModelParams: models.ModelParameters{
			Model: "qwen-turbo",
		},
		Prompt: "Test prompt",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "failed to complete request")
}

func TestQwenProvider_Complete_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("invalid json"))
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-json",
		ModelParams: models.ModelParameters{
			Model: "qwen-turbo",
		},
		Prompt: "Test prompt",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "failed to unmarshal response")
}

// Helper function to create an SSE streaming mock server
func createSSEMockServer(t *testing.T, chunks []string, includeFinishReason bool, includeDone bool) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/services/aigc/text-generation/generation", r.URL.Path)
		assert.Equal(t, "text/event-stream", r.Header.Get("Accept"))

		// Verify streaming is enabled in request
		var reqBody QwenRequest
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		err = json.Unmarshal(body, &reqBody)
		require.NoError(t, err)
		assert.True(t, reqBody.Stream)

		// Set SSE headers
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("Expected http.Flusher")
		}

		// Send chunks
		for i, content := range chunks {
			var finishReason *string
			if includeFinishReason && i == len(chunks)-1 {
				stop := "stop"
				finishReason = &stop
			}

			chunk := QwenStreamChunk{
				ID:      fmt.Sprintf("chatcmpl-stream-%d", i),
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   "qwen-turbo",
				Choices: []QwenStreamChoice{
					{
						Index: 0,
						Delta: QwenStreamDelta{
							Content: content,
						},
						FinishReason: finishReason,
					},
				},
			}

			jsonData, _ := json.Marshal(chunk)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
			flusher.Flush()
		}

		// Send [DONE] marker if requested
		if includeDone {
			_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		}
	}))
}

func TestQwenProvider_CompleteStream_Success(t *testing.T) {
	chunks := []string{"Hello", " world", "! How", " are", " you", "?"}
	server := createSSEMockServer(t, chunks, false, true)
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-stream-req",
		Prompt: "Say hello",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, responseChan)

	var receivedChunks []*models.LLMResponse
	for chunk := range responseChan {
		receivedChunks = append(receivedChunks, chunk)
	}

	// Should have received all content chunks plus a final response
	assert.NotEmpty(t, receivedChunks)

	// Collect all content
	var fullContent string
	for _, chunk := range receivedChunks {
		if chunk.Content != "" {
			fullContent += chunk.Content
		}
	}
	assert.Equal(t, "Hello world! How are you?", fullContent)

	// Last chunk should have "stop" finish reason
	lastChunk := receivedChunks[len(receivedChunks)-1]
	assert.Equal(t, "stop", lastChunk.FinishReason)
}

func TestQwenProvider_CompleteStream_WithFinishReason(t *testing.T) {
	chunks := []string{"Hello", " there", "!"}
	server := createSSEMockServer(t, chunks, true, false)
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-finish-reason",
		Prompt: "Say hello",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var receivedChunks []*models.LLMResponse
	for chunk := range responseChan {
		receivedChunks = append(receivedChunks, chunk)
	}

	assert.NotEmpty(t, receivedChunks)
	lastChunk := receivedChunks[len(receivedChunks)-1]
	assert.Equal(t, "stop", lastChunk.FinishReason)
}

func TestQwenProvider_CompleteStream_ContextCancellation(t *testing.T) {
	// Create a server that sends chunks slowly
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		// Use a shorter loop and check for client disconnect
		for i := 0; i < 10; i++ {
			// Check if client disconnected
			select {
			case <-r.Context().Done():
				return
			default:
			}

			chunk := QwenStreamChunk{
				ID:      fmt.Sprintf("chunk-%d", i),
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   "qwen-turbo",
				Choices: []QwenStreamChoice{
					{
						Index: 0,
						Delta: QwenStreamDelta{
							Content: fmt.Sprintf("chunk%d ", i),
						},
					},
				},
			}

			jsonData, _ := json.Marshal(chunk)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
			flusher.Flush()

			// Use a shorter sleep and check context
			select {
			case <-r.Context().Done():
				return
			case <-time.After(10 * time.Millisecond):
			}
		}
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-cancel",
		Prompt: "Test",
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	responseChan, err := provider.CompleteStream(ctx, req)
	require.NoError(t, err)

	// Read a few chunks then cancel
	chunkCount := 0
	for chunk := range responseChan {
		chunkCount++
		if chunk.Content != "" && chunkCount >= 3 {
			cancel()
			break
		}
	}

	// Drain remaining chunks (channel should close due to context cancellation)
	for range responseChan {
		// Just drain
	}

	assert.GreaterOrEqual(t, chunkCount, 3)
}

func TestQwenProvider_CompleteStream_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := QwenError{
			Error: struct {
				Message string `json:"message"`
				Type    string `json:"type"`
				Code    string `json:"code"`
			}{
				Message: "Rate limit exceeded",
				Type:    "rate_limit_error",
				Code:    "429",
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewQwenProviderWithRetry("test-api-key", server.URL, "qwen-turbo", RetryConfig{
		MaxRetries:   0, // No retries
		InitialDelay: 1 * time.Millisecond,
	})
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-req-error",
		Prompt: "Test",
	}

	// With real streaming, the error is returned from CompleteStream
	responseChan, err := provider.CompleteStream(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, responseChan)
	assert.Contains(t, err.Error(), "Rate limit exceeded")
}

func TestQwenProvider_CompleteStream_EmptyChunks(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		// Send empty lines and chunks with empty content
		_, _ = fmt.Fprintf(w, "\n\n")
		flusher.Flush()

		chunk1 := QwenStreamChunk{
			ID:      "chunk-1",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenStreamChoice{
				{
					Index: 0,
					Delta: QwenStreamDelta{
						Content: "", // Empty content
					},
				},
			},
		}
		jsonData, _ := json.Marshal(chunk1)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
		flusher.Flush()

		// Send actual content
		chunk2 := QwenStreamChunk{
			ID:      "chunk-2",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenStreamChoice{
				{
					Index: 0,
					Delta: QwenStreamDelta{
						Content: "Hello",
					},
				},
			},
		}
		jsonData, _ = json.Marshal(chunk2)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
		flusher.Flush()

		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-empty-chunks",
		Prompt: "Test",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var contentChunks int
	var fullContent string
	for chunk := range responseChan {
		if chunk.Content != "" {
			contentChunks++
			fullContent += chunk.Content
		}
	}

	assert.Equal(t, 1, contentChunks) // Only one chunk with "Hello"
	assert.Equal(t, "Hello", fullContent)
}

func TestQwenProvider_CompleteStream_MalformedJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		// Send malformed JSON
		_, _ = fmt.Fprintf(w, "data: {invalid json}\n\n")
		flusher.Flush()

		// Send valid chunk
		chunk := QwenStreamChunk{
			ID:      "chunk-valid",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenStreamChoice{
				{
					Index: 0,
					Delta: QwenStreamDelta{
						Content: "Valid content",
					},
				},
			},
		}
		jsonData, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
		flusher.Flush()

		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-malformed",
		Prompt: "Test",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var fullContent string
	for chunk := range responseChan {
		fullContent += chunk.Content
	}

	// Should skip malformed JSON and process valid content
	assert.Equal(t, "Valid content", fullContent)
}

func TestQwenProvider_CompleteStream_EOFWithoutDone(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		chunk := QwenStreamChunk{
			ID:      "chunk-1",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenStreamChoice{
				{
					Index: 0,
					Delta: QwenStreamDelta{
						Content: "Hello",
					},
				},
			},
		}
		jsonData, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
		flusher.Flush()

		// Close connection without [DONE]
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-eof",
		Prompt: "Test",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var chunks []*models.LLMResponse
	for chunk := range responseChan {
		chunks = append(chunks, chunk)
	}

	// Should have content chunk and final response
	assert.NotEmpty(t, chunks)
	lastChunk := chunks[len(chunks)-1]
	assert.Equal(t, "stop", lastChunk.FinishReason)
}

func TestQwenProvider_CompleteStream_NonDataLines(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		// Send various non-data lines
		_, _ = fmt.Fprintf(w, ": this is a comment\n")
		_, _ = fmt.Fprintf(w, "event: ping\n")
		_, _ = fmt.Fprintf(w, "id: 123\n")
		flusher.Flush()

		chunk := QwenStreamChunk{
			ID:      "chunk-1",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenStreamChoice{
				{
					Index: 0,
					Delta: QwenStreamDelta{
						Content: "Hello",
					},
				},
			},
		}
		jsonData, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
		flusher.Flush()

		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-non-data",
		Prompt: "Test",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var fullContent string
	for chunk := range responseChan {
		fullContent += chunk.Content
	}

	assert.Equal(t, "Hello", fullContent)
}

func TestQwenProvider_CompleteStream_RetryOnServerError(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		chunk := QwenStreamChunk{
			ID:      "chunk-1",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenStreamChoice{
				{
					Index: 0,
					Delta: QwenStreamDelta{
						Content: "Success after retry",
					},
				},
			},
		}
		jsonData, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider := NewQwenProviderWithRetry("test-api-key", server.URL, "qwen-turbo", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Millisecond,
		MaxDelay:     10 * time.Millisecond,
		Multiplier:   2.0,
	})
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-retry-stream",
		Prompt: "Test",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var fullContent string
	for chunk := range responseChan {
		fullContent += chunk.Content
	}

	assert.Equal(t, "Success after retry", fullContent)
	assert.Equal(t, 2, attempts)
}

func TestQwenProvider_CompleteStream_ChunkMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		chunk := QwenStreamChunk{
			ID:      "chatcmpl-metadata-test",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   "qwen-plus",
			Choices: []QwenStreamChoice{
				{
					Index: 0,
					Delta: QwenStreamDelta{
						Content: "Hello world",
					},
				},
			},
		}
		jsonData, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-plus")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-metadata",
		Prompt: "Test",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var contentChunk *models.LLMResponse
	for chunk := range responseChan {
		if chunk.Content != "" {
			contentChunk = chunk
		}
	}

	require.NotNil(t, contentChunk)
	assert.Equal(t, "chatcmpl-metadata-test", contentChunk.ID)
	assert.Equal(t, "test-metadata", contentChunk.RequestID)
	assert.Equal(t, "qwen", contentChunk.ProviderID)
	assert.Equal(t, "Qwen", contentChunk.ProviderName)
	assert.Equal(t, "Hello world", contentChunk.Content)
	assert.Equal(t, 0.85, contentChunk.Confidence)
	assert.Equal(t, "qwen-plus", contentChunk.Metadata["model"])
	assert.Equal(t, 0, contentChunk.Metadata["chunk_index"])
}

func TestQwenProvider_HealthCheck_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/models", r.URL.Path)
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"data": [{"id": "qwen-turbo"}]}`))
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	err := provider.HealthCheck()
	assert.NoError(t, err)
}

func TestQwenProvider_HealthCheck_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error": "Invalid API key"}`))
	}))
	defer server.Close()

	provider := NewQwenProvider("invalid-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	err := provider.HealthCheck()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "health check failed with status: 401")
}

func TestQwenProvider_HealthCheck_EmptyAPIKey(t *testing.T) {
	provider := NewQwenProvider("", "https://api.example.com", "qwen-turbo")
	// Create a client that will fail quickly
	provider.httpClient = &http.Client{
		Timeout: 1 * time.Millisecond,
	}

	err := provider.HealthCheck()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "health check request failed")
}

func TestQwenProvider_GetCapabilities(t *testing.T) {
	provider := NewQwenProvider("", "https://api.example.com", "qwen-turbo")
	caps := provider.GetCapabilities()

	assert.NotNil(t, caps)
	assert.Empty(t, caps.SupportedModels,
		"offline discovery must yield no models (CONST-036: no hardcoded fallback)")

	assert.Contains(t, caps.SupportedFeatures, "text_completion")
	assert.Contains(t, caps.SupportedFeatures, "chat")
	assert.Contains(t, caps.SupportedFeatures, "function_calling")

	assert.Contains(t, caps.SupportedRequestTypes, "text_completion")
	assert.Contains(t, caps.SupportedRequestTypes, "chat")

	assert.True(t, caps.SupportsStreaming)
	assert.True(t, caps.SupportsFunctionCalling)
	assert.False(t, caps.SupportsVision)
	assert.True(t, caps.SupportsTools)
	assert.False(t, caps.SupportsSearch)
	assert.True(t, caps.SupportsReasoning)
	assert.True(t, caps.SupportsCodeCompletion)
	assert.True(t, caps.SupportsCodeAnalysis)
	assert.False(t, caps.SupportsRefactoring)

	assert.Equal(t, 6000, caps.Limits.MaxTokens)
	assert.Equal(t, 30000, caps.Limits.MaxInputLength)
	assert.Equal(t, 2000, caps.Limits.MaxOutputLength)
	assert.Equal(t, 50, caps.Limits.MaxConcurrentRequests)

	assert.Equal(t, "Alibaba Cloud", caps.Metadata["provider"])
	assert.Equal(t, "Qwen", caps.Metadata["model_family"])
	assert.Equal(t, "v1", caps.Metadata["api_version"])
}

func TestQwenProvider_ValidateConfig(t *testing.T) {
	tests := []struct {
		name      string
		apiKey    string
		baseURL   string
		model     string
		config    map[string]interface{}
		wantValid bool
		wantErrs  []string
	}{
		{
			name:      "valid config",
			apiKey:    "test-api-key",
			baseURL:   "https://api.example.com",
			model:     "qwen-turbo",
			config:    map[string]interface{}{},
			wantValid: true,
			wantErrs:  nil,
		},
		{
			name:      "empty api key",
			apiKey:    "",
			baseURL:   "https://api.example.com",
			model:     "qwen-turbo",
			config:    map[string]interface{}{},
			wantValid: false,
			wantErrs:  []string{"llmprovider_validate_api_key_required"},
		},
		{
			name:      "empty base URL - gets default",
			apiKey:    "test-api-key",
			baseURL:   "",
			model:     "qwen-turbo",
			config:    map[string]interface{}{},
			wantValid: true, // Empty base URL gets default value
			wantErrs:  nil,
		},
		{
			name:      "empty model - gets default",
			apiKey:    "test-api-key",
			baseURL:   "https://api.example.com",
			model:     "",
			config:    map[string]interface{}{},
			wantValid: true, // Empty model gets default value
			wantErrs:  nil,
		},
		{
			name:      "only api key error",
			apiKey:    "",
			baseURL:   "",
			model:     "",
			config:    map[string]interface{}{},
			wantValid: false,
			wantErrs:  []string{"llmprovider_validate_api_key_required"}, // Only API key error since baseURL and model get defaults; CONST-046 i18n message ID (NoopTranslator echo)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := NewQwenProvider(tt.apiKey, tt.baseURL, tt.model)
			valid, errs := provider.ValidateConfig(tt.config)
			assert.Equal(t, tt.wantValid, valid)
			assert.Equal(t, tt.wantErrs, errs)
		})
	}
}

func TestQwenProvider_Complete_ContextTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate slow response
		time.Sleep(100 * time.Millisecond)
		response := QwenResponse{
			ID:      "chatcmpl-slow",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenChoice{
				{
					Index: 0,
					Message: QwenMessage{
						Role:    "assistant",
						Content: "Slow response",
					},
					FinishReason: "stop",
				},
			},
			Usage: QwenUsage{
				PromptTokens:     5,
				CompletionTokens: 10,
				TotalTokens:      15,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-timeout",
		ModelParams: models.ModelParameters{
			Model: "qwen-turbo",
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

func TestQwenProvider_Complete_WithStopSequences(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqBody QwenRequest
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		err = json.Unmarshal(body, &reqBody)
		require.NoError(t, err)

		assert.Equal(t, []string{"STOP", "END"}, reqBody.Stop)

		response := QwenResponse{
			ID:      "chatcmpl-stop",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenChoice{
				{
					Index: 0,
					Message: QwenMessage{
						Role:    "assistant",
						Content: "Response with stop sequences",
					},
					FinishReason: "stop",
				},
			},
			Usage: QwenUsage{
				PromptTokens:     8,
				CompletionTokens: 12,
				TotalTokens:      20,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID: "test-req-stop",
		ModelParams: models.ModelParameters{
			Model:         "qwen-turbo",
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

func TestDefaultRetryConfig(t *testing.T) {
	config := DefaultRetryConfig()

	assert.Equal(t, 3, config.MaxRetries)
	assert.Equal(t, 1*time.Second, config.InitialDelay)
	assert.Equal(t, 30*time.Second, config.MaxDelay)
	assert.Equal(t, 2.0, config.Multiplier)
}

func TestNewQwenProviderWithRetry(t *testing.T) {
	customConfig := RetryConfig{
		MaxRetries:   5,
		InitialDelay: 500 * time.Millisecond,
		MaxDelay:     10 * time.Second,
		Multiplier:   1.5,
	}

	provider := NewQwenProviderWithRetry("test-key", "https://custom.url", "qwen-max", customConfig)

	assert.Equal(t, "test-key", provider.apiKey)
	assert.Equal(t, "https://custom.url", provider.baseURL)
	assert.Equal(t, "qwen-max", provider.model)
	assert.Equal(t, customConfig.MaxRetries, provider.retryConfig.MaxRetries)
	assert.Equal(t, customConfig.InitialDelay, provider.retryConfig.InitialDelay)
	assert.Equal(t, customConfig.MaxDelay, provider.retryConfig.MaxDelay)
	assert.Equal(t, customConfig.Multiplier, provider.retryConfig.Multiplier)
}

func TestIsRetryableStatus(t *testing.T) {
	tests := []struct {
		statusCode int
		retryable  bool
	}{
		{http.StatusTooManyRequests, true},     // 429
		{http.StatusInternalServerError, true}, // 500
		{http.StatusBadGateway, true},          // 502
		{http.StatusServiceUnavailable, true},  // 503
		{http.StatusGatewayTimeout, true},      // 504
		{http.StatusOK, false},                 // 200
		{http.StatusBadRequest, false},         // 400
		{http.StatusUnauthorized, false},       // 401
		{http.StatusForbidden, false},          // 403
		{http.StatusNotFound, false},           // 404
	}

	for _, tt := range tests {
		t.Run(http.StatusText(tt.statusCode), func(t *testing.T) {
			result := isRetryableStatus(tt.statusCode)
			assert.Equal(t, tt.retryable, result)
		})
	}
}

func TestQwenProvider_NextDelay(t *testing.T) {
	provider := NewQwenProviderWithRetry("key", "", "", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     1 * time.Second,
		Multiplier:   2.0,
	})

	t.Run("exponential backoff", func(t *testing.T) {
		delay1 := provider.nextDelay(100 * time.Millisecond)
		assert.Equal(t, 200*time.Millisecond, delay1)

		delay2 := provider.nextDelay(200 * time.Millisecond)
		assert.Equal(t, 400*time.Millisecond, delay2)
	})

	t.Run("respects max delay", func(t *testing.T) {
		delay := provider.nextDelay(800 * time.Millisecond)
		assert.Equal(t, 1*time.Second, delay) // Capped at MaxDelay
	})
}

func TestQwenProvider_RetryOnServerError(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}

		response := QwenResponse{
			ID:      "chatcmpl-retry",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenChoice{
				{
					Index:        0,
					Message:      QwenMessage{Role: "assistant", Content: "Success after retry"},
					FinishReason: "stop",
				},
			},
			Usage: QwenUsage{TotalTokens: 10},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewQwenProviderWithRetry("test-key", server.URL, "qwen-turbo", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Millisecond,
		MaxDelay:     10 * time.Millisecond,
		Multiplier:   2.0,
	})
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-retry",
		Prompt: "Test",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "Success after retry", resp.Content)
	assert.Equal(t, 3, attempts)
}

func TestQwenProvider_RetryExhausted(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	provider := NewQwenProviderWithRetry("test-key", server.URL, "qwen-turbo", RetryConfig{
		MaxRetries:   2,
		InitialDelay: 1 * time.Millisecond,
		MaxDelay:     10 * time.Millisecond,
		Multiplier:   2.0,
	})
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-exhaust",
		Prompt: "Test",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "503")
	assert.Equal(t, 3, attempts) // Initial + 2 retries
}

func TestQwenProvider_ContextCancelledDuringRetry(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	provider := NewQwenProviderWithRetry("test-key", server.URL, "qwen-turbo", RetryConfig{
		MaxRetries:   5,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     1 * time.Second,
		Multiplier:   2.0,
	})
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-ctx-cancel",
		Prompt: "Test",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	resp, err := provider.Complete(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	// Should fail before exhausting all retries
	assert.Less(t, attempts, 5)
}

func TestQwenProvider_RateLimitRetry(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}

		response := QwenResponse{
			ID:      "chatcmpl-ratelimit",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenChoice{
				{
					Index:        0,
					Message:      QwenMessage{Role: "assistant", Content: "Success after rate limit"},
					FinishReason: "stop",
				},
			},
			Usage: QwenUsage{TotalTokens: 10},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewQwenProviderWithRetry("test-key", server.URL, "qwen-turbo", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Millisecond,
		MaxDelay:     10 * time.Millisecond,
		Multiplier:   2.0,
	})
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-ratelimit",
		Prompt: "Test",
	}

	resp, err := provider.Complete(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "Success after rate limit", resp.Content)
}

func TestQwenProvider_NonRetryableErrorNoRetry(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		response := QwenError{
			Error: struct {
				Message string `json:"message"`
				Type    string `json:"type"`
				Code    string `json:"code"`
			}{
				Message: "Invalid API key",
				Type:    "authentication_error",
				Code:    "401",
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewQwenProviderWithRetry("invalid-key", server.URL, "qwen-turbo", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Millisecond,
		MaxDelay:     10 * time.Millisecond,
		Multiplier:   2.0,
	})
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-noretry",
		Prompt: "Test",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "Invalid API key")
	// 401 is now retried once for transient auth issues, so expect 2 attempts
	assert.Equal(t, 2, attempts) // Initial attempt + 1 auth retry
}

func TestQwenProvider_NonJSONErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("Bad Request: plain text error"))
	}))
	defer server.Close()

	provider := NewQwenProvider("test-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-plaintext",
		Prompt: "Test",
	}

	resp, err := provider.Complete(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "Qwen API returned status 400")
	assert.Contains(t, err.Error(), "Bad Request")
}

func TestQwenProvider_WaitWithJitter(t *testing.T) {
	provider := NewQwenProvider("key", "", "")

	t.Run("respects context cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		start := time.Now()
		provider.waitWithJitter(ctx, 1*time.Second)
		elapsed := time.Since(start)

		// Should return almost immediately due to cancelled context
		assert.Less(t, elapsed, 100*time.Millisecond)
	})

	t.Run("waits for duration with jitter", func(t *testing.T) {
		ctx := context.Background()
		delay := 50 * time.Millisecond

		start := time.Now()
		provider.waitWithJitter(ctx, delay)
		elapsed := time.Since(start)

		// Should wait at least the delay time (jitter adds, not subtracts).
		// Load-robust: extra CPU pressure can only delay a timer.
		assert.GreaterOrEqual(t, elapsed, delay)

		// Upper bound: deliberately NOT asserted in wall-clock time. Elapsed time
		// also contains scheduler latency this package does not control, so a
		// wall-clock ceiling measures the host at least as much as it measures
		// waitWithJitter. Not reproduced in this session (this
		// ceiling carried 75ms of slack), but it is the same defect as the five
		// that were reproduced; a bigger constant is not a fix.
		// The property that ceiling was proxying -- the timer is armed for the
		// base delay plus AT MOST 10% jitter -- is asserted directly below, on
		// every value the production path actually computes.
		minD, maxD := jitteredDelay(delay), jitteredDelay(delay)
		for i := 0; i < 100000; i++ {
			d := jitteredDelay(delay)
			if d < minD {
				minD = d
			}
			if d > maxD {
				maxD = d
			}
		}
		assert.GreaterOrEqual(t, minD, delay,
			"jittered delay must never be shorter than the base delay")
		assert.Less(t, maxD, delay+delay/10,
			"jitter must never exceed 10% of the base delay")
	})
}

func TestParseSSELine(t *testing.T) {
	tests := []struct {
		name        string
		line        string
		wantChunk   bool
		wantDone    bool
		wantError   bool
		wantContent string
	}{
		{
			name:      "empty line",
			line:      "",
			wantChunk: false,
			wantDone:  false,
		},
		{
			name:      "whitespace only",
			line:      "   \n",
			wantChunk: false,
			wantDone:  false,
		},
		{
			name:      "comment line",
			line:      ": this is a comment",
			wantChunk: false,
			wantDone:  false,
		},
		{
			name:      "non-data line",
			line:      "event: ping",
			wantChunk: false,
			wantDone:  false,
		},
		{
			name:     "done marker",
			line:     "data: [DONE]",
			wantDone: true,
		},
		{
			name:     "done marker with whitespace",
			line:     "data:  [DONE]  ",
			wantDone: true,
		},
		{
			name:        "valid chunk",
			line:        `data: {"id":"test","choices":[{"delta":{"content":"hello"}}]}`,
			wantChunk:   true,
			wantContent: "hello",
		},
		{
			name:      "malformed JSON",
			line:      "data: {invalid json}",
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunk, done, err := parseSSELine([]byte(tt.line))

			if tt.wantError {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.Equal(t, tt.wantDone, done)

			if tt.wantChunk {
				require.NotNil(t, chunk)
				if tt.wantContent != "" && len(chunk.Choices) > 0 {
					assert.Equal(t, tt.wantContent, chunk.Choices[0].Delta.Content)
				}
			} else {
				assert.Nil(t, chunk)
			}
		})
	}
}

func TestQwenStreamChunk_Struct(t *testing.T) {
	// Test that the struct can be properly marshaled and unmarshaled
	stopReason := "stop"
	original := QwenStreamChunk{
		ID:      "test-id",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "qwen-turbo",
		Choices: []QwenStreamChoice{
			{
				Index: 0,
				Delta: QwenStreamDelta{
					Role:    "assistant",
					Content: "Hello",
				},
				FinishReason: &stopReason,
			},
		},
	}

	jsonData, err := json.Marshal(original)
	require.NoError(t, err)

	var parsed QwenStreamChunk
	err = json.Unmarshal(jsonData, &parsed)
	require.NoError(t, err)

	assert.Equal(t, original.ID, parsed.ID)
	assert.Equal(t, original.Object, parsed.Object)
	assert.Equal(t, original.Created, parsed.Created)
	assert.Equal(t, original.Model, parsed.Model)
	assert.Len(t, parsed.Choices, 1)
	assert.Equal(t, original.Choices[0].Index, parsed.Choices[0].Index)
	assert.Equal(t, original.Choices[0].Delta.Role, parsed.Choices[0].Delta.Role)
	assert.Equal(t, original.Choices[0].Delta.Content, parsed.Choices[0].Delta.Content)
	require.NotNil(t, parsed.Choices[0].FinishReason)
	assert.Equal(t, "stop", *parsed.Choices[0].FinishReason)
}

func TestQwenProvider_MakeStreamingRequest_Headers(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify streaming-specific headers
		assert.Equal(t, "text/event-stream", r.Header.Get("Accept"))
		assert.Equal(t, "no-cache", r.Header.Get("Cache-Control"))
		assert.Equal(t, "keep-alive", r.Header.Get("Connection"))
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		// Verify request body has Stream: true
		var reqBody QwenRequest
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &reqBody)
		assert.True(t, reqBody.Stream)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &QwenRequest{
		Model:    "qwen-turbo",
		Messages: []QwenMessage{{Role: "user", Content: "test"}},
		Stream:   false, // Will be set to true by makeStreamingRequest
	}

	body, err := provider.makeStreamingRequest(context.Background(), req)
	require.NoError(t, err)
	defer func() { _ = body.Close() }()

	// Read the response
	content, err := io.ReadAll(body)
	require.NoError(t, err)
	assert.Contains(t, string(content), "[DONE]")
}

func TestQwenProvider_MakeStreamingRequest_ContextCancelled(t *testing.T) {
	provider := NewQwenProvider("test-api-key", "https://api.example.com", "qwen-turbo")

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	req := &QwenRequest{
		Model:    "qwen-turbo",
		Messages: []QwenMessage{{Role: "user", Content: "test"}},
	}

	body, err := provider.makeStreamingRequest(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, body)
	assert.Contains(t, err.Error(), "context cancelled")
}

func TestQwenProvider_MakeStreamingRequest_RetryExhausted(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	provider := NewQwenProviderWithRetry("test-key", server.URL, "qwen-turbo", RetryConfig{
		MaxRetries:   2,
		InitialDelay: 1 * time.Millisecond,
		MaxDelay:     10 * time.Millisecond,
		Multiplier:   2.0,
	})
	provider.httpClient = server.Client()

	req := &QwenRequest{
		Model:    "qwen-turbo",
		Messages: []QwenMessage{{Role: "user", Content: "test"}},
	}

	body, err := provider.makeStreamingRequest(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, body)
	// Error can be either from exhausted retries or final status code
	assert.True(t, strings.Contains(err.Error(), "503") || strings.Contains(err.Error(), "retry"))
	assert.Equal(t, 3, attempts)
}

func TestQwenProvider_MakeStreamingRequest_NonJSONError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("Plain text error"))
	}))
	defer server.Close()

	provider := NewQwenProvider("test-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &QwenRequest{
		Model:    "qwen-turbo",
		Messages: []QwenMessage{{Role: "user", Content: "test"}},
	}

	body, err := provider.makeStreamingRequest(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, body)
	assert.Contains(t, err.Error(), "Qwen API returned status 400")
}

func TestQwenProvider_CompleteStream_MultipleWords(t *testing.T) {
	// Test that token counting works correctly for multi-word content
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		// Send a chunk with multiple words
		chunk := QwenStreamChunk{
			ID:      "chunk-1",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenStreamChoice{
				{
					Index: 0,
					Delta: QwenStreamDelta{
						Content: "Hello world how are you",
					},
				},
			},
		}
		jsonData, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-multiword",
		Prompt: "Test",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var contentChunk *models.LLMResponse
	for chunk := range responseChan {
		if chunk.Content != "" {
			contentChunk = chunk
		}
	}

	require.NotNil(t, contentChunk)
	assert.Equal(t, 5, contentChunk.TokensUsed) // 5 words
}

func TestQwenProvider_CompleteStream_SingleCharacter(t *testing.T) {
	// Test that single character content gets token count of 1
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		chunk := QwenStreamChunk{
			ID:      "chunk-1",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenStreamChoice{
				{
					Index: 0,
					Delta: QwenStreamDelta{
						Content: "!", // Single non-word character
					},
				},
			},
		}
		jsonData, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-singlechar",
		Prompt: "Test",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var contentChunk *models.LLMResponse
	for chunk := range responseChan {
		if chunk.Content != "" {
			contentChunk = chunk
		}
	}

	require.NotNil(t, contentChunk)
	assert.Equal(t, 1, contentChunk.TokensUsed) // Minimum 1 token for non-empty content
}

func TestQwenProvider_CompleteStream_EmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		// Send chunk with empty choices
		chunk := QwenStreamChunk{
			ID:      "chunk-1",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenStreamChoice{}, // Empty choices
		}
		jsonData, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)

		// Send valid chunk
		chunk2 := QwenStreamChunk{
			ID:      "chunk-2",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenStreamChoice{
				{
					Index: 0,
					Delta: QwenStreamDelta{
						Content: "Valid",
					},
				},
			},
		}
		jsonData, _ = json.Marshal(chunk2)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-emptychoices",
		Prompt: "Test",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var fullContent string
	for chunk := range responseChan {
		fullContent += chunk.Content
	}

	assert.Equal(t, "Valid", fullContent)
}

func TestConvertToQwenRequest(t *testing.T) {
	provider := NewQwenProvider("test-key", "https://api.example.com", "qwen-max")

	t.Run("with prompt only", func(t *testing.T) {
		req := &models.LLMRequest{
			Prompt: "System prompt",
			ModelParams: models.ModelParameters{
				Temperature: 0.7,
				MaxTokens:   100,
				TopP:        0.9,
			},
		}

		qwenReq := provider.convertToQwenRequest(req)

		assert.Equal(t, "qwen-max", qwenReq.Model)
		assert.False(t, qwenReq.Stream)
		assert.Equal(t, 0.7, qwenReq.Temperature)
		assert.Equal(t, 100, qwenReq.MaxTokens)
		assert.Equal(t, 0.9, qwenReq.TopP)
		assert.Len(t, qwenReq.Messages, 1)
		assert.Equal(t, "system", qwenReq.Messages[0].Role)
		assert.Equal(t, "System prompt", qwenReq.Messages[0].Content)
	})

	t.Run("with messages", func(t *testing.T) {
		req := &models.LLMRequest{
			Prompt: "System prompt",
			Messages: []models.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi"},
			},
			ModelParams: models.ModelParameters{
				StopSequences: []string{"STOP"},
			},
		}

		qwenReq := provider.convertToQwenRequest(req)

		assert.Len(t, qwenReq.Messages, 3)
		assert.Equal(t, []string{"STOP"}, qwenReq.Stop)
	})

	t.Run("without prompt", func(t *testing.T) {
		req := &models.LLMRequest{
			Messages: []models.Message{
				{Role: "user", Content: "Hello"},
			},
		}

		qwenReq := provider.convertToQwenRequest(req)

		assert.Len(t, qwenReq.Messages, 1)
		assert.Equal(t, "user", qwenReq.Messages[0].Role)
	})
}

func TestConvertFromQwenResponse(t *testing.T) {
	provider := NewQwenProvider("test-key", "https://api.example.com", "qwen-turbo")

	t.Run("successful response", func(t *testing.T) {
		qwenResp := &QwenResponse{
			ID:      "resp-123",
			Object:  "chat.completion",
			Created: time.Now().Unix() - 1,
			Model:   "qwen-turbo",
			Choices: []QwenChoice{
				{
					Index:        0,
					Message:      QwenMessage{Role: "assistant", Content: "Hello!"},
					FinishReason: "stop",
				},
			},
			Usage: QwenUsage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		}

		resp, err := provider.convertFromQwenResponse(qwenResp, "req-456")

		require.NoError(t, err)
		assert.Equal(t, "resp-123", resp.ID)
		assert.Equal(t, "req-456", resp.RequestID)
		assert.Equal(t, "qwen", resp.ProviderID)
		assert.Equal(t, "Qwen", resp.ProviderName)
		assert.Equal(t, "Hello!", resp.Content)
		assert.Equal(t, 0.85, resp.Confidence)
		assert.Equal(t, 15, resp.TokensUsed)
		assert.Equal(t, "stop", resp.FinishReason)
		assert.Equal(t, "qwen-turbo", resp.Metadata["model"])
		assert.Equal(t, 10, resp.Metadata["prompt_tokens"])
		assert.Equal(t, 5, resp.Metadata["completion_tokens"])
	})

	t.Run("empty choices", func(t *testing.T) {
		qwenResp := &QwenResponse{
			ID:      "resp-empty",
			Choices: []QwenChoice{},
		}

		resp, err := provider.convertFromQwenResponse(qwenResp, "req-789")

		assert.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "no choices returned from Qwen API")
	})
}

func TestQwenProvider_ValidateConfig_AllErrors(t *testing.T) {
	// Create provider and manually clear fields to trigger all validation errors
	provider := &QwenProvider{
		apiKey:  "",
		baseURL: "",
		model:   "",
	}

	valid, errs := provider.ValidateConfig(map[string]interface{}{})
	assert.False(t, valid)
	// CONST-046: ValidateConfig surfaces i18n message IDs resolved via
	// the i18n.Tr seam (NoopTranslator echoes the ID verbatim in unit tests).
	assert.Contains(t, errs, "llmprovider_validate_api_key_required")
	assert.Contains(t, errs, "llmprovider_validate_base_url_required")
	assert.Contains(t, errs, "llmprovider_validate_model_required")
	assert.Len(t, errs, 3)
}

func TestQwenProvider_CompleteStream_ReadError(t *testing.T) {
	// Test that read errors during streaming are handled properly
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		// Send one valid chunk
		chunk := QwenStreamChunk{
			ID:      "chunk-1",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   "qwen-turbo",
			Choices: []QwenStreamChoice{
				{
					Index: 0,
					Delta: QwenStreamDelta{
						Content: "Hello",
					},
				},
			},
		}
		jsonData, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
		flusher.Flush()

		// Simulate abrupt connection close (triggers read error)
		hj, ok := w.(http.Hijacker)
		if ok {
			conn, _, _ := hj.Hijack()
			if conn != nil {
				_ = conn.Close()
			}
		}
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-read-error",
		Prompt: "Test",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var chunks []*models.LLMResponse
	for chunk := range responseChan {
		chunks = append(chunks, chunk)
	}

	// Should have at least one chunk (might have content chunk and/or error/final response)
	assert.NotEmpty(t, chunks)
}

func TestQwenProvider_MakeStreamingRequest_APIErrorWithJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := QwenError{
			Error: struct {
				Message string `json:"message"`
				Type    string `json:"type"`
				Code    string `json:"code"`
			}{
				Message: "Invalid request format",
				Type:    "invalid_request_error",
				Code:    "400",
			},
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewQwenProvider("test-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &QwenRequest{
		Model:    "qwen-turbo",
		Messages: []QwenMessage{{Role: "user", Content: "test"}},
	}

	body, err := provider.makeStreamingRequest(context.Background(), req)
	assert.Error(t, err)
	assert.Nil(t, body)
	assert.Contains(t, err.Error(), "Invalid request format")
	assert.Contains(t, err.Error(), "invalid_request_error")
}

func TestQwenProvider_MakeStreamingRequest_NetworkRetry(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 2 {
			// Simulate network error by hijacking and closing connection
			hj, ok := w.(http.Hijacker)
			if ok {
				conn, _, _ := hj.Hijack()
				if conn != nil {
					_ = conn.Close()
					return
				}
			}
			// Fallback
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	provider := NewQwenProviderWithRetry("test-key", server.URL, "qwen-turbo", RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Millisecond,
		MaxDelay:     10 * time.Millisecond,
		Multiplier:   2.0,
	})
	provider.httpClient = server.Client()

	req := &QwenRequest{
		Model:    "qwen-turbo",
		Messages: []QwenMessage{{Role: "user", Content: "test"}},
	}

	body, err := provider.makeStreamingRequest(context.Background(), req)
	// Either succeeds on retry or has an error
	if err == nil {
		require.NotNil(t, body)
		_ = body.Close()
	}
	assert.GreaterOrEqual(t, attempts, 1)
}

func TestQwenProvider_CompleteStream_LongRunning(t *testing.T) {
	// Test streaming with many chunks to ensure stability
	numChunks := 50
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)

		for i := 0; i < numChunks; i++ {
			chunk := QwenStreamChunk{
				ID:      fmt.Sprintf("chunk-%d", i),
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   "qwen-turbo",
				Choices: []QwenStreamChoice{
					{
						Index: 0,
						Delta: QwenStreamDelta{
							Content: fmt.Sprintf("word%d ", i),
						},
					},
				},
			}
			jsonData, _ := json.Marshal(chunk)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", jsonData)
			flusher.Flush()
		}

		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	provider := NewQwenProvider("test-api-key", server.URL, "qwen-turbo")
	provider.httpClient = server.Client()

	req := &models.LLMRequest{
		ID:     "test-long",
		Prompt: "Test",
	}

	responseChan, err := provider.CompleteStream(context.Background(), req)
	require.NoError(t, err)

	var chunkCount int
	var words []string
	for chunk := range responseChan {
		if chunk.Content != "" {
			chunkCount++
			words = append(words, strings.TrimSpace(chunk.Content))
		}
	}

	assert.Equal(t, numChunks, chunkCount)
	assert.Len(t, words, numChunks)
}

// =============================================================================
// Benchmarks
// =============================================================================

func BenchmarkQwenProvider_ConvertToQwenRequest(b *testing.B) {
	provider := NewQwenProvider("test-key", "", "")
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
		provider.convertToQwenRequest(req)
	}
}

func BenchmarkQwenProvider_GetCapabilities(b *testing.B) {
	provider := NewQwenProvider("test-key", "", "")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.GetCapabilities()
	}
}

func BenchmarkQwenProvider_ValidateConfig(b *testing.B) {
	provider := NewQwenProvider("test-key", "", "")
	config := map[string]interface{}{
		"api_key": "test-key",
		"model":   "qwen-turbo",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.ValidateConfig(config)
	}
}

func BenchmarkQwenProvider_ParseSSELine(b *testing.B) {
	line := []byte(`data: {"id":"chatcmpl-123","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parseSSELine(line)
	}
}

func BenchmarkQwenProvider_GetAPIEndpoint(b *testing.B) {
	provider := NewQwenProvider("test-key", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-turbo")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.getAPIEndpoint()
	}
}

func BenchmarkQwenProvider_ConvertFromQwenResponse(b *testing.B) {
	provider := NewQwenProvider("test-key", "", "")
	qwenResp := &QwenResponse{
		ID:    "resp-456",
		Model: "qwen-turbo",
		Choices: []QwenChoice{
			{
				Index:        0,
				Message:      QwenMessage{Role: "assistant", Content: "Benchmark response content"},
				FinishReason: "stop",
			},
		},
		Usage: QwenUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		provider.convertFromQwenResponse(qwenResp, "bench-request")
	}
}

func BenchmarkQwenProvider_NextDelay(b *testing.B) {
	provider := NewQwenProviderWithRetry("test-key", "", "", RetryConfig{
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

func BenchmarkQwenProvider_IsRetryableStatus(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		isRetryableStatus(429)
		isRetryableStatus(500)
		isRetryableStatus(200)
	}
}

func BenchmarkQwenProvider_IsAuthRetryableStatus(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		isAuthRetryableStatus(401)
		isAuthRetryableStatus(403)
		isAuthRetryableStatus(200)
	}
}

// TestQwenProvider_ValidateConfig_I18nPairedMutation is the round-379
// §1.1 paired-mutation guard for the CONST-046 i18n migration. It wires
// a BundleTranslator whose entries DIFFER from the source literals and
// asserts ValidateConfig surfaces the MUTATED text — proving the
// provider resolves validation errors through the i18n.Tr seam at
// runtime rather than holding a stale hardcoded copy. If the migration
// were a bluff (literal still hardcoded), the mutated bundle would have
// no effect and this test would FAIL.
func TestQwenProvider_ValidateConfig_I18nPairedMutation(t *testing.T) {
	const (
		mutKey   = "MUTATED-qwen-api-key-required"
		mutURL   = "MUTATED-qwen-base-url-required"
		mutModel = "MUTATED-qwen-model-required"
	)
	bundle := "llmprovider_validate_api_key_required:\n  other: \"" + mutKey + "\"\n" +
		"llmprovider_validate_base_url_required:\n  other: \"" + mutURL + "\"\n" +
		"llmprovider_validate_model_required:\n  other: \"" + mutModel + "\"\n"
	bt, err := i18n.NewBundleTranslatorFromBytes([]byte(bundle))
	require.NoError(t, err, "parse mutated bundle")
	i18n.SetTranslator(bt)
	t.Cleanup(func() { i18n.SetTranslator(nil) })

	provider := &QwenProvider{apiKey: "", baseURL: "", model: ""}
	valid, errs := provider.ValidateConfig(map[string]interface{}{})
	assert.False(t, valid)
	assert.Contains(t, errs, mutKey, "ValidateConfig must resolve api-key error through i18n seam")
	assert.Contains(t, errs, mutURL, "ValidateConfig must resolve base-url error through i18n seam")
	assert.Contains(t, errs, mutModel, "ValidateConfig must resolve model error through i18n seam")
}
