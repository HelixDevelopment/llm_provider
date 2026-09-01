package zai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"time"

	"digital.vasic.llmprovider/pkg/discovery"
	"digital.vasic.llmprovider/pkg/i18n"
	"digital.vasic.llmprovider/pkg/models"
	"digital.vasic.llmprovider/pkg/settings"
)

// RetryConfig defines retry behavior for API calls
type RetryConfig struct {
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

// DefaultRetryConfig returns sensible defaults for Z.AI API retry behavior
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

// ZAIEndpointChina is the China (mainland) API endpoint
const ZAIEndpointChina = "https://open.bigmodel.cn/api/paas/v4"

// ZAIEndpointInternational is the International API endpoint (for Coding Plan subscriptions)
const ZAIEndpointInternational = "https://api.z.ai/api/paas/v4"

// ZAIProvider implements the LLMProvider interface for Z.AI
type ZAIProvider struct {
	apiKey       string
	baseURL      string
	model        string
	httpClient   *http.Client
	streamClient *http.Client
	retryConfig  RetryConfig
	discoverer   *discovery.Discoverer
}

// newTunedTransport returns an *http.Transport tuned for repeated
// requests to a single API host: HTTP/2 negotiation is forced and the
// per-host idle-connection pool is widened so bursts of calls reuse
// warm TLS connections instead of paying a fresh handshake each time.
func newTunedTransport() *http.Transport {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.ForceAttemptHTTP2 = true
	t.MaxIdleConns = 100
	t.MaxIdleConnsPerHost = 32
	t.IdleConnTimeout = 90 * time.Second
	return t
}

// ZAIRequest represents a request to the Z.AI API
type ZAIRequest struct {
	Model       string                 `json:"model"`
	Prompt      string                 `json:"prompt,omitempty"`
	Messages    []ZAIMessage           `json:"messages,omitempty"`
	Stream      bool                   `json:"stream,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	TopP        float64                `json:"top_p,omitempty"`
	Stop        []string               `json:"stop,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
	Tools       []ZAITool              `json:"tools,omitempty"`
	ToolChoice  interface{}            `json:"tool_choice,omitempty"`
}

// ZAIMessage represents a message in the Z.AI API format
type ZAIMessage struct {
	Role      string        `json:"role"`
	Content   string        `json:"content"`
	ToolCalls []ZAIToolCall `json:"tool_calls,omitempty"`
}

// ZAIResponse represents a response from the Z.AI API
type ZAIResponse struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created"`
	Model   string      `json:"model"`
	Choices []ZAIChoice `json:"choices"`
	Usage   ZAIUsage    `json:"usage"`
}

// ZAIChoice represents a choice in the Z.AI response
type ZAIChoice struct {
	Index        int        `json:"index"`
	Text         string     `json:"text,omitempty"`
	Message      ZAIMessage `json:"message,omitempty"`
	FinishReason string     `json:"finish_reason"`
}

// ZAIUsage represents token usage in the Z.AI response
type ZAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ZAIError represents an error from the Z.AI API (Zhipu/GLM)
type ZAIError struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"` // Zhipu uses string codes like "1113", "1211"
	} `json:"error"`
}

// Zhipu API error codes
const (
	ZhipuErrInsufficientBalance = "1113" // 余额不足 (Insufficient balance)
	ZhipuErrModelNotFound       = "1211" // 模型不存在 (Model not found)
	ZhipuErrUnauthorized        = "401"  // 令牌已过期 (Token expired)
	ZhipuErrRateLimited         = "1301" // 请求频率过高 (Rate limited)
)

// ZAIStreamResponse represents a streaming response chunk from the Z.AI API
type ZAIStreamResponse struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []ZAIStreamChoice `json:"choices"`
}

// ZAIStreamChoice represents a choice in a streaming response
type ZAIStreamChoice struct {
	Index        int            `json:"index"`
	Delta        ZAIStreamDelta `json:"delta"`
	FinishReason *string        `json:"finish_reason"`
}

// ZAIStreamDelta represents the delta content in a streaming chunk
type ZAIStreamDelta struct {
	Role      string        `json:"role,omitempty"`
	Content   string        `json:"content,omitempty"`
	ToolCalls []ZAIToolCall `json:"tool_calls,omitempty"`
}

// ZAITool represents a tool definition for Z.AI API
type ZAITool struct {
	Type     string      `json:"type"`
	Function ZAIToolFunc `json:"function"`
}

// ZAIToolFunc represents the function definition within a tool
type ZAIToolFunc struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ZAIToolCall represents a tool call in the response
type ZAIToolCall struct {
	ID       string              `json:"id"`
	Type     string              `json:"type"`
	Function ZAIToolCallFunction `json:"function"`
}

// ZAIToolCallFunction represents the function call details
type ZAIToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// NewZAIProvider creates a new Z.AI provider instance
func NewZAIProvider(apiKey, baseURL, model string) *ZAIProvider {
	return NewZAIProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

// NewZAIProviderWithRetry creates a new Z.AI provider instance with custom retry config
// DefaultHTTPTimeout is the compiled fallback for this adapter's HTTP client.
// Override with LLMPROVIDER_ZAI_TIMEOUT (see pkg/settings).
const DefaultHTTPTimeout = 60 * time.Second

// DefaultZAIModel is the model used when neither the caller nor the
// environment names one. Previously an unnamed literal inside the
// constructor, which made it invisible to anyone reading the package API.
const DefaultZAIModel = "glm-4.5"

func NewZAIProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *ZAIProvider {
	if baseURL == "" {
		// FALLBACK, not a decision — LLMPROVIDER_ZAI_BASE_URL. An operator on
		// the China endpoint should not need a code change to reach it.
		baseURL = settings.BaseURL("zai", ZAIEndpointInternational)
	}
	if model == "" {
		// FALLBACK, not a decision — see pkg/settings.
		// LLMPROVIDER_ZAI_MODEL overrides it without a release.
		model = settings.Model("zai", DefaultZAIModel)
	}

	p := &ZAIProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			// LLMPROVIDER_ZAI_TIMEOUT.
			Timeout:   settings.Timeout("zai", DefaultHTTPTimeout),
			Transport: newTunedTransport(),
		},
		// Streaming responses have no overall deadline, but they still
		// benefit from the shared tuned transport's warm connection
		// pool — a per-call &http.Client{} pooled nothing.
		streamClient: &http.Client{
			Transport: newTunedTransport(),
		},
		retryConfig: retryConfig,
	}

	modelsEndpoint := baseURL + "/models"
	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "zai",
		ModelsEndpoint: modelsEndpoint,
		ModelsDevID:    "zhipu",
		APIKey:         apiKey,
		ResponseParser: discovery.ParseZAIModelsResponse,
		FallbackModels: []string{
			"glm-5",
			"glm-4.7",
			"glm-4.6",
			"glm-4.5",
			"glm-4.5-air",
			"glm-4-plus",
			"glm-4",
			"glm-4-air",
			"glm-4-airx",
			"glm-4-flash",
			"glm-4-flashx",
			"glm-4-long",
			"glm-4v",
			"glm-4v-plus",
		},
	})

	return p
}

// Complete implements the LLMProvider interface
func (z *ZAIProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	// Convert internal request to Z.AI format
	zaiReq := z.convertToZAIRequest(req)

	// Make API call
	resp, err := z.makeRequest(ctx, zaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to complete request: %w", err)
	}

	// Convert response back to internal format
	return z.convertFromZAIResponse(resp, req.ID)
}

// CompleteStream implements streaming completion for Z.AI
func (z *ZAIProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	// Convert internal request to Z.AI format with streaming enabled
	zaiReq := z.convertToZAIRequest(req)
	zaiReq.Stream = true

	// Make streaming API call
	resp, err := z.makeStreamingRequest(ctx, zaiReq)
	if err != nil {
		return nil, fmt.Errorf("Z.AI streaming API call failed: %w", err)
	}

	// Create response channel
	ch := make(chan *models.LLMResponse)

	go func() {
		defer func() { _ = resp.Body.Close() }()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		var fullContent string

		for {
			// Check context cancellation
			select {
			case <-ctx.Done():
				return
			default:
			}

			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				// Send error response and exit
				errorResp := &models.LLMResponse{
					ID:             "stream-error-" + req.ID,
					RequestID:      req.ID,
					ProviderID:     "zai",
					ProviderName:   "Z.AI",
					Content:        "",
					Confidence:     0.0,
					TokensUsed:     0,
					ResponseTime:   time.Since(startTime).Milliseconds(),
					FinishReason:   "error",
					Selected:       false,
					SelectionScore: 0.0,
					CreatedAt:      time.Now(),
				}
				ch <- errorResp
				return
			}

			// Skip empty lines and "data: " prefix
			line = bytes.TrimSpace(line)
			if !bytes.HasPrefix(line, []byte("data: ")) {
				continue
			}
			line = bytes.TrimPrefix(line, []byte("data: "))

			// Skip "[DONE]" marker
			if bytes.Equal(line, []byte("[DONE]")) {
				break
			}

			// Parse JSON
			var streamResp ZAIStreamResponse
			if err := json.Unmarshal(line, &streamResp); err != nil {
				continue // Skip malformed JSON
			}

			// Extract content
			if len(streamResp.Choices) > 0 {
				delta := streamResp.Choices[0].Delta.Content
				if delta != "" {
					fullContent += delta

					// Send chunk response
					chunkResp := &models.LLMResponse{
						ID:             streamResp.ID,
						RequestID:      req.ID,
						ProviderID:     "zai",
						ProviderName:   "Z.AI",
						Content:        delta,
						Confidence:     0.80, // Default confidence for streaming
						TokensUsed:     1,    // Estimated per chunk
						ResponseTime:   time.Since(startTime).Milliseconds(),
						FinishReason:   "",
						Selected:       false,
						SelectionScore: 0.0,
						Metadata: map[string]interface{}{
							"model":        streamResp.Model,
							"stream_chunk": true,
						},
						CreatedAt: time.Now(),
					}
					ch <- chunkResp
				}

				// Check if stream is finished
				if streamResp.Choices[0].FinishReason != nil {
					// Send final response with complete content
					finalResp := &models.LLMResponse{
						ID:             streamResp.ID,
						RequestID:      req.ID,
						ProviderID:     "zai",
						ProviderName:   "Z.AI",
						Content:        fullContent,
						Confidence:     0.80,
						TokensUsed:     len(fullContent) / 4, // Rough token estimate
						ResponseTime:   time.Since(startTime).Milliseconds(),
						FinishReason:   *streamResp.Choices[0].FinishReason,
						Selected:       false,
						SelectionScore: 0.0,
						Metadata: map[string]interface{}{
							"model":           streamResp.Model,
							"stream_complete": true,
						},
						CreatedAt: time.Now(),
					}
					ch <- finalResp
					break
				}
			}
		}
	}()

	return ch, nil
}

// makeStreamingRequest sends a streaming request to the Z.AI API
func (z *ZAIProvider) makeStreamingRequest(ctx context.Context, req *ZAIRequest) (*http.Response, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Determine endpoint based on request type
	endpoint := "/completions"
	if len(req.Messages) > 0 {
		endpoint = "/chat/completions"
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", z.baseURL+endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Bearer "+z.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	// Use the shared no-timeout streaming client (tuned transport,
	// warm connection pool) rather than a fresh per-call client.
	resp, err := z.streamClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("Zhipu GLM API error: %d (failed to read body: %v)", resp.StatusCode, readErr)
		}
		var zaiErr ZAIError
		if err := json.Unmarshal(body, &zaiErr); err == nil && zaiErr.Error.Message != "" {
			// Handle Zhipu-specific error codes
			switch zaiErr.Error.Code {
			case ZhipuErrInsufficientBalance:
				return nil, fmt.Errorf("Zhipu GLM API error: insufficient balance - please recharge your account")
			case ZhipuErrModelNotFound:
				return nil, fmt.Errorf("Zhipu GLM API error: model not found")
			case ZhipuErrUnauthorized:
				return nil, fmt.Errorf("Zhipu GLM API error: API key expired or invalid")
			case ZhipuErrRateLimited:
				return nil, fmt.Errorf("Zhipu GLM API error: rate limited")
			default:
				return nil, fmt.Errorf("Zhipu GLM API error [%s]: %s", zaiErr.Error.Code, zaiErr.Error.Message)
			}
		}
		return nil, fmt.Errorf("Zhipu GLM API returned status %d: %s", resp.StatusCode, string(body))
	}

	return resp, nil
}

// HealthCheck implements health checking for the Z.AI provider
func (z *ZAIProvider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Simple health check - try to get models list or basic endpoint
	req, err := http.NewRequestWithContext(ctx, "GET", z.baseURL+"/models", nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+z.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := z.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}

	return nil
}

// GetCapabilities returns the capabilities of the Z.AI provider
func (z *ZAIProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: z.discoverer.DiscoverModels(),
		SupportedFeatures: []string{
			"text_completion",
			"chat",
			"function_calling",
			"code_generation",
			"reasoning",
		},
		SupportedRequestTypes: []string{
			"text_completion",
			"chat",
		},
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          true, // GLM-4V supports vision
		SupportsTools:           true,
		Limits: models.ModelLimits{
			MaxTokens:             8192,
			MaxInputLength:        128000, // GLM-4 supports 128K context
			MaxOutputLength:       8192,
			MaxConcurrentRequests: 20,
		},
		Metadata: map[string]string{
			"provider":     "Zhipu AI (GLM)",
			"model_family": "GLM-4",
			"api_version":  "v4",
		},
	}
}

// ValidateConfig validates the provider configuration
func (z *ZAIProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string

	if z.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}

	if z.baseURL == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_base_url_required", nil))
	}

	if z.model == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_model_required", nil))
	}

	return len(errors) == 0, errors
}

// convertToZAIRequest converts internal request format to Z.AI API format
func (z *ZAIProvider) convertToZAIRequest(req *models.LLMRequest) *ZAIRequest {
	zaiReq := &ZAIRequest{
		Model:       z.model,
		Stream:      false,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   req.ModelParams.MaxTokens,
		TopP:        req.ModelParams.TopP,
		Stop:        req.ModelParams.StopSequences,
		Parameters:  make(map[string]interface{}),
	}

	// Handle different request types
	if len(req.Messages) > 0 {
		// Chat format
		messages := make([]ZAIMessage, 0, len(req.Messages))
		for _, msg := range req.Messages {
			messages = append(messages, ZAIMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
		zaiReq.Messages = messages
	} else {
		// Completion format
		zaiReq.Prompt = req.Prompt
	}

	// Convert tools if provided
	if len(req.Tools) > 0 {
		zaiTools := make([]ZAITool, 0, len(req.Tools))
		for _, tool := range req.Tools {
			zaiTools = append(zaiTools, ZAITool{
				Type: tool.Type,
				Function: ZAIToolFunc{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			})
		}
		zaiReq.Tools = zaiTools
		zaiReq.ToolChoice = "auto"
	}

	return zaiReq
}

// convertFromZAIResponse converts Z.AI API response to internal format
func (z *ZAIProvider) convertFromZAIResponse(resp *ZAIResponse, requestID string) (*models.LLMResponse, error) {
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned from Z.AI API")
	}

	choice := resp.Choices[0]

	// Extract content from either text or message field
	var content string
	if choice.Text != "" {
		content = choice.Text
	} else if choice.Message.Content != "" {
		content = choice.Message.Content
	}
	// Note: Content may be empty if only tool_calls are returned

	llmResp := &models.LLMResponse{
		ID:           resp.ID,
		RequestID:    requestID,
		ProviderID:   "zai",
		ProviderName: "Z.AI",
		Content:      content,
		Confidence:   0.80, // Z.AI doesn't provide confidence scores
		TokensUsed:   resp.Usage.TotalTokens,
		ResponseTime: time.Now().UnixMilli() - (resp.Created * 1000),
		FinishReason: choice.FinishReason,
		Metadata: map[string]interface{}{
			"model":             resp.Model,
			"object":            resp.Object,
			"prompt_tokens":     resp.Usage.PromptTokens,
			"completion_tokens": resp.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}

	// Convert tool calls if present
	if len(choice.Message.ToolCalls) > 0 {
		toolCalls := make([]models.ToolCall, 0, len(choice.Message.ToolCalls))
		for _, tc := range choice.Message.ToolCalls {
			toolCalls = append(toolCalls, models.ToolCall{
				ID:   tc.ID,
				Type: tc.Type,
				Function: models.ToolCallFunction{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			})
		}
		llmResp.ToolCalls = toolCalls
		// Update finish_reason if tool_calls are present
		if choice.FinishReason == "" || choice.FinishReason == "stop" {
			llmResp.FinishReason = "tool_calls"
		}
	} else if content == "" {
		return nil, fmt.Errorf("no content or tool_calls found in Z.AI response")
	}

	return llmResp, nil
}

// makeRequest sends a request to the Z.AI API with retry logic
func (z *ZAIProvider) makeRequest(ctx context.Context, req *ZAIRequest) (*ZAIResponse, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Determine endpoint based on request type
	endpoint := "/completions"
	if len(req.Messages) > 0 {
		endpoint = "/chat/completions"
	}

	var lastErr error
	delay := z.retryConfig.InitialDelay

	for attempt := 0; attempt <= z.retryConfig.MaxRetries; attempt++ {
		// Check context before making request
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
		default:
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", z.baseURL+endpoint, bytes.NewBuffer(jsonData))
		if err != nil {
			return nil, fmt.Errorf("failed to create HTTP request: %w", err)
		}

		httpReq.Header.Set("Authorization", "Bearer "+z.apiKey)
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := z.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("HTTP request failed: %w", err)
			if attempt < z.retryConfig.MaxRetries {
				z.waitWithJitter(ctx, delay)
				delay = z.nextDelay(delay)
				continue
			}
			return nil, lastErr
		}

		// Check for retryable status codes
		if isRetryableStatus(resp.StatusCode) && attempt < z.retryConfig.MaxRetries {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("HTTP %d: retryable error", resp.StatusCode)
			z.waitWithJitter(ctx, delay)
			delay = z.nextDelay(delay)
			continue
		}

		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %w", err)
		}

		if resp.StatusCode != http.StatusOK {
			var zaiErr ZAIError
			if err := json.Unmarshal(body, &zaiErr); err == nil && zaiErr.Error.Message != "" {
				// Handle Zhipu-specific error codes with clear messages
				switch zaiErr.Error.Code {
				case ZhipuErrInsufficientBalance:
					return nil, fmt.Errorf("Zhipu GLM API error: insufficient balance - please recharge your account at https://open.bigmodel.cn")
				case ZhipuErrModelNotFound:
					return nil, fmt.Errorf("Zhipu GLM API error: model '%s' not found - check available models at https://open.bigmodel.cn", z.model)
				case ZhipuErrUnauthorized:
					return nil, fmt.Errorf("Zhipu GLM API error: API key expired or invalid")
				case ZhipuErrRateLimited:
					return nil, fmt.Errorf("Zhipu GLM API error: rate limited - too many requests")
				default:
					return nil, fmt.Errorf("Zhipu GLM API error [%s]: %s", zaiErr.Error.Code, zaiErr.Error.Message)
				}
			}
			return nil, fmt.Errorf("Zhipu GLM API returned status %d: %s", resp.StatusCode, string(body))
		}

		var zaiResp ZAIResponse
		if err := json.Unmarshal(body, &zaiResp); err != nil {
			return nil, fmt.Errorf("failed to unmarshal response: %w", err)
		}

		return &zaiResp, nil
	}

	return nil, fmt.Errorf("all %d retry attempts failed: %w", z.retryConfig.MaxRetries+1, lastErr)
}

// isRetryableStatus returns true for HTTP status codes that warrant a retry
func isRetryableStatus(statusCode int) bool {
	switch statusCode {
	case http.StatusTooManyRequests, // 429 - Rate limited
		http.StatusInternalServerError, // 500
		http.StatusBadGateway,          // 502
		http.StatusServiceUnavailable,  // 503
		http.StatusGatewayTimeout:      // 504
		return true
	default:
		return false
	}
}

// waitWithJitter waits for the specified duration plus random jitter
func (z *ZAIProvider) waitWithJitter(ctx context.Context, delay time.Duration) {
	// Add 10% jitter - using math/rand is acceptable for non-security jitter
	jitter := time.Duration(rand.Float64() * 0.1 * float64(delay)) // #nosec G404 - jitter doesn't require cryptographic randomness
	select {
	case <-ctx.Done():
	case <-time.After(delay + jitter):
	}
}

// nextDelay calculates the next delay using exponential backoff
func (z *ZAIProvider) nextDelay(currentDelay time.Duration) time.Duration {
	nextDelay := time.Duration(float64(currentDelay) * z.retryConfig.Multiplier)
	if nextDelay > z.retryConfig.MaxDelay {
		nextDelay = z.retryConfig.MaxDelay
	}
	return nextDelay
}
