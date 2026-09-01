package qwen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"digital.vasic.llmprovider/pkg/discovery"
	"digital.vasic.llmprovider/pkg/i18n"
	"digital.vasic.llmprovider/pkg/models"
	"digital.vasic.llmprovider/pkg/settings"
)

// Settings keys, and the compiled fallbacks behind them.
//
// The two auth paths do NOT talk to the same endpoint — the API-key path posts
// to DashScope's native API, the OAuth path to its OpenAI-compatible shim — so
// they get two BASE_URL keys. Sharing one would mean an operator redirecting
// their OAuth traffic would silently redirect API-key traffic to a path that
// speaks a different wire format. The MODEL is genuinely shared, so it stays on
// one key.
const (
	// SettingsProvider keys the API-key endpoint, the shared model and the
	// shared timeout: LLMPROVIDER_QWEN_BASE_URL, LLMPROVIDER_QWEN_MODEL,
	// LLMPROVIDER_QWEN_TIMEOUT.
	SettingsProvider = "qwen"
	// SettingsProviderOAuth keys the OAuth endpoint:
	// LLMPROVIDER_QWEN_OAUTH_BASE_URL.
	SettingsProviderOAuth = "qwen_oauth"

	// QwenAPIURL is DashScope's native API, used with an API key.
	QwenAPIURL = "https://dashscope.aliyuncs.com/api/v1"
	// QwenOAuthAPIURL is DashScope's OpenAI-compatible shim, which is what the
	// Qwen Code CLI's OAuth tokens are accepted against.
	QwenOAuthAPIURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	// QwenModel is the compiled fallback model for both transports.
	QwenModel = "qwen-turbo"
)

// DefaultHTTPTimeout is the compiled fallback for this adapter's HTTP client.
// It is a starting point, not a decision the library keeps making:
// LLMPROVIDER_QWEN_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 60 * time.Second

// AuthType represents the type of authentication used
type AuthType string

const (
	AuthTypeAPIKey AuthType = "api_key"
	AuthTypeOAuth  AuthType = "oauth"
)

// RetryConfig defines retry behavior for API calls
type RetryConfig struct {
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

// DefaultRetryConfig returns sensible defaults for Qwen API retry behavior
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

// QwenProvider implements the LLMProvider interface for Alibaba Cloud Qwen
// QwenOAuthCredentialReader is an interface for reading Qwen OAuth credentials.
type QwenOAuthCredentialReader interface {
	HasValidQwenCredentials() bool
	GetQwenAccessToken() (string, error)
}

type QwenProvider struct {
	apiKey          string
	baseURL         string
	model           string
	httpClient      *http.Client
	retryConfig     RetryConfig
	authType        AuthType
	oauthCredReader QwenOAuthCredentialReader
	discoverer      *discovery.Discoverer
}

// QwenRequest represents a request to the Qwen API
type QwenRequest struct {
	Model       string        `json:"model"`
	Messages    []QwenMessage `json:"messages"`
	Stream      bool          `json:"stream,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	TopP        float64       `json:"top_p,omitempty"`
	Stop        []string      `json:"stop,omitempty"`
	Tools       []QwenTool    `json:"tools,omitempty"`
	ToolChoice  interface{}   `json:"tool_choice,omitempty"`
}

// QwenMessage represents a message in the Qwen API format
type QwenMessage struct {
	Role      string         `json:"role"`
	Content   string         `json:"content"`
	ToolCalls []QwenToolCall `json:"tool_calls,omitempty"`
}

// QwenResponse represents a response from the Qwen API
type QwenResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []QwenChoice `json:"choices"`
	Usage   QwenUsage    `json:"usage"`
}

// QwenChoice represents a choice in the Qwen response
type QwenChoice struct {
	Index        int         `json:"index"`
	Message      QwenMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// QwenUsage represents token usage in the Qwen response
type QwenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// QwenError represents an error from the Qwen API
type QwenError struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

// QwenStreamChunk represents a streaming chunk from the Qwen API
type QwenStreamChunk struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []QwenStreamChoice `json:"choices"`
}

// QwenStreamChoice represents a choice in a streaming response
type QwenStreamChoice struct {
	Index        int             `json:"index"`
	Delta        QwenStreamDelta `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
}

// QwenStreamDelta represents the delta content in a streaming response
type QwenStreamDelta struct {
	Role      string         `json:"role,omitempty"`
	Content   string         `json:"content,omitempty"`
	ToolCalls []QwenToolCall `json:"tool_calls,omitempty"`
}

// QwenTool represents a tool definition for Qwen API
type QwenTool struct {
	Type     string       `json:"type"`
	Function QwenToolFunc `json:"function"`
}

// QwenToolFunc represents the function definition within a tool
type QwenToolFunc struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// QwenToolCall represents a tool call in the response
type QwenToolCall struct {
	ID       string               `json:"id"`
	Type     string               `json:"type"`
	Function QwenToolCallFunction `json:"function"`
}

// QwenToolCallFunction represents the function call details
type QwenToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// NewQwenProvider creates a new Qwen provider instance
func NewQwenProvider(apiKey, baseURL, model string) *QwenProvider {
	return NewQwenProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

// NewQwenProviderWithRetry creates a new Qwen provider instance with custom retry config
func NewQwenProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *QwenProvider {
	if baseURL == "" {
		// LLMPROVIDER_QWEN_BASE_URL.
		baseURL = settings.BaseURL(SettingsProvider, QwenAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_QWEN_MODEL — shared with the OAuth path by design.
		model = settings.Model(SettingsProvider, QwenModel)
	}

	p := &QwenProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			// LLMPROVIDER_QWEN_TIMEOUT.
			Timeout: settings.Timeout(SettingsProvider, DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
		authType:    AuthTypeAPIKey,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "qwen",
		ModelsEndpoint: "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
		ModelsDevID:    "alibaba",
		APIKey:         apiKey,
		FallbackModels: []string{
			"qwen-turbo",
			"qwen-plus",
			"qwen-max",
			"qwen-max-longcontext",
		},
	})

	return p
}

// NewQwenProviderWithOAuth creates a new Qwen provider using OAuth credentials from Qwen Code CLI
func NewQwenProviderWithOAuth(baseURL, model string) (*QwenProvider, error) {
	return NewQwenProviderWithOAuthAndRetry(baseURL, model, DefaultRetryConfig())
}

// NewQwenProviderWithOAuthAndRetry creates a new Qwen provider using OAuth credentials with custom retry config
func NewQwenProviderWithOAuthAndRetry(baseURL, model string, retryConfig RetryConfig) (*QwenProvider, error) {
	// OAuth tokens from Qwen Code CLI work with the DashScope API
	if baseURL == "" {
		// LLMPROVIDER_QWEN_OAUTH_BASE_URL — deliberately NOT the API-key path's
		// key; see SettingsProviderOAuth.
		baseURL = settings.BaseURL(SettingsProviderOAuth, QwenOAuthAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_QWEN_MODEL — shared with the API-key path by design.
		model = settings.Model(SettingsProvider, QwenModel)
	}

	// OAuth not available in standalone module

	// Verify credentials are available
	if false { // OAuth stub
		return nil, fmt.Errorf("no valid Qwen OAuth credentials available: ensure you are logged in via Qwen Code CLI")
	}

	p := &QwenProvider{
		apiKey:  "", // Will use OAuth token instead
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			// LLMPROVIDER_QWEN_TIMEOUT.
			Timeout: settings.Timeout(SettingsProvider, DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
		authType:    AuthTypeOAuth,
		// oauthCredReader: set via constructor
	}

	// OAuth providers have no API key, so Tier 1 is skipped (APIKey is empty).
	// Discovery will fall through to Tier 2 (models.dev) or Tier 3 (fallback).
	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "qwen",
		ModelsEndpoint: "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
		ModelsDevID:    "alibaba",
		APIKey:         "",
		FallbackModels: []string{
			"qwen-turbo",
			"qwen-plus",
			"qwen-max",
			"qwen-max-longcontext",
		},
	})

	return p, nil
}

// NewQwenProviderAuto creates a Qwen provider using API key authentication.
// OAuth is not available in the standalone module.
func NewQwenProviderAuto(apiKey, baseURL, model string) (*QwenProvider, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("no API key provided")
	}
	return NewQwenProvider(apiKey, baseURL, model), nil
}

// GetAuthType returns the authentication type being used
func (q *QwenProvider) GetAuthType() AuthType {
	return q.authType
}

// getAPIEndpoint returns the correct API endpoint based on the base URL
// OpenAI-compatible mode uses /chat/completions, DashScope native uses /services/aigc/text-generation/generation
func (q *QwenProvider) getAPIEndpoint() string {
	if strings.Contains(q.baseURL, "compatible-mode") {
		return q.baseURL + "/chat/completions"
	}
	return q.baseURL + "/services/aigc/text-generation/generation"
}

// getAuthHeader returns the appropriate authorization header based on auth type
func (q *QwenProvider) getAuthHeader() (string, error) {
	switch q.authType {
	case AuthTypeOAuth:
		if q.oauthCredReader == nil {
			return "", fmt.Errorf("OAuth credential reader not initialized")
		}
		token, err := q.oauthCredReader.GetQwenAccessToken()
		if err != nil {
			return "", fmt.Errorf("failed to get OAuth token: %w", err)
		}
		return "Bearer " + token, nil
	default:
		return "Bearer " + q.apiKey, nil
	}
}

// Complete implements the LLMProvider interface
func (q *QwenProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	// Convert internal request to Qwen format
	qwenReq := q.convertToQwenRequest(req)

	// Make API call
	resp, err := q.makeRequest(ctx, qwenReq)
	if err != nil {
		return nil, fmt.Errorf("failed to complete request: %w", err)
	}

	// Convert response back to internal format
	return q.convertFromQwenResponse(resp, req.ID)
}

// CompleteStream implements streaming completion for Qwen using real SSE streaming
func (q *QwenProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	// Convert internal request to Qwen format with streaming enabled
	qwenReq := q.convertToQwenRequest(req)
	qwenReq.Stream = true

	// Make streaming API call
	body, err := q.makeStreamingRequest(ctx, qwenReq)
	if err != nil {
		return nil, fmt.Errorf("failed to start streaming request: %w", err)
	}

	// Create response channel
	responseChan := make(chan *models.LLMResponse, 10)

	go func() {
		defer func() { _ = body.Close() }()
		defer close(responseChan)

		reader := bufio.NewReader(body)
		var chunkIndex int
		var totalTokens int

		for {
			// Check for context cancellation
			select {
			case <-ctx.Done():
				return
			default:
			}

			// Read a line from the SSE stream
			line, err := reader.ReadBytes('\n')
			// On io.EOF, ReadBytes may still return a final partial line (the last
			// data chunk without a trailing newline). Process it before breaking.
			eof := err == io.EOF
			if err != nil && !eof {
				// Send error response for read errors
				errorResp := &models.LLMResponse{
					ID:           fmt.Sprintf("%s-error-%d", req.ID, chunkIndex),
					RequestID:    req.ID,
					ProviderID:   "qwen",
					ProviderName: "Qwen",
					Content:      "",
					Confidence:   0.0,
					TokensUsed:   totalTokens,
					ResponseTime: time.Since(startTime).Milliseconds(),
					FinishReason: "error",
					Metadata: map[string]interface{}{
						"error": err.Error(),
					},
					CreatedAt: time.Now(),
				}
				select {
				case responseChan <- errorResp:
				case <-ctx.Done():
				}
				return
			}

			// Parse the SSE line
			chunk, done, parseErr := parseSSELine(line)

			// Handle [DONE] marker
			if done {
				// Send final response indicating stream completion
				finalResp := &models.LLMResponse{
					ID:           fmt.Sprintf("%s-final", req.ID),
					RequestID:    req.ID,
					ProviderID:   "qwen",
					ProviderName: "Qwen",
					Content:      "",
					Confidence:   0.85,
					TokensUsed:   totalTokens,
					ResponseTime: time.Since(startTime).Milliseconds(),
					FinishReason: "stop",
					CreatedAt:    time.Now(),
				}
				select {
				case responseChan <- finalResp:
				case <-ctx.Done():
				}
				return
			}

			// Skip empty lines or non-data lines
			if chunk == nil {
				// If there was a parse error, log it but continue
				if parseErr != nil {
					// Skip malformed JSON lines silently
					if eof {
						break
					}
					continue
				}
				if eof {
					break
				}
				continue
			}

			// Extract content from the chunk
			if len(chunk.Choices) > 0 {
				delta := chunk.Choices[0].Delta.Content
				if delta != "" {
					// Estimate tokens (rough approximation)
					estimatedTokens := len(strings.Fields(delta))
					if estimatedTokens == 0 && len(delta) > 0 {
						estimatedTokens = 1
					}
					totalTokens += estimatedTokens

					// Build the response
					streamResp := &models.LLMResponse{
						ID:           chunk.ID,
						RequestID:    req.ID,
						ProviderID:   "qwen",
						ProviderName: "Qwen",
						Content:      delta,
						Confidence:   0.85,
						TokensUsed:   estimatedTokens,
						ResponseTime: time.Since(startTime).Milliseconds(),
						FinishReason: "",
						Metadata: map[string]interface{}{
							"model":       chunk.Model,
							"chunk_index": chunkIndex,
						},
						CreatedAt: time.Now(),
					}

					select {
					case responseChan <- streamResp:
						chunkIndex++
					case <-ctx.Done():
						return
					}
				}

				// Check if stream is finished based on finish_reason
				if chunk.Choices[0].FinishReason != nil {
					finishReason := *chunk.Choices[0].FinishReason
					if finishReason != "" {
						// Send final response with the finish reason
						finalResp := &models.LLMResponse{
							ID:           fmt.Sprintf("%s-final", chunk.ID),
							RequestID:    req.ID,
							ProviderID:   "qwen",
							ProviderName: "Qwen",
							Content:      "",
							Confidence:   0.85,
							TokensUsed:   totalTokens,
							ResponseTime: time.Since(startTime).Milliseconds(),
							FinishReason: finishReason,
							CreatedAt:    time.Now(),
						}
						select {
						case responseChan <- finalResp:
						case <-ctx.Done():
						}
						return
					}
				}
			}

			if eof {
				break
			}
		}

		// If we exit the loop without a [DONE] or finish_reason, send a final response
		finalResp := &models.LLMResponse{
			ID:           fmt.Sprintf("%s-final", req.ID),
			RequestID:    req.ID,
			ProviderID:   "qwen",
			ProviderName: "Qwen",
			Content:      "",
			Confidence:   0.85,
			TokensUsed:   totalTokens,
			ResponseTime: time.Since(startTime).Milliseconds(),
			FinishReason: "stop",
			CreatedAt:    time.Now(),
		}
		select {
		case responseChan <- finalResp:
		case <-ctx.Done():
		}
	}()

	return responseChan, nil
}

// HealthCheck implements health checking for the Qwen provider
func (q *QwenProvider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Simple health check - try to get models list
	req, err := http.NewRequestWithContext(ctx, "GET", q.baseURL+"/models", nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	// Set authentication header
	authHeader, authErr := q.getAuthHeader()
	if authErr != nil {
		return fmt.Errorf("failed to get auth header: %w", authErr)
	}
	req.Header.Set("Authorization", authHeader)
	req.Header.Set("Content-Type", "application/json")

	resp, err := q.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}

	return nil
}

// GetCapabilities returns the capabilities of the Qwen provider
func (q *QwenProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: q.discoverer.DiscoverModels(),
		SupportedFeatures: []string{
			"text_completion",
			"chat",
			"function_calling",
		},
		SupportedRequestTypes: []string{
			"text_completion",
			"chat",
		},
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          false,
		SupportsTools:           true,
		SupportsSearch:          false,
		SupportsReasoning:       true,
		SupportsCodeCompletion:  true,
		SupportsCodeAnalysis:    true,
		SupportsRefactoring:     false,
		Limits: models.ModelLimits{
			MaxTokens:             6000,
			MaxInputLength:        30000,
			MaxOutputLength:       2000,
			MaxConcurrentRequests: 50,
		},
		Metadata: map[string]string{
			"provider":     "Alibaba Cloud",
			"model_family": "Qwen",
			"api_version":  "v1",
		},
	}
}

// ValidateConfig validates the provider configuration
func (q *QwenProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string

	// For OAuth auth, we don't need API key - check OAuth credentials instead
	if q.authType == AuthTypeOAuth {
		if q.oauthCredReader == nil {
			errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_oauth_reader_required", nil))
		} else if !q.oauthCredReader.HasValidQwenCredentials() {
			errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_oauth_creds_invalid", nil))
		}
	} else {
		// API key auth
		if q.apiKey == "" {
			errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
		}
	}

	if q.baseURL == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_base_url_required", nil))
	}

	if q.model == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_model_required", nil))
	}

	return len(errors) == 0, errors
}

// convertToQwenRequest converts internal request format to Qwen API format
func (q *QwenProvider) convertToQwenRequest(req *models.LLMRequest) *QwenRequest {
	messages := make([]QwenMessage, 0, len(req.Messages))

	// Add system message if present in prompt
	if req.Prompt != "" {
		messages = append(messages, QwenMessage{
			Role:    "system",
			Content: req.Prompt,
		})
	}

	// Convert internal messages
	for _, msg := range req.Messages {
		messages = append(messages, QwenMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	qwenReq := &QwenRequest{
		Model:       q.model,
		Messages:    messages,
		Stream:      false,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   req.ModelParams.MaxTokens,
		TopP:        req.ModelParams.TopP,
		Stop:        req.ModelParams.StopSequences,
	}

	// Convert tools if provided
	if len(req.Tools) > 0 {
		qwenTools := make([]QwenTool, 0, len(req.Tools))
		for _, tool := range req.Tools {
			qwenTools = append(qwenTools, QwenTool{
				Type: tool.Type,
				Function: QwenToolFunc{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			})
		}
		qwenReq.Tools = qwenTools
		qwenReq.ToolChoice = "auto"
	}

	return qwenReq
}

// convertFromQwenResponse converts Qwen API response to internal format
func (q *QwenProvider) convertFromQwenResponse(resp *QwenResponse, requestID string) (*models.LLMResponse, error) {
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned from Qwen API")
	}

	choice := resp.Choices[0]

	llmResp := &models.LLMResponse{
		ID:           resp.ID,
		RequestID:    requestID,
		ProviderID:   "qwen",
		ProviderName: "Qwen",
		Content:      choice.Message.Content,
		Confidence:   0.85, // Qwen doesn't provide confidence scores
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
	}

	return llmResp, nil
}

// makeRequest sends a request to the Qwen API with retry logic
func (q *QwenProvider) makeRequest(ctx context.Context, req *QwenRequest) (*QwenResponse, error) {
	return q.makeRequestWithAuthRetry(ctx, req, true)
}

// makeRequestWithAuthRetry sends a request to the Qwen API with optional 401 retry
func (q *QwenProvider) makeRequestWithAuthRetry(ctx context.Context, req *QwenRequest, allowAuthRetry bool) (*QwenResponse, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var lastErr error
	delay := q.retryConfig.InitialDelay

	for attempt := 0; attempt <= q.retryConfig.MaxRetries; attempt++ {
		// Check context before making request
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
		default:
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", q.getAPIEndpoint(), bytes.NewBuffer(jsonData))
		if err != nil {
			return nil, fmt.Errorf("failed to create HTTP request: %w", err)
		}

		// Set authentication header
		authHeader, authErr := q.getAuthHeader()
		if authErr != nil {
			return nil, fmt.Errorf("failed to get auth header: %w", authErr)
		}
		httpReq.Header.Set("Authorization", authHeader)
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := q.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("HTTP request failed: %w", err)
			if attempt < q.retryConfig.MaxRetries {
				q.waitWithJitter(ctx, delay)
				delay = q.nextDelay(delay)
				continue
			}
			return nil, lastErr
		}

		// Check for auth errors (401) - retry once with a short delay
		// This handles transient auth issues (token validation delays, auth service hiccups)
		if isAuthRetryableStatus(resp.StatusCode) && allowAuthRetry {
			_ = resp.Body.Close()
			// Short delay before auth retry (500ms with jitter)
			authRetryDelay := 500 * time.Millisecond
			q.waitWithJitter(ctx, authRetryDelay)
			// Recursive call with auth retry disabled to prevent infinite loops
			return q.makeRequestWithAuthRetry(ctx, req, false)
		}

		// Check for retryable status codes (429, 5xx)
		if isRetryableStatus(resp.StatusCode) && attempt < q.retryConfig.MaxRetries {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("HTTP %d: retryable error", resp.StatusCode)
			q.waitWithJitter(ctx, delay)
			delay = q.nextDelay(delay)
			continue
		}

		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %w", err)
		}

		if resp.StatusCode != http.StatusOK {
			var qwenErr QwenError
			if err := json.Unmarshal(body, &qwenErr); err == nil && qwenErr.Error.Message != "" {
				return nil, fmt.Errorf("Qwen API error: %s (%s)", qwenErr.Error.Message, qwenErr.Error.Type)
			}
			return nil, fmt.Errorf("Qwen API returned status %d: %s", resp.StatusCode, string(body))
		}

		var qwenResp QwenResponse
		if err := json.Unmarshal(body, &qwenResp); err != nil {
			return nil, fmt.Errorf("failed to unmarshal response: %w", err)
		}

		return &qwenResp, nil
	}

	return nil, fmt.Errorf("all %d retry attempts failed: %w", q.retryConfig.MaxRetries+1, lastErr)
}

// makeStreamingRequest sends a streaming request to the Qwen API with retry logic
// It returns an io.ReadCloser for reading SSE events
func (q *QwenProvider) makeStreamingRequest(ctx context.Context, req *QwenRequest) (io.ReadCloser, error) {
	// Ensure streaming is enabled
	req.Stream = true

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var lastErr error
	delay := q.retryConfig.InitialDelay

	for attempt := 0; attempt <= q.retryConfig.MaxRetries; attempt++ {
		// Check context before making request
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
		default:
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", q.getAPIEndpoint(), bytes.NewBuffer(jsonData))
		if err != nil {
			return nil, fmt.Errorf("failed to create HTTP request: %w", err)
		}

		// Set authentication header
		authHeader, authErr := q.getAuthHeader()
		if authErr != nil {
			return nil, fmt.Errorf("failed to get auth header: %w", authErr)
		}
		httpReq.Header.Set("Authorization", authHeader)
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")
		httpReq.Header.Set("Cache-Control", "no-cache")
		httpReq.Header.Set("Connection", "keep-alive")

		resp, err := q.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("HTTP request failed: %w", err)
			if attempt < q.retryConfig.MaxRetries {
				q.waitWithJitter(ctx, delay)
				delay = q.nextDelay(delay)
				continue
			}
			return nil, lastErr
		}

		// Check for retryable status codes
		if isRetryableStatus(resp.StatusCode) && attempt < q.retryConfig.MaxRetries {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("HTTP %d: retryable error", resp.StatusCode)
			q.waitWithJitter(ctx, delay)
			delay = q.nextDelay(delay)
			continue
		}

		// Check for non-OK status codes (non-retryable errors)
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body) //nolint:errcheck
			_ = resp.Body.Close()            //nolint:errcheck
			var qwenErr QwenError
			if err := json.Unmarshal(body, &qwenErr); err == nil && qwenErr.Error.Message != "" {
				return nil, fmt.Errorf("Qwen API error: %s (%s)", qwenErr.Error.Message, qwenErr.Error.Type)
			}
			return nil, fmt.Errorf("Qwen API returned status %d: %s", resp.StatusCode, string(body))
		}

		// Return the body for streaming
		return resp.Body, nil
	}

	return nil, fmt.Errorf("all %d retry attempts failed: %w", q.retryConfig.MaxRetries+1, lastErr)
}

// parseSSELine parses an SSE data line and returns the QwenStreamChunk
// Returns nil if the line is not a valid data line or is the [DONE] marker
func parseSSELine(line []byte) (*QwenStreamChunk, bool, error) {
	// Trim whitespace
	line = bytes.TrimSpace(line)

	// Skip empty lines
	if len(line) == 0 {
		return nil, false, nil
	}

	// Check for data prefix
	if !bytes.HasPrefix(line, []byte("data:")) {
		return nil, false, nil
	}

	// Extract the data portion
	data := bytes.TrimPrefix(line, []byte("data:"))
	data = bytes.TrimSpace(data)

	// Check for [DONE] marker
	if bytes.Equal(data, []byte("[DONE]")) {
		return nil, true, nil
	}

	// Parse JSON
	var chunk QwenStreamChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil, false, fmt.Errorf("failed to parse SSE chunk: %w", err)
	}

	return &chunk, false, nil
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

// isAuthRetryableStatus returns true for auth errors that may be transient
// (e.g., token validation delays, temporary auth service issues)
func isAuthRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusUnauthorized // 401
}

// waitWithJitter waits for the specified duration plus random jitter
func (q *QwenProvider) waitWithJitter(ctx context.Context, delay time.Duration) {
	// Add 10% jitter - using math/rand is acceptable for non-security jitter
	jitter := time.Duration(rand.Float64() * 0.1 * float64(delay)) // #nosec G404 - jitter doesn't require cryptographic randomness
	select {
	case <-ctx.Done():
	case <-time.After(delay + jitter):
	}
}

// nextDelay calculates the next delay using exponential backoff
func (q *QwenProvider) nextDelay(currentDelay time.Duration) time.Duration {
	nextDelay := time.Duration(float64(currentDelay) * q.retryConfig.Multiplier)
	if nextDelay > q.retryConfig.MaxDelay {
		nextDelay = q.retryConfig.MaxDelay
	}
	return nextDelay
}
