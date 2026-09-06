package mistral

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
	"github.com/sirupsen/logrus"
)

// DefaultHTTPTimeout is the compiled fallback for this adapter's HTTP
// client. It is a starting point, not a decision the library keeps making:
// LLMPROVIDER_MISTRAL_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

var log = logrus.New()

const (
	MistralAPIURL = "https://api.mistral.ai/v1/chat/completions"
	MistralModel  = "mistral-large-latest"
)

type MistralProvider struct {
	apiKey      string
	baseURL     string
	model       string
	httpClient  *http.Client
	retryConfig RetryConfig
	discoverer  *discovery.Discoverer
}

// RetryConfig defines retry behavior for API calls
type RetryConfig struct {
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

type MistralRequest struct {
	Model       string           `json:"model"`
	Messages    []MistralMessage `json:"messages"`
	Temperature float64          `json:"temperature,omitempty"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
	TopP        float64          `json:"top_p,omitempty"`
	Stream      bool             `json:"stream,omitempty"`
	SafePrompt  bool             `json:"safe_prompt,omitempty"`
	Tools       []MistralTool    `json:"tools,omitempty"`
	ToolChoice  interface{}      `json:"tool_choice,omitempty"`
}

type MistralMessage struct {
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	ToolCalls []MistralToolCall `json:"tool_calls,omitempty"`
}

// MistralTool represents a tool definition for Mistral API
type MistralTool struct {
	Type     string          `json:"type"`
	Function MistralToolFunc `json:"function"`
}

// MistralToolFunc represents a function definition
type MistralToolFunc struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// MistralToolCall represents a tool call in the response
type MistralToolCall struct {
	ID       string                  `json:"id"`
	Type     string                  `json:"type"`
	Function MistralToolCallFunction `json:"function"`
}

// MistralToolCallFunction represents the function call details
type MistralToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type MistralResponse struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Created int64           `json:"created"`
	Model   string          `json:"model"`
	Choices []MistralChoice `json:"choices"`
	Usage   MistralUsage    `json:"usage"`
}

type MistralChoice struct {
	Index        int            `json:"index"`
	Message      MistralMessage `json:"message"`
	FinishReason string         `json:"finish_reason"`
}

type MistralUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type MistralStreamResponse struct {
	ID      string                `json:"id"`
	Object  string                `json:"object"`
	Created int64                 `json:"created"`
	Model   string                `json:"model"`
	Choices []MistralStreamChoice `json:"choices"`
}

type MistralStreamChoice struct {
	Index        int            `json:"index"`
	Delta        MistralMessage `json:"delta"`
	FinishReason *string        `json:"finish_reason"`
}

type MistralErrorResponse struct {
	Object  string  `json:"object,omitempty"`
	Message string  `json:"message,omitempty"`
	Type    string  `json:"type,omitempty"`
	Param   *string `json:"param,omitempty"`
	Code    *string `json:"code,omitempty"`
}

// DefaultRetryConfig returns sensible defaults for Mistral API retry behavior
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

func NewMistralProvider(apiKey, baseURL, model string) *MistralProvider {
	return NewMistralProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewMistralProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *MistralProvider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_MISTRAL_BASE_URL.
		baseURL = settings.BaseURL("mistral", MistralAPIURL)
	}
	if model == "" {
		// The compiled constant is a FALLBACK, not a decision this
		// library gets to keep making. A vendor retiring or rate-capping
		// a model must be answerable with an environment variable, not a
		// release. See pkg/settings: LLMPROVIDER_MISTRAL_MODEL.
		model = settings.Model("mistral", MistralModel)
	}

	p := &MistralProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("mistral", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "mistral",
		ModelsEndpoint: "https://api.mistral.ai/v1/models",
		ModelsDevID:    "mistral",
		APIKey:         apiKey,
		FallbackModels: []string{
			"mistral-large-latest",
			"mistral-medium",
			"mistral-small-latest",
			"open-mistral-7b",
			"open-mixtral-8x7b",
			"open-mixtral-8x22b",
			"codestral-latest",
		},
	})

	return p
}

func (p *MistralProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("mistral-%d", time.Now().UnixNano())
	}

	log.WithFields(logrus.Fields{
		"provider":   "mistral",
		"model":      p.model,
		"request_id": requestID,
		"messages":   len(req.Messages),
	}).Debug("Starting Mistral API call")

	// Convert internal request to Mistral format
	mistralReq := p.convertRequest(req)

	log.WithFields(logrus.Fields{
		"provider":   "mistral",
		"request_id": requestID,
		"max_tokens": mistralReq.MaxTokens,
		"temp":       mistralReq.Temperature,
	}).Debug("Request converted, making API call")

	// Make API call
	resp, err := p.makeAPICall(ctx, mistralReq)
	if err != nil {
		log.WithFields(logrus.Fields{
			"provider":   "mistral",
			"request_id": requestID,
			"error":      err.Error(),
			"duration":   time.Since(startTime).String(),
		}).Error("Mistral API call failed")
		return nil, fmt.Errorf("Mistral API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Parse response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.WithFields(logrus.Fields{
			"provider":   "mistral",
			"request_id": requestID,
			"error":      err.Error(),
		}).Error("Failed to read response body")
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	log.WithFields(logrus.Fields{
		"provider":    "mistral",
		"request_id":  requestID,
		"status_code": resp.StatusCode,
		"body_size":   len(body),
	}).Debug("Received API response")

	if resp.StatusCode != http.StatusOK {
		var errResp MistralErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Message != "" {
			log.WithFields(logrus.Fields{
				"provider":    "mistral",
				"request_id":  requestID,
				"status_code": resp.StatusCode,
				"error_msg":   errResp.Message,
				"error_type":  errResp.Type,
			}).Error("Mistral API returned error")
			return nil, fmt.Errorf("Mistral API error: %d - %s", resp.StatusCode, errResp.Message)
		}
		log.WithFields(logrus.Fields{
			"provider":    "mistral",
			"request_id":  requestID,
			"status_code": resp.StatusCode,
			"body":        string(body[:min(500, len(body))]),
		}).Error("Mistral API error response")
		return nil, fmt.Errorf("Mistral API error: %d - %s", resp.StatusCode, string(body))
	}

	var mistralResp MistralResponse
	if err := json.Unmarshal(body, &mistralResp); err != nil {
		log.WithFields(logrus.Fields{
			"provider":   "mistral",
			"request_id": requestID,
			"error":      err.Error(),
			"body":       string(body[:min(200, len(body))]),
		}).Error("Failed to parse Mistral response")
		return nil, fmt.Errorf("failed to parse Mistral response: %w", err)
	}

	// Check for empty choices
	if len(mistralResp.Choices) == 0 {
		log.WithFields(logrus.Fields{
			"provider":   "mistral",
			"request_id": requestID,
			"response":   string(body[:min(500, len(body))]),
		}).Error("Mistral API returned no choices")
		return nil, fmt.Errorf("Mistral API returned no choices")
	}

	duration := time.Since(startTime)
	log.WithFields(logrus.Fields{
		"provider":      "mistral",
		"request_id":    requestID,
		"duration":      duration.String(),
		"tokens_used":   mistralResp.Usage.TotalTokens,
		"content_len":   len(mistralResp.Choices[0].Message.Content),
		"finish_reason": mistralResp.Choices[0].FinishReason,
	}).Info("Mistral API call completed successfully")

	// Convert back to internal format
	return p.convertResponse(req, &mistralResp, startTime), nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (p *MistralProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	// Convert internal request to Mistral format
	mistralReq := p.convertRequest(req)
	mistralReq.Stream = true

	// Make streaming API call
	resp, err := p.makeAPICall(ctx, mistralReq)
	if err != nil {
		return nil, fmt.Errorf("Mistral streaming API call failed: %w", err)
	}

	// Check for HTTP errors before starting stream
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf("Mistral API error: HTTP %d - %s", resp.StatusCode, string(body))
	}

	// Create response channel
	ch := make(chan *models.LLMResponse)

	go func() {
		defer func() { _ = resp.Body.Close() }()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		var fullContent string

		for {
			line, err := reader.ReadBytes('\n')
			// On io.EOF, ReadBytes may still return a final partial line (the last
			// data chunk without a trailing newline). Process it before breaking.
			eof := err == io.EOF
			if err != nil && !eof {
				// Send error response and exit
				errorResp := &models.LLMResponse{
					ID:             "stream-error-" + req.ID,
					RequestID:      req.ID,
					ProviderID:     "mistral",
					ProviderName:   "Mistral",
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
				if eof {
					break
				}
				continue
			}
			line = bytes.TrimPrefix(line, []byte("data: "))

			// Skip "[DONE]" marker
			if bytes.Equal(line, []byte("[DONE]")) {
				break
			}

			// Parse JSON
			var streamResp MistralStreamResponse
			if err := json.Unmarshal(line, &streamResp); err != nil {
				if eof {
					break
				}
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
						ProviderID:     "mistral",
						ProviderName:   "Mistral",
						Content:        delta,
						Confidence:     0.8,
						TokensUsed:     1,
						ResponseTime:   time.Since(startTime).Milliseconds(),
						FinishReason:   "",
						Selected:       false,
						SelectionScore: 0.0,
						CreatedAt:      time.Now(),
					}
					ch <- chunkResp
				}

				// Check if stream is finished
				if streamResp.Choices[0].FinishReason != nil {
					break
				}
			}

			if eof {
				break
			}
		}

		// Send final response
		finalResp := &models.LLMResponse{
			ID:             "stream-final-" + req.ID,
			RequestID:      req.ID,
			ProviderID:     "mistral",
			ProviderName:   "Mistral",
			Content:        "",
			Confidence:     0.8,
			TokensUsed:     len(fullContent) / 4,
			ResponseTime:   time.Since(startTime).Milliseconds(),
			FinishReason:   "stop",
			Selected:       false,
			SelectionScore: 0.0,
			CreatedAt:      time.Now(),
		}
		ch <- finalResp
	}()

	return ch, nil
}

func (p *MistralProvider) convertRequest(req *models.LLMRequest) MistralRequest {
	// Convert messages
	messages := make([]MistralMessage, 0, len(req.Messages)+1)

	// Add system prompt if present
	if req.Prompt != "" {
		messages = append(messages, MistralMessage{
			Role:    "system",
			Content: req.Prompt,
		})
	}

	// Add conversation messages
	for _, msg := range req.Messages {
		messages = append(messages, MistralMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	// Cap max_tokens to Mistral's limit
	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096 // Default
	} else if maxTokens > 32768 {
		maxTokens = 32768 // Mistral's max limit for large models
	}

	mistralReq := MistralRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
		SafePrompt:  false,
	}

	// Convert tools if provided
	if len(req.Tools) > 0 {
		mistralReq.Tools = make([]MistralTool, len(req.Tools))
		for i, tool := range req.Tools {
			mistralReq.Tools[i] = MistralTool{
				Type: tool.Type,
				Function: MistralToolFunc{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			}
		}
		if req.ToolChoice != "" {
			mistralReq.ToolChoice = req.ToolChoice
		}
	}

	return mistralReq
}

func (p *MistralProvider) convertResponse(req *models.LLMRequest, mistralResp *MistralResponse, startTime time.Time) *models.LLMResponse {
	var content string
	var finishReason string
	var toolCalls []models.ToolCall

	if len(mistralResp.Choices) > 0 {
		content = mistralResp.Choices[0].Message.Content
		finishReason = mistralResp.Choices[0].FinishReason

		// Parse tool_calls from response
		if len(mistralResp.Choices[0].Message.ToolCalls) > 0 {
			toolCalls = make([]models.ToolCall, len(mistralResp.Choices[0].Message.ToolCalls))
			for i, tc := range mistralResp.Choices[0].Message.ToolCalls {
				toolCalls[i] = models.ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: models.ToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
			// Set finish_reason to tool_calls if we have tool calls
			if finishReason == "" || finishReason == "stop" {
				finishReason = "tool_calls"
			}
		}
	}

	// Calculate confidence based on finish reason and response quality
	confidence := p.calculateConfidence(content, finishReason)

	return &models.LLMResponse{
		ID:           mistralResp.ID,
		RequestID:    req.ID,
		ProviderID:   "mistral",
		ProviderName: "Mistral",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   mistralResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		ToolCalls:    toolCalls,
		Metadata: map[string]any{
			"model":             mistralResp.Model,
			"prompt_tokens":     mistralResp.Usage.PromptTokens,
			"completion_tokens": mistralResp.Usage.CompletionTokens,
		},
		Selected:       false,
		SelectionScore: 0.0,
		CreatedAt:      time.Now(),
	}
}

func (p *MistralProvider) calculateConfidence(content, finishReason string) float64 {
	confidence := 0.8 // Base confidence

	// Adjust based on finish reason
	switch finishReason {
	case "stop":
		confidence += 0.1
	case "length":
		confidence -= 0.1
	case "model_length":
		confidence -= 0.15
	}

	// Adjust based on content length
	if len(content) > 100 {
		confidence += 0.05
	}
	if len(content) > 500 {
		confidence += 0.05
	}

	// Ensure confidence is within bounds
	if confidence > 1.0 {
		confidence = 1.0
	}
	if confidence < 0.0 {
		confidence = 0.0
	}

	return confidence
}

func (p *MistralProvider) makeAPICall(ctx context.Context, req MistralRequest) (*http.Response, error) {
	return p.makeAPICallWithAuthRetry(ctx, req, true)
}

// makeAPICallWithAuthRetry performs the API call with optional 401 retry
func (p *MistralProvider) makeAPICallWithAuthRetry(ctx context.Context, req MistralRequest, allowAuthRetry bool) (*http.Response, error) {
	// Marshal request
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var lastErr error
	delay := p.retryConfig.InitialDelay

	for attempt := 0; attempt <= p.retryConfig.MaxRetries; attempt++ {
		// Check context before making request
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
		default:
		}

		// Create HTTP request (fresh for each attempt)
		httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL, bytes.NewBuffer(body))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		// Set headers - Mistral uses Bearer token auth
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
		httpReq.Header.Set("User-Agent", "LLMProvider/1.0")

		// Make request
		resp, err := p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("HTTP request failed: %w", err)
			// Retry on network errors
			if attempt < p.retryConfig.MaxRetries {
				p.waitWithJitter(ctx, delay)
				delay = p.nextDelay(delay)
				continue
			}
			return nil, lastErr
		}

		// Check for auth errors (401) - retry once with a short delay
		// This handles transient auth issues (token validation delays, auth service hiccups)
		if isAuthRetryableStatus(resp.StatusCode) && allowAuthRetry {
			_ = resp.Body.Close()
			log.WithFields(logrus.Fields{
				"provider":    "mistral",
				"status_code": resp.StatusCode,
				"attempt":     attempt + 1,
			}).Warn("Received 401 Unauthorized, retrying once after short delay")

			// Short delay before auth retry (500ms with jitter)
			authRetryDelay := 500 * time.Millisecond
			p.waitWithJitter(ctx, authRetryDelay)

			// Recursive call with auth retry disabled to prevent infinite loops
			return p.makeAPICallWithAuthRetry(ctx, req, false)
		}

		// Check for retryable status codes (429, 5xx)
		if isRetryableStatus(resp.StatusCode) && attempt < p.retryConfig.MaxRetries {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("HTTP %d: retryable error", resp.StatusCode)
			log.WithFields(logrus.Fields{
				"provider":    "mistral",
				"status_code": resp.StatusCode,
				"attempt":     attempt + 1,
				"max_retries": p.retryConfig.MaxRetries,
			}).Debug("Retrying after retryable error")
			p.waitWithJitter(ctx, delay)
			delay = p.nextDelay(delay)
			continue
		}

		return resp, nil
	}

	return nil, fmt.Errorf("all %d retry attempts failed: %w", p.retryConfig.MaxRetries+1, lastErr)
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
// (e.g., token refresh needed, temporary auth service issues)
func isAuthRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusUnauthorized // 401
}

// waitWithJitter waits for the specified duration plus random jitter
func (p *MistralProvider) waitWithJitter(ctx context.Context, delay time.Duration) {
	select {
	case <-ctx.Done():
	case <-time.After(jitteredDelay(delay)):
	}
}

// jitteredDelay returns delay plus up to 10% random jitter -- the exact
// duration waitWithJitter arms its timer with.
//
// Extracted so that the "at most 10% over" bound can be asserted directly and
// deterministically. Asserting it through wall-clock elapsed time does not
// work: elapsed time also contains scheduler latency the host controls, so a
// wall-clock ceiling measures the machine at least as much as it measures this
// package, and fails on a busy host while the code is correct.
func jitteredDelay(delay time.Duration) time.Duration {
	// Add 10% jitter - using math/rand is acceptable for non-security jitter
	jitter := time.Duration(rand.Float64() * 0.1 * float64(delay)) // #nosec G404 - jitter doesn't require cryptographic randomness
	return delay + jitter
}

// nextDelay calculates the next delay using exponential backoff
func (p *MistralProvider) nextDelay(currentDelay time.Duration) time.Duration {
	nextDelay := time.Duration(float64(currentDelay) * p.retryConfig.Multiplier)
	if nextDelay > p.retryConfig.MaxDelay {
		nextDelay = p.retryConfig.MaxDelay
	}
	return nextDelay
}

// GetCapabilities returns the capabilities of the Mistral provider
func (p *MistralProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: p.discoverer.DiscoverModels(),
		SupportedFeatures: []string{
			"text_completion",
			"chat",
			"function_calling",
			"streaming",
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
		SupportsRefactoring:     true,
		Limits: models.ModelLimits{
			MaxTokens:             32768,
			MaxInputLength:        32768,
			MaxOutputLength:       32768,
			MaxConcurrentRequests: 10,
		},
		Metadata: map[string]string{
			"provider":     "Mistral",
			"model_family": "Mistral",
			"api_version":  "v1",
		},
	}
}

// ValidateConfig validates the provider configuration
func (p *MistralProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string

	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}

	if p.baseURL == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_base_url_required", nil))
	}

	if p.model == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_model_required", nil))
	}

	return len(errors) == 0, errors
}

// modelsURL derives the models endpoint from the configured base URL so the
// health check honors a custom baseURL (CONST-051(B) config-injection) instead
// of a hardcoded production literal.
func (p *MistralProvider) modelsURL() string {
	return strings.TrimSuffix(p.baseURL, "/chat/completions") + "/models"
}

// HealthCheck implements health checking for the Mistral provider
func (p *MistralProvider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Simple health check - try to get models list
	req, err := http.NewRequestWithContext(ctx, "GET", p.modelsURL(), nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}

	return nil
}
