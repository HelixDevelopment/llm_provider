package cerebras

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

	"digital.vasic.llmprovider/pkg/models"

	"digital.vasic.llmprovider/pkg/discovery"
	"digital.vasic.llmprovider/pkg/i18n"
	"digital.vasic.llmprovider/pkg/settings"
	"github.com/sirupsen/logrus"
)

// DefaultHTTPTimeout is the compiled fallback for this adapter's HTTP
// client. It is a starting point, not a decision the library keeps making:
// LLMPROVIDER_CEREBRAS_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

var log = logrus.New()

const (
	CerebrasAPIURL = "https://api.cerebras.ai/v1/chat/completions"
	CerebrasModel  = "llama3.1-8b"
)

type CerebrasProvider struct {
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

type CerebrasRequest struct {
	Model       string            `json:"model"`
	Messages    []CerebrasMessage `json:"messages"`
	Temperature float64           `json:"temperature,omitempty"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	TopP        float64           `json:"top_p,omitempty"`
	Stream      bool              `json:"stream,omitempty"`
}

type CerebrasMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type CerebrasResponse struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []CerebrasChoice `json:"choices"`
	Usage   CerebrasUsage    `json:"usage"`
}

type CerebrasChoice struct {
	Index        int             `json:"index"`
	Message      CerebrasMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type CerebrasUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type CerebrasStreamResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []CerebrasStreamChoice `json:"choices"`
}

type CerebrasStreamChoice struct {
	Index        int             `json:"index"`
	Delta        CerebrasMessage `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
}

type CerebrasErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

// DefaultRetryConfig returns sensible defaults for Cerebras API retry behavior
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

func NewCerebrasProvider(apiKey, baseURL, model string) *CerebrasProvider {
	return NewCerebrasProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewCerebrasProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *CerebrasProvider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_CEREBRAS_BASE_URL.
		baseURL = settings.BaseURL("cerebras", CerebrasAPIURL)
	}
	if model == "" {
		// The compiled constant is a FALLBACK, not a decision this
		// library gets to keep making. A vendor retiring or rate-capping
		// a model must be answerable with an environment variable, not a
		// release. See pkg/settings: LLMPROVIDER_CEREBRAS_MODEL.
		model = settings.Model("cerebras", CerebrasModel)
	}

	p := &CerebrasProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("cerebras", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "cerebras",
		ModelsEndpoint: "https://api.cerebras.ai/v1/models",
		ModelsDevID:    "cerebras",
		APIKey:         apiKey,
		FallbackModels: []string{
			"llama3.1-8b",
			"gpt-oss-120b",
		},
	})

	return p
}

func (p *CerebrasProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("cerebras-%d", time.Now().UnixNano())
	}

	log.WithFields(logrus.Fields{
		"provider":   "cerebras",
		"model":      p.model,
		"request_id": requestID,
		"messages":   len(req.Messages),
	}).Debug("Starting Cerebras API call")

	// Convert internal request to Cerebras format
	cerebrasReq := p.convertRequest(req)

	log.WithFields(logrus.Fields{
		"provider":   "cerebras",
		"request_id": requestID,
		"max_tokens": cerebrasReq.MaxTokens,
		"temp":       cerebrasReq.Temperature,
	}).Debug("Request converted, making API call")

	// Make API call
	resp, err := p.makeAPICall(ctx, cerebrasReq)
	if err != nil {
		log.WithFields(logrus.Fields{
			"provider":   "cerebras",
			"request_id": requestID,
			"error":      err.Error(),
			"duration":   time.Since(startTime).String(),
		}).Error("Cerebras API call failed")
		return nil, fmt.Errorf("Cerebras API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Parse response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.WithFields(logrus.Fields{
			"provider":   "cerebras",
			"request_id": requestID,
			"error":      err.Error(),
		}).Error("Failed to read response body")
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	log.WithFields(logrus.Fields{
		"provider":    "cerebras",
		"request_id":  requestID,
		"status_code": resp.StatusCode,
		"body_size":   len(body),
	}).Debug("Received API response")

	if resp.StatusCode != http.StatusOK {
		var errResp CerebrasErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			log.WithFields(logrus.Fields{
				"provider":    "cerebras",
				"request_id":  requestID,
				"status_code": resp.StatusCode,
				"error_msg":   errResp.Error.Message,
				"error_type":  errResp.Error.Type,
			}).Error("Cerebras API returned error")
			return nil, fmt.Errorf("Cerebras API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		log.WithFields(logrus.Fields{
			"provider":    "cerebras",
			"request_id":  requestID,
			"status_code": resp.StatusCode,
			"body":        string(body[:min(500, len(body))]),
		}).Error("Cerebras API error response")
		return nil, fmt.Errorf("Cerebras API error: %d - %s", resp.StatusCode, string(body))
	}

	var cerebrasResp CerebrasResponse
	if err := json.Unmarshal(body, &cerebrasResp); err != nil {
		log.WithFields(logrus.Fields{
			"provider":   "cerebras",
			"request_id": requestID,
			"error":      err.Error(),
			"body":       string(body[:min(200, len(body))]),
		}).Error("Failed to parse Cerebras response")
		return nil, fmt.Errorf("failed to parse Cerebras response: %w", err)
	}

	// Check for empty choices
	if len(cerebrasResp.Choices) == 0 {
		log.WithFields(logrus.Fields{
			"provider":   "cerebras",
			"request_id": requestID,
			"response":   string(body[:min(500, len(body))]),
		}).Error("Cerebras API returned no choices")
		return nil, fmt.Errorf("Cerebras API returned no choices")
	}

	duration := time.Since(startTime)
	log.WithFields(logrus.Fields{
		"provider":      "cerebras",
		"request_id":    requestID,
		"duration":      duration.String(),
		"tokens_used":   cerebrasResp.Usage.TotalTokens,
		"content_len":   len(cerebrasResp.Choices[0].Message.Content),
		"finish_reason": cerebrasResp.Choices[0].FinishReason,
	}).Info("Cerebras API call completed successfully")

	// Convert back to internal format
	return p.convertResponse(req, &cerebrasResp, startTime), nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (p *CerebrasProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	// Convert internal request to Cerebras format
	cerebrasReq := p.convertRequest(req)
	cerebrasReq.Stream = true

	// Make streaming API call
	resp, err := p.makeAPICall(ctx, cerebrasReq)
	if err != nil {
		return nil, fmt.Errorf("Cerebras streaming API call failed: %w", err)
	}

	// Check for HTTP errors before starting stream
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("Cerebras API error: HTTP %d - failed to read response body: %v", resp.StatusCode, readErr)
		}
		return nil, fmt.Errorf("Cerebras API error: HTTP %d - %s", resp.StatusCode, string(body))
	}

	// Create response channel
	ch := make(chan *models.LLMResponse)

	go func() {
		defer func() { _ = resp.Body.Close() }()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		var fullContent string

		// processLine handles one SSE line; returns stop=true when the stream
		// is complete (DONE marker or finish_reason set). It is invoked even
		// for the final partial line ReadBytes returns alongside io.EOF, so a
		// last "data: {...}" / "data: [DONE]" chunk written without a trailing
		// newline is never dropped (mirrors the openrouter reference loop).
		processLine := func(raw []byte) (stop bool) {
			// Skip empty lines and "data: " prefix
			line := bytes.TrimSpace(raw)
			if !bytes.HasPrefix(line, []byte("data: ")) {
				return false
			}
			line = bytes.TrimPrefix(line, []byte("data: "))

			// Skip "[DONE]" marker
			if bytes.Equal(line, []byte("[DONE]")) {
				return true
			}

			// Parse JSON
			var streamResp CerebrasStreamResponse
			if err := json.Unmarshal(line, &streamResp); err != nil {
				return false // Skip malformed JSON
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
						ProviderID:     "cerebras",
						ProviderName:   "Cerebras",
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
					return true
				}
			}
			return false
		}

		for {
			line, err := reader.ReadBytes('\n')

			// Process any complete line content returned alongside the error
			// (ReadBytes returns the data read so far on EOF without a newline).
			// break (not return) so the post-loop final frame is still emitted.
			if len(line) > 0 {
				if processLine(line) {
					break
				}
			}

			if err != nil {
				if err == io.EOF {
					break
				}
				// Send error response and exit
				errorResp := &models.LLMResponse{
					ID:             "stream-error-" + req.ID,
					RequestID:      req.ID,
					ProviderID:     "cerebras",
					ProviderName:   "Cerebras",
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
		}

		// Send final response
		finalResp := &models.LLMResponse{
			ID:             "stream-final-" + req.ID,
			RequestID:      req.ID,
			ProviderID:     "cerebras",
			ProviderName:   "Cerebras",
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

func (p *CerebrasProvider) convertRequest(req *models.LLMRequest) CerebrasRequest {
	// Convert messages
	messages := make([]CerebrasMessage, 0, len(req.Messages)+1)

	// Add system prompt if present
	if req.Prompt != "" {
		messages = append(messages, CerebrasMessage{
			Role:    "system",
			Content: req.Prompt,
		})
	}

	// Add conversation messages
	for _, msg := range req.Messages {
		messages = append(messages, CerebrasMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	// Cap max_tokens to Cerebras's limit
	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096 // Default
	} else if maxTokens > 8192 {
		maxTokens = 8192 // Cerebras's limit
	}

	return CerebrasRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
	}
}

func (p *CerebrasProvider) convertResponse(req *models.LLMRequest, cerebrasResp *CerebrasResponse, startTime time.Time) *models.LLMResponse {
	var content string
	var finishReason string

	if len(cerebrasResp.Choices) > 0 {
		content = cerebrasResp.Choices[0].Message.Content
		finishReason = cerebrasResp.Choices[0].FinishReason
	}

	// Calculate confidence based on finish reason and response quality
	confidence := p.calculateConfidence(content, finishReason)

	return &models.LLMResponse{
		ID:           cerebrasResp.ID,
		RequestID:    req.ID,
		ProviderID:   "cerebras",
		ProviderName: "Cerebras",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   cerebrasResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             cerebrasResp.Model,
			"prompt_tokens":     cerebrasResp.Usage.PromptTokens,
			"completion_tokens": cerebrasResp.Usage.CompletionTokens,
		},
		Selected:       false,
		SelectionScore: 0.0,
		CreatedAt:      time.Now(),
	}
}

func (p *CerebrasProvider) calculateConfidence(content, finishReason string) float64 {
	confidence := 0.8 // Base confidence

	// Adjust based on finish reason
	switch finishReason {
	case "stop":
		confidence += 0.1
	case "length":
		confidence -= 0.1
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

func (p *CerebrasProvider) makeAPICall(ctx context.Context, req CerebrasRequest) (*http.Response, error) {
	return p.makeAPICallWithAuthRetry(ctx, req, true)
}

// makeAPICallWithAuthRetry performs the API call with optional 401 retry
func (p *CerebrasProvider) makeAPICallWithAuthRetry(ctx context.Context, req CerebrasRequest, allowAuthRetry bool) (*http.Response, error) {
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

		// Set headers - Cerebras uses Bearer token auth
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
				"provider":    "cerebras",
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
				"provider":    "cerebras",
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
// (e.g., token validation delays, temporary auth service issues)
func isAuthRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusUnauthorized // 401
}

// waitWithJitter waits for the specified duration plus random jitter
func (p *CerebrasProvider) waitWithJitter(ctx context.Context, delay time.Duration) {
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
func (p *CerebrasProvider) nextDelay(currentDelay time.Duration) time.Duration {
	nextDelay := time.Duration(float64(currentDelay) * p.retryConfig.Multiplier)
	if nextDelay > p.retryConfig.MaxDelay {
		nextDelay = p.retryConfig.MaxDelay
	}
	return nextDelay
}

// GetCapabilities returns the capabilities of the Cerebras provider
func (p *CerebrasProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: p.discoverer.DiscoverModels(),
		SupportedFeatures: []string{
			"text_completion",
			"chat",
			"streaming",
		},
		SupportedRequestTypes: []string{
			"text_completion",
			"chat",
		},
		SupportsStreaming:       true,
		SupportsFunctionCalling: false,
		SupportsVision:          false,
		SupportsTools:           false,
		SupportsSearch:          false,
		SupportsReasoning:       true,
		SupportsCodeCompletion:  true,
		SupportsCodeAnalysis:    true,
		SupportsRefactoring:     true,
		Limits: models.ModelLimits{
			MaxTokens:             8192,
			MaxInputLength:        8192,
			MaxOutputLength:       8192,
			MaxConcurrentRequests: 10,
		},
		Metadata: map[string]string{
			"provider":     "Cerebras",
			"model_family": "Llama",
			"api_version":  "v1",
			"note":         i18n.Tr(context.Background(), "provider.cerebras.description", nil),
		},
	}
}

// ValidateConfig validates the provider configuration
func (p *CerebrasProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
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
func (p *CerebrasProvider) modelsURL() string {
	return strings.TrimSuffix(p.baseURL, "/chat/completions") + "/models"
}

// HealthCheck implements health checking for the Cerebras provider
func (p *CerebrasProvider) HealthCheck() error {
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
