package publicai

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
// LLMPROVIDER_PUBLICAI_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

var log = logrus.New()

const (
	PublicAIAPIURL          = "https://api.publicai.co/v1/chat/completions"
	PublicAIModel           = "swiss-ai/apertus-8b-instruct"
	PublicAIModelsURL       = "https://api.publicai.co/v1/models"
	PublicAIMaxContext      = 65536
	PublicAIMaxOutput       = 8192
	PublicAIRecommendedTemp = 0.8
	PublicAIRecommendedTopP = 0.9
)

type PublicAIProvider struct {
	apiKey      string
	baseURL     string
	model       string
	httpClient  *http.Client
	retryConfig RetryConfig
	discoverer  *discovery.Discoverer
}

type RetryConfig struct {
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

type PublicAIRequest struct {
	Model       string            `json:"model"`
	Messages    []PublicAIMessage `json:"messages"`
	Temperature float64           `json:"temperature,omitempty"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	TopP        float64           `json:"top_p,omitempty"`
	Stream      bool              `json:"stream,omitempty"`
}

type PublicAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type PublicAIResponse struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []PublicAIChoice `json:"choices"`
	Usage   PublicAIUsage    `json:"usage"`
}

type PublicAIChoice struct {
	Index        int             `json:"index"`
	Message      PublicAIMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type PublicAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type PublicAIStreamResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []PublicAIStreamChoice `json:"choices"`
}

type PublicAIStreamChoice struct {
	Index        int             `json:"index"`
	Delta        PublicAIMessage `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
}

type PublicAIErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 2 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

func NewPublicAIProvider(apiKey, baseURL, model string) *PublicAIProvider {
	return NewPublicAIProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewPublicAIProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *PublicAIProvider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_PUBLICAI_BASE_URL.
		baseURL = settings.BaseURL("publicai", PublicAIAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_PUBLICAI_MODEL overrides this compiled fallback.
		model = settings.Model("publicai", PublicAIModel)
	}

	p := &PublicAIProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("publicai", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "publicai",
		ModelsEndpoint: PublicAIModelsURL,
		ModelsDevID:    "publicai",
		APIKey:         apiKey,
		FallbackModels: []string{
			"swiss-ai/apertus-8b-instruct",
		},
	})

	return p
}

func (p *PublicAIProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("publicai-%d", time.Now().UnixNano())
	}

	log.WithFields(logrus.Fields{
		"provider":   "publicai",
		"model":      p.model,
		"request_id": requestID,
		"messages":   len(req.Messages),
	}).Debug("Starting PublicAI API call")

	publicaiReq := p.convertRequest(req)

	log.WithFields(logrus.Fields{
		"provider":   "publicai",
		"request_id": requestID,
		"max_tokens": publicaiReq.MaxTokens,
		"temp":       publicaiReq.Temperature,
	}).Debug("Request converted, making API call")

	resp, err := p.makeAPICall(ctx, publicaiReq)
	if err != nil {
		log.WithFields(logrus.Fields{
			"provider":   "publicai",
			"request_id": requestID,
			"error":      err.Error(),
			"duration":   time.Since(startTime).String(),
		}).Error("PublicAI API call failed")
		return nil, fmt.Errorf("PublicAI API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.WithFields(logrus.Fields{
			"provider":   "publicai",
			"request_id": requestID,
			"error":      err.Error(),
		}).Error("Failed to read response body")
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	log.WithFields(logrus.Fields{
		"provider":    "publicai",
		"request_id":  requestID,
		"status_code": resp.StatusCode,
		"body_size":   len(body),
	}).Debug("Received API response")

	if resp.StatusCode != http.StatusOK {
		var errResp PublicAIErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			log.WithFields(logrus.Fields{
				"provider":    "publicai",
				"request_id":  requestID,
				"status_code": resp.StatusCode,
				"error_msg":   errResp.Error.Message,
				"error_type":  errResp.Error.Type,
			}).Error("PublicAI API returned error")
			return nil, fmt.Errorf("PublicAI API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		log.WithFields(logrus.Fields{
			"provider":    "publicai",
			"request_id":  requestID,
			"status_code": resp.StatusCode,
			"body":        string(body[:min(500, len(body))]),
		}).Error("PublicAI API error response")
		return nil, fmt.Errorf("PublicAI API error: %d - %s", resp.StatusCode, string(body))
	}

	var publicaiResp PublicAIResponse
	if err := json.Unmarshal(body, &publicaiResp); err != nil {
		log.WithFields(logrus.Fields{
			"provider":   "publicai",
			"request_id": requestID,
			"error":      err.Error(),
			"body":       string(body[:min(200, len(body))]),
		}).Error("Failed to parse PublicAI response")
		return nil, fmt.Errorf("failed to parse PublicAI response: %w", err)
	}

	if len(publicaiResp.Choices) == 0 {
		log.WithFields(logrus.Fields{
			"provider":   "publicai",
			"request_id": requestID,
			"response":   string(body[:min(500, len(body))]),
		}).Error("PublicAI API returned no choices")
		return nil, fmt.Errorf("PublicAI API returned no choices")
	}

	duration := time.Since(startTime)
	log.WithFields(logrus.Fields{
		"provider":      "publicai",
		"request_id":    requestID,
		"duration":      duration.String(),
		"tokens_used":   publicaiResp.Usage.TotalTokens,
		"content_len":   len(publicaiResp.Choices[0].Message.Content),
		"finish_reason": publicaiResp.Choices[0].FinishReason,
	}).Info("PublicAI API call completed successfully")

	return p.convertResponse(req, &publicaiResp, startTime), nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (p *PublicAIProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	publicaiReq := p.convertRequest(req)
	publicaiReq.Stream = true

	resp, err := p.makeAPICall(ctx, publicaiReq)
	if err != nil {
		return nil, fmt.Errorf("PublicAI streaming API call failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()            //nolint:errcheck
		return nil, fmt.Errorf("PublicAI API error: HTTP %d - %s", resp.StatusCode, string(body))
	}

	ch := make(chan *models.LLMResponse)

	go func() {
		defer func() { _ = resp.Body.Close() }()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		var fullContent string

		for {
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				errorResp := &models.LLMResponse{
					ID:             "stream-error-" + req.ID,
					RequestID:      req.ID,
					ProviderID:     "publicai",
					ProviderName:   "Public AI",
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

			line = bytes.TrimSpace(line)
			if !bytes.HasPrefix(line, []byte("data: ")) {
				continue
			}
			line = bytes.TrimPrefix(line, []byte("data: "))

			if bytes.Equal(line, []byte("[DONE]")) {
				break
			}

			var streamResp PublicAIStreamResponse
			if err := json.Unmarshal(line, &streamResp); err != nil {
				continue
			}

			if len(streamResp.Choices) > 0 {
				delta := streamResp.Choices[0].Delta.Content
				if delta != "" {
					fullContent += delta

					chunkResp := &models.LLMResponse{
						ID:             streamResp.ID,
						RequestID:      req.ID,
						ProviderID:     "publicai",
						ProviderName:   "Public AI",
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

				if streamResp.Choices[0].FinishReason != nil {
					break
				}
			}
		}

		finalResp := &models.LLMResponse{
			ID:             "stream-final-" + req.ID,
			RequestID:      req.ID,
			ProviderID:     "publicai",
			ProviderName:   "Public AI",
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

func (p *PublicAIProvider) convertRequest(req *models.LLMRequest) PublicAIRequest {
	messages := make([]PublicAIMessage, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, PublicAIMessage{
			Role:    "system",
			Content: req.Prompt,
		})
	}

	for _, msg := range req.Messages {
		messages = append(messages, PublicAIMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = PublicAIMaxOutput
	} else if maxTokens > PublicAIMaxOutput {
		maxTokens = PublicAIMaxOutput
	}

	temp := req.ModelParams.Temperature
	if temp <= 0 {
		temp = PublicAIRecommendedTemp
	}

	topP := req.ModelParams.TopP
	if topP <= 0 {
		topP = PublicAIRecommendedTopP
	}

	return PublicAIRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: temp,
		MaxTokens:   maxTokens,
		TopP:        topP,
		Stream:      false,
	}
}

func (p *PublicAIProvider) convertResponse(req *models.LLMRequest, publicaiResp *PublicAIResponse, startTime time.Time) *models.LLMResponse {
	var content string
	var finishReason string

	if len(publicaiResp.Choices) > 0 {
		content = publicaiResp.Choices[0].Message.Content
		finishReason = publicaiResp.Choices[0].FinishReason
	}

	confidence := p.calculateConfidence(content, finishReason)

	return &models.LLMResponse{
		ID:           publicaiResp.ID,
		RequestID:    req.ID,
		ProviderID:   "publicai",
		ProviderName: "Public AI",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   publicaiResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             publicaiResp.Model,
			"prompt_tokens":     publicaiResp.Usage.PromptTokens,
			"completion_tokens": publicaiResp.Usage.CompletionTokens,
		},
		Selected:       false,
		SelectionScore: 0.0,
		CreatedAt:      time.Now(),
	}
}

func (p *PublicAIProvider) calculateConfidence(content, finishReason string) float64 {
	confidence := 0.8

	switch finishReason {
	case "stop":
		confidence += 0.1
	case "length":
		confidence -= 0.1
	}

	if len(content) > 100 {
		confidence += 0.05
	}
	if len(content) > 500 {
		confidence += 0.05
	}

	if confidence > 1.0 {
		confidence = 1.0
	}
	if confidence < 0.0 {
		confidence = 0.0
	}

	return confidence
}

func (p *PublicAIProvider) makeAPICall(ctx context.Context, req PublicAIRequest) (*http.Response, error) {
	return p.makeAPICallWithAuthRetry(ctx, req, true)
}

func (p *PublicAIProvider) makeAPICallWithAuthRetry(ctx context.Context, req PublicAIRequest, allowAuthRetry bool) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var lastErr error
	delay := p.retryConfig.InitialDelay

	for attempt := 0; attempt <= p.retryConfig.MaxRetries; attempt++ {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
		default:
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL, bytes.NewBuffer(body))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
		httpReq.Header.Set("User-Agent", "LLMProvider/1.0")

		resp, err := p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("HTTP request failed: %w", err)
			if attempt < p.retryConfig.MaxRetries {
				p.waitWithJitter(ctx, delay)
				delay = p.nextDelay(delay)
				continue
			}
			return nil, lastErr
		}

		if isAuthRetryableStatus(resp.StatusCode) && allowAuthRetry {
			_ = resp.Body.Close()
			log.WithFields(logrus.Fields{
				"provider":    "publicai",
				"status_code": resp.StatusCode,
				"attempt":     attempt + 1,
			}).Warn("Received 401 Unauthorized, retrying once after short delay")

			authRetryDelay := 500 * time.Millisecond
			p.waitWithJitter(ctx, authRetryDelay)

			return p.makeAPICallWithAuthRetry(ctx, req, false)
		}

		if isRetryableStatus(resp.StatusCode) && attempt < p.retryConfig.MaxRetries {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("HTTP %d: retryable error", resp.StatusCode)
			log.WithFields(logrus.Fields{
				"provider":    "publicai",
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

func isRetryableStatus(statusCode int) bool {
	switch statusCode {
	case http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}

func isAuthRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusUnauthorized
}

func (p *PublicAIProvider) waitWithJitter(ctx context.Context, delay time.Duration) {
	jitter := time.Duration(rand.Float64() * 0.1 * float64(delay))
	select {
	case <-ctx.Done():
	case <-time.After(delay + jitter):
	}
}

func (p *PublicAIProvider) nextDelay(currentDelay time.Duration) time.Duration {
	nextDelay := time.Duration(float64(currentDelay) * p.retryConfig.Multiplier)
	if nextDelay > p.retryConfig.MaxDelay {
		nextDelay = p.retryConfig.MaxDelay
	}
	return nextDelay
}

func (p *PublicAIProvider) GetCapabilities() *models.ProviderCapabilities {
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
			MaxTokens:             PublicAIMaxOutput,
			MaxInputLength:        PublicAIMaxContext,
			MaxOutputLength:       PublicAIMaxOutput,
			MaxConcurrentRequests: 5,
		},
		Metadata: map[string]string{
			"provider":          "Public AI",
			"model_family":      "Apertus",
			"api_version":       "v1",
			"recommended_temp":  "0.8",
			"recommended_top_p": "0.9",
			"note":              i18n.Tr(context.Background(), "provider.publicai.description", nil),
		},
	}
}

func (p *PublicAIProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
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
// of a hardcoded production constant.
func (p *PublicAIProvider) modelsURL() string {
	return strings.TrimSuffix(p.baseURL, "/chat/completions") + "/models"
}

func (p *PublicAIProvider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", p.modelsURL(), nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("User-Agent", "LLMProvider/1.0")

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
