package codestral

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

// modelsURL derives the /models endpoint from the configured baseURL so
// health checks honor operator overrides (proxies, mirrors, httptest in
// unit tests) instead of always hitting the hardcoded production URL.
func (p *CodestralProvider) modelsURL() string {
	return strings.TrimSuffix(p.baseURL, "/chat/completions") + "/models"
}

const (
	CodestralAPIURL     = "https://api.mistral.ai/v1/chat/completions"
	CodestralModel      = "codestral-latest"
	CodestralModelsURL  = "https://api.mistral.ai/v1/models"
	CodestralMaxContext = 32768
	CodestralMaxOutput  = 8192
)

type CodestralProvider struct {
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

type CodestralRequest struct {
	Model       string             `json:"model"`
	Messages    []CodestralMessage `json:"messages"`
	Temperature float64            `json:"temperature,omitempty"`
	MaxTokens   int                `json:"max_tokens,omitempty"`
	TopP        float64            `json:"top_p,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
}

type CodestralMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type CodestralResponse struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []CodestralChoice `json:"choices"`
	Usage   CodestralUsage    `json:"usage"`
}

type CodestralChoice struct {
	Index        int              `json:"index"`
	Message      CodestralMessage `json:"message"`
	FinishReason string           `json:"finish_reason"`
}

type CodestralUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type CodestralStreamResponse struct {
	ID      string                  `json:"id"`
	Object  string                  `json:"object"`
	Created int64                   `json:"created"`
	Model   string                  `json:"model"`
	Choices []CodestralStreamChoice `json:"choices"`
}

type CodestralStreamChoice struct {
	Index        int              `json:"index"`
	Delta        CodestralMessage `json:"delta"`
	FinishReason *string          `json:"finish_reason"`
}

type CodestralErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

func NewCodestralProvider(apiKey, baseURL, model string) *CodestralProvider {
	return NewCodestralProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewCodestralProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *CodestralProvider {
	if baseURL == "" {
		baseURL = CodestralAPIURL
	}
	if model == "" {
		// The compiled constant is a FALLBACK, not a decision this
		// library gets to keep making. A vendor retiring or rate-capping
		// a model must be answerable with an environment variable, not a
		// release. See pkg/settings: LLMPROVIDER_CODESTRAL_MODEL.
		model = settings.Model("codestral", CodestralModel)
	}

	p := &CodestralProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "codestral",
		ModelsEndpoint: CodestralModelsURL,
		ModelsDevID:    "mistral",
		APIKey:         apiKey,
		FallbackModels: []string{
			"codestral-latest",
			"codestral-2405",
		},
	})

	return p
}

func (p *CodestralProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("codestral-%d", time.Now().UnixNano())
	}

	codestralReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, codestralReq)
	if err != nil {
		return nil, fmt.Errorf("Codestral API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp CodestralErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("Codestral API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		return nil, fmt.Errorf("Codestral API error: %d - %s", resp.StatusCode, string(body))
	}

	var codestralResp CodestralResponse
	if err := json.Unmarshal(body, &codestralResp); err != nil {
		return nil, fmt.Errorf("failed to parse Codestral response: %w", err)
	}

	if len(codestralResp.Choices) == 0 {
		return nil, fmt.Errorf("Codestral API returned no choices")
	}

	return p.convertResponse(req, &codestralResp, startTime), nil
}

func (p *CodestralProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	codestralReq := p.convertRequest(req)
	codestralReq.Stream = true

	resp, err := p.makeAPICall(ctx, codestralReq)
	if err != nil {
		return nil, fmt.Errorf("Codestral streaming API call failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("Codestral API error: HTTP %d - failed to read response body: %v", resp.StatusCode, readErr)
		}
		return nil, fmt.Errorf("Codestral API error: HTTP %d - %s", resp.StatusCode, string(body))
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
					ID:           "stream-error-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "codestral",
					ProviderName: "Codestral",
					Content:      "",
					Confidence:   0.0,
					FinishReason: "error",
					CreatedAt:    time.Now(),
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

			var streamResp CodestralStreamResponse
			if err := json.Unmarshal(line, &streamResp); err != nil {
				continue
			}

			if len(streamResp.Choices) > 0 {
				delta := streamResp.Choices[0].Delta.Content
				if delta != "" {
					fullContent += delta

					chunkResp := &models.LLMResponse{
						ID:           streamResp.ID,
						RequestID:    req.ID,
						ProviderID:   "codestral",
						ProviderName: "Codestral",
						Content:      delta,
						Confidence:   0.8,
						TokensUsed:   1,
						ResponseTime: time.Since(startTime).Milliseconds(),
						CreatedAt:    time.Now(),
					}
					ch <- chunkResp
				}

				if streamResp.Choices[0].FinishReason != nil {
					break
				}
			}
		}

		finalResp := &models.LLMResponse{
			ID:           "stream-final-" + req.ID,
			RequestID:    req.ID,
			ProviderID:   "codestral",
			ProviderName: "Codestral",
			Content:      "",
			Confidence:   0.8,
			TokensUsed:   len(fullContent) / 4,
			ResponseTime: time.Since(startTime).Milliseconds(),
			FinishReason: "stop",
			CreatedAt:    time.Now(),
		}
		ch <- finalResp
	}()

	return ch, nil
}

func (p *CodestralProvider) convertRequest(req *models.LLMRequest) CodestralRequest {
	messages := make([]CodestralMessage, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, CodestralMessage{
			Role:    "system",
			Content: req.Prompt,
		})
	}

	for _, msg := range req.Messages {
		messages = append(messages, CodestralMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = CodestralMaxOutput
	} else if maxTokens > CodestralMaxOutput {
		maxTokens = CodestralMaxOutput
	}

	return CodestralRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
	}
}

func (p *CodestralProvider) convertResponse(req *models.LLMRequest, codestralResp *CodestralResponse, startTime time.Time) *models.LLMResponse {
	var content string
	var finishReason string

	if len(codestralResp.Choices) > 0 {
		content = codestralResp.Choices[0].Message.Content
		finishReason = codestralResp.Choices[0].FinishReason
	}

	confidence := p.calculateConfidence(content, finishReason)

	return &models.LLMResponse{
		ID:           codestralResp.ID,
		RequestID:    req.ID,
		ProviderID:   "codestral",
		ProviderName: "Codestral",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   codestralResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             codestralResp.Model,
			"prompt_tokens":     codestralResp.Usage.PromptTokens,
			"completion_tokens": codestralResp.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *CodestralProvider) calculateConfidence(content, finishReason string) float64 {
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

func (p *CodestralProvider) makeAPICall(ctx context.Context, req CodestralRequest) (*http.Response, error) {
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

		if isRetryableStatus(resp.StatusCode) && attempt < p.retryConfig.MaxRetries {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("HTTP %d: retryable error", resp.StatusCode)
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

func (p *CodestralProvider) waitWithJitter(ctx context.Context, delay time.Duration) {
	jitter := time.Duration(rand.Float64() * 0.1 * float64(delay))
	select {
	case <-ctx.Done():
	case <-time.After(delay + jitter):
	}
}

func (p *CodestralProvider) nextDelay(currentDelay time.Duration) time.Duration {
	nextDelay := time.Duration(float64(currentDelay) * p.retryConfig.Multiplier)
	if nextDelay > p.retryConfig.MaxDelay {
		nextDelay = p.retryConfig.MaxDelay
	}
	return nextDelay
}

func (p *CodestralProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: p.discoverer.DiscoverModels(),
		SupportedFeatures: []string{
			"text_completion",
			"chat",
			"streaming",
			"code_completion",
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
			MaxTokens:             CodestralMaxOutput,
			MaxInputLength:        CodestralMaxContext,
			MaxOutputLength:       CodestralMaxOutput,
			MaxConcurrentRequests: 10,
		},
		Metadata: map[string]string{
			"provider":     "Codestral",
			"model_family": "Mistral",
			"api_version":  "v1",
			"note":         i18n.Tr(context.Background(), "provider.codestral.description", nil),
		},
	}
}

func (p *CodestralProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string

	if p.apiKey == "" {
		// CONST-046 round-425: validation error routed through i18n.
		errors = append(errors, i18n.Tr(context.Background(),
			"llmprovider_validate_api_key_required", nil))
	}

	if p.baseURL == "" {
		// CONST-046 round-425: validation error routed through i18n.
		errors = append(errors, i18n.Tr(context.Background(),
			"llmprovider_validate_base_url_required", nil))
	}

	if p.model == "" {
		// CONST-046 round-425: validation error routed through i18n.
		errors = append(errors, i18n.Tr(context.Background(),
			"llmprovider_validate_model_required", nil))
	}

	return len(errors) == 0, errors
}

func (p *CodestralProvider) HealthCheck() error {
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
