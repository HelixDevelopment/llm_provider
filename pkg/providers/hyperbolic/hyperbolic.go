package hyperbolic

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

// DefaultHTTPTimeout is the compiled fallback for this adapter's HTTP
// client. It is a starting point, not a decision the library keeps making:
// LLMPROVIDER_HYPERBOLIC_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

// modelsURL derives the /models endpoint from the configured baseURL so
// health checks honor operator overrides (proxies, mirrors, httptest in
// unit tests) instead of always hitting the hardcoded production URL.
func (p *HyperbolicProvider) modelsURL() string {
	return strings.TrimSuffix(p.baseURL, "/chat/completions") + "/models"
}

const (
	HyperbolicAPIURL     = "https://api.hyperbolic.xyz/v1/chat/completions"
	HyperbolicModel      = "meta-llama/Meta-Llama-3-8B-Instruct"
	HyperbolicModelsURL  = "https://api.hyperbolic.xyz/v1/models"
	HyperbolicMaxContext = 8192
	HyperbolicMaxOutput  = 4096
)

type HyperbolicProvider struct {
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

type HyperbolicRequest struct {
	Model       string              `json:"model"`
	Messages    []HyperbolicMessage `json:"messages"`
	Temperature float64             `json:"temperature,omitempty"`
	MaxTokens   int                 `json:"max_tokens,omitempty"`
	TopP        float64             `json:"top_p,omitempty"`
	Stream      bool                `json:"stream,omitempty"`
}

type HyperbolicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type HyperbolicResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []HyperbolicChoice `json:"choices"`
	Usage   HyperbolicUsage    `json:"usage"`
}

type HyperbolicChoice struct {
	Index        int               `json:"index"`
	Message      HyperbolicMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

type HyperbolicUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type HyperbolicStreamResponse struct {
	ID      string                   `json:"id"`
	Object  string                   `json:"object"`
	Created int64                    `json:"created"`
	Model   string                   `json:"model"`
	Choices []HyperbolicStreamChoice `json:"choices"`
}

type HyperbolicStreamChoice struct {
	Index        int               `json:"index"`
	Delta        HyperbolicMessage `json:"delta"`
	FinishReason *string           `json:"finish_reason"`
}

type HyperbolicErrorResponse struct {
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

func NewHyperbolicProvider(apiKey, baseURL, model string) *HyperbolicProvider {
	return NewHyperbolicProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewHyperbolicProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *HyperbolicProvider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_HYPERBOLIC_BASE_URL.
		baseURL = settings.BaseURL("hyperbolic", HyperbolicAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_HYPERBOLIC_MODEL overrides this compiled fallback.
		model = settings.Model("hyperbolic", HyperbolicModel)
	}

	p := &HyperbolicProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("hyperbolic", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "hyperbolic",
		ModelsEndpoint: HyperbolicModelsURL,
		ModelsDevID:    "hyperbolic",
		APIKey:         apiKey,
		FallbackModels: []string{
			"meta-llama/Meta-Llama-3-8B-Instruct",
			"meta-llama/Meta-Llama-3-70B-Instruct",
			"mistralai/Mistral-7B-Instruct-v0.1",
		},
	})

	return p
}

func (p *HyperbolicProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("hyperbolic-%d", time.Now().UnixNano())
	}

	hReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, hReq)
	if err != nil {
		return nil, fmt.Errorf("Hyperbolic API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp HyperbolicErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("Hyperbolic API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		return nil, fmt.Errorf("Hyperbolic API error: %d - %s", resp.StatusCode, string(body))
	}

	var hResp HyperbolicResponse
	if err := json.Unmarshal(body, &hResp); err != nil {
		return nil, fmt.Errorf("failed to parse Hyperbolic response: %w", err)
	}

	if len(hResp.Choices) == 0 {
		return nil, fmt.Errorf("Hyperbolic API returned no choices")
	}

	return p.convertResponse(req, &hResp, startTime), nil
}

func (p *HyperbolicProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	hReq := p.convertRequest(req)
	hReq.Stream = true

	resp, err := p.makeAPICall(ctx, hReq)
	if err != nil {
		return nil, fmt.Errorf("Hyperbolic streaming API call failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf("Hyperbolic API error: HTTP %d - %s", resp.StatusCode, string(body))
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
				ch <- &models.LLMResponse{
					ID:           "stream-error-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "hyperbolic",
					ProviderName: "Hyperbolic",
					FinishReason: "error",
					CreatedAt:    time.Now(),
				}
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

			var streamResp HyperbolicStreamResponse
			if err := json.Unmarshal(line, &streamResp); err != nil {
				continue
			}

			if len(streamResp.Choices) > 0 {
				delta := streamResp.Choices[0].Delta.Content
				if delta != "" {
					fullContent += delta
					ch <- &models.LLMResponse{
						ID:           streamResp.ID,
						RequestID:    req.ID,
						ProviderID:   "hyperbolic",
						ProviderName: "Hyperbolic",
						Content:      delta,
						Confidence:   0.8,
						TokensUsed:   1,
						ResponseTime: time.Since(startTime).Milliseconds(),
						CreatedAt:    time.Now(),
					}
				}

				if streamResp.Choices[0].FinishReason != nil {
					break
				}
			}
		}

		ch <- &models.LLMResponse{
			ID:           "stream-final-" + req.ID,
			RequestID:    req.ID,
			ProviderID:   "hyperbolic",
			ProviderName: "Hyperbolic",
			Content:      "",
			Confidence:   0.8,
			TokensUsed:   len(fullContent) / 4,
			ResponseTime: time.Since(startTime).Milliseconds(),
			FinishReason: "stop",
			CreatedAt:    time.Now(),
		}
	}()

	return ch, nil
}

func (p *HyperbolicProvider) convertRequest(req *models.LLMRequest) HyperbolicRequest {
	messages := make([]HyperbolicMessage, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, HyperbolicMessage{Role: "system", Content: req.Prompt})
	}

	for _, msg := range req.Messages {
		messages = append(messages, HyperbolicMessage{Role: msg.Role, Content: msg.Content})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = HyperbolicMaxOutput
	} else if maxTokens > HyperbolicMaxOutput {
		maxTokens = HyperbolicMaxOutput
	}

	return HyperbolicRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
	}
}

func (p *HyperbolicProvider) convertResponse(req *models.LLMRequest, hResp *HyperbolicResponse, startTime time.Time) *models.LLMResponse {
	var content, finishReason string
	if len(hResp.Choices) > 0 {
		content = hResp.Choices[0].Message.Content
		finishReason = hResp.Choices[0].FinishReason
	}

	confidence := 0.8
	if finishReason == "stop" {
		confidence += 0.1
	}
	if len(content) > 100 {
		confidence += 0.05
	}
	if confidence > 1.0 {
		confidence = 1.0
	}

	return &models.LLMResponse{
		ID:           hResp.ID,
		RequestID:    req.ID,
		ProviderID:   "hyperbolic",
		ProviderName: "Hyperbolic",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   hResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             hResp.Model,
			"prompt_tokens":     hResp.Usage.PromptTokens,
			"completion_tokens": hResp.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *HyperbolicProvider) makeAPICall(ctx context.Context, req HyperbolicRequest) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var lastErr error
	delay := p.retryConfig.InitialDelay

	for attempt := 0; attempt <= p.retryConfig.MaxRetries; attempt++ {
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
				jitter := time.Duration(rand.Float64() * 0.1 * float64(delay))
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-time.After(delay + jitter):
				}
				delay = time.Duration(float64(delay) * p.retryConfig.Multiplier)
				if delay > p.retryConfig.MaxDelay {
					delay = p.retryConfig.MaxDelay
				}
				continue
			}
			return nil, lastErr
		}

		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			if attempt < p.retryConfig.MaxRetries {
				_ = resp.Body.Close()
				lastErr = fmt.Errorf("HTTP %d: retryable error", resp.StatusCode)
				jitter := time.Duration(rand.Float64() * 0.1 * float64(delay))
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-time.After(delay + jitter):
				}
				delay = time.Duration(float64(delay) * p.retryConfig.Multiplier)
				if delay > p.retryConfig.MaxDelay {
					delay = p.retryConfig.MaxDelay
				}
				continue
			}
		}

		return resp, nil
	}

	return nil, fmt.Errorf("all retries failed: %w", lastErr)
}

func (p *HyperbolicProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels:        p.discoverer.DiscoverModels(),
		SupportedFeatures:      []string{"text_completion", "chat", "streaming", "code_completion"},
		SupportsStreaming:      true,
		SupportsCodeCompletion: true,
		SupportsCodeAnalysis:   true,
		Limits: models.ModelLimits{
			MaxTokens:       HyperbolicMaxOutput,
			MaxInputLength:  HyperbolicMaxContext,
			MaxOutputLength: HyperbolicMaxOutput,
		},
		Metadata: map[string]string{
			"provider": "Hyperbolic",
			"note":     i18n.Tr(context.Background(), "provider.hyperbolic.description", nil),
		},
	}
}

func (p *HyperbolicProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}
	return len(errors) == 0, errors
}

func (p *HyperbolicProvider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req, _ := http.NewRequestWithContext(ctx, "GET", p.modelsURL(), nil) //nolint:errcheck
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}
	return nil
}
