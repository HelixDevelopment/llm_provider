package siliconflow

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
// LLMPROVIDER_SILICONFLOW_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

// modelsURL derives the /models endpoint from the configured baseURL so
// health checks honor operator overrides (proxies, mirrors, httptest in
// unit tests) instead of always hitting the hardcoded production URL.
func (p *SiliconFlowProvider) modelsURL() string {
	return strings.TrimSuffix(p.baseURL, "/chat/completions") + "/models"
}

const (
	SiliconFlowAPIURL     = "https://api.siliconflow.cn/v1/chat/completions"
	SiliconFlowModel      = "Qwen/Qwen2.5-7B-Instruct"
	SiliconFlowModelsURL  = "https://api.siliconflow.cn/v1/models"
	SiliconFlowMaxContext = 32768
	SiliconFlowMaxOutput  = 8192
)

type SiliconFlowProvider struct {
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

type SiliconFlowRequest struct {
	Model       string               `json:"model"`
	Messages    []SiliconFlowMessage `json:"messages"`
	Temperature float64              `json:"temperature,omitempty"`
	MaxTokens   int                  `json:"max_tokens,omitempty"`
	TopP        float64              `json:"top_p,omitempty"`
	Stream      bool                 `json:"stream,omitempty"`
}

type SiliconFlowMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type SiliconFlowResponse struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []SiliconFlowChoice `json:"choices"`
	Usage   SiliconFlowUsage    `json:"usage"`
}

type SiliconFlowChoice struct {
	Index        int                `json:"index"`
	Message      SiliconFlowMessage `json:"message"`
	FinishReason string             `json:"finish_reason"`
}

type SiliconFlowUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type SiliconFlowStreamResponse struct {
	ID      string                    `json:"id"`
	Object  string                    `json:"object"`
	Created int64                     `json:"created"`
	Model   string                    `json:"model"`
	Choices []SiliconFlowStreamChoice `json:"choices"`
}

type SiliconFlowStreamChoice struct {
	Index        int                `json:"index"`
	Delta        SiliconFlowMessage `json:"delta"`
	FinishReason *string            `json:"finish_reason"`
}

type SiliconFlowErrorResponse struct {
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

func NewSiliconFlowProvider(apiKey, baseURL, model string) *SiliconFlowProvider {
	return NewSiliconFlowProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewSiliconFlowProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *SiliconFlowProvider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_SILICONFLOW_BASE_URL.
		baseURL = settings.BaseURL("siliconflow", SiliconFlowAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_SILICONFLOW_MODEL overrides this compiled fallback.
		model = settings.Model("siliconflow", SiliconFlowModel)
	}

	p := &SiliconFlowProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("siliconflow", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "siliconflow",
		ModelsEndpoint: SiliconFlowModelsURL,
		ModelsDevID:    "siliconflow",
		APIKey:         apiKey,
		FallbackModels: []string{
			"Qwen/Qwen2.5-7B-Instruct",
			"Qwen/Qwen2.5-72B-Instruct",
			"deepseek-ai/DeepSeek-V2.5",
			"meta-llama/Meta-Llama-3.1-8B-Instruct",
		},
	})

	return p
}

func (p *SiliconFlowProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("siliconflow-%d", time.Now().UnixNano())
	}

	sfReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, sfReq)
	if err != nil {
		return nil, fmt.Errorf("SiliconFlow API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp SiliconFlowErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("SiliconFlow API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		return nil, fmt.Errorf("SiliconFlow API error: %d - %s", resp.StatusCode, string(body))
	}

	var sfResp SiliconFlowResponse
	if err := json.Unmarshal(body, &sfResp); err != nil {
		return nil, fmt.Errorf("failed to parse SiliconFlow response: %w", err)
	}

	if len(sfResp.Choices) == 0 {
		return nil, fmt.Errorf("SiliconFlow API returned no choices")
	}

	return p.convertResponse(req, &sfResp, startTime), nil
}

func (p *SiliconFlowProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	sfReq := p.convertRequest(req)
	sfReq.Stream = true

	resp, err := p.makeAPICall(ctx, sfReq)
	if err != nil {
		return nil, fmt.Errorf("SiliconFlow streaming API call failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf("SiliconFlow API error: HTTP %d - %s", resp.StatusCode, string(body))
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
					ProviderID:   "siliconflow",
					ProviderName: "SiliconFlow",
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

			var streamResp SiliconFlowStreamResponse
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
						ProviderID:   "siliconflow",
						ProviderName: "SiliconFlow",
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
			ProviderID:   "siliconflow",
			ProviderName: "SiliconFlow",
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

func (p *SiliconFlowProvider) convertRequest(req *models.LLMRequest) SiliconFlowRequest {
	messages := make([]SiliconFlowMessage, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, SiliconFlowMessage{Role: "system", Content: req.Prompt})
	}

	for _, msg := range req.Messages {
		messages = append(messages, SiliconFlowMessage{Role: msg.Role, Content: msg.Content})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = SiliconFlowMaxOutput
	} else if maxTokens > SiliconFlowMaxOutput {
		maxTokens = SiliconFlowMaxOutput
	}

	return SiliconFlowRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
	}
}

func (p *SiliconFlowProvider) convertResponse(req *models.LLMRequest, sfResp *SiliconFlowResponse, startTime time.Time) *models.LLMResponse {
	var content, finishReason string
	if len(sfResp.Choices) > 0 {
		content = sfResp.Choices[0].Message.Content
		finishReason = sfResp.Choices[0].FinishReason
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
		ID:           sfResp.ID,
		RequestID:    req.ID,
		ProviderID:   "siliconflow",
		ProviderName: "SiliconFlow",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   sfResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             sfResp.Model,
			"prompt_tokens":     sfResp.Usage.PromptTokens,
			"completion_tokens": sfResp.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *SiliconFlowProvider) makeAPICall(ctx context.Context, req SiliconFlowRequest) (*http.Response, error) {
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

func (p *SiliconFlowProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels:        p.discoverer.DiscoverModels(),
		SupportedFeatures:      []string{"text_completion", "chat", "streaming", "code_completion"},
		SupportsStreaming:      true,
		SupportsCodeCompletion: true,
		SupportsCodeAnalysis:   true,
		Limits: models.ModelLimits{
			MaxTokens:       SiliconFlowMaxOutput,
			MaxInputLength:  SiliconFlowMaxContext,
			MaxOutputLength: SiliconFlowMaxOutput,
		},
		Metadata: map[string]string{
			"provider": "SiliconFlow",
			"note":     i18n.Tr(context.Background(), "provider.siliconflow.description", nil),
		},
	}
}

func (p *SiliconFlowProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}
	return len(errors) == 0, errors
}

func (p *SiliconFlowProvider) HealthCheck() error {
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
