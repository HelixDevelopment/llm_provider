package zhipu

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
// LLMPROVIDER_ZHIPU_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

const (
	ZhipuAPIURL     = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
	ZhipuModel      = "glm-4-flash"
	ZhipuModelsURL  = "https://open.bigmodel.cn/api/paas/v4/models"
	ZhipuMaxContext = 128000
	ZhipuMaxOutput  = 4096
)

type ZhipuProvider struct {
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

type ZhipuRequest struct {
	Model       string         `json:"model"`
	Messages    []ZhipuMessage `json:"messages"`
	Temperature float64        `json:"temperature,omitempty"`
	MaxTokens   int            `json:"max_tokens,omitempty"`
	TopP        float64        `json:"top_p,omitempty"`
	Stream      bool           `json:"stream,omitempty"`
}

type ZhipuMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ZhipuResponse struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ZhipuChoice `json:"choices"`
	Usage   ZhipuUsage    `json:"usage"`
}

type ZhipuChoice struct {
	Index        int          `json:"index"`
	Message      ZhipuMessage `json:"message"`
	FinishReason string       `json:"finish_reason"`
}

type ZhipuUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ZhipuStreamResponse struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []ZhipuStreamChoice `json:"choices"`
}

type ZhipuStreamChoice struct {
	Index        int          `json:"index"`
	Delta        ZhipuMessage `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

type ZhipuErrorResponse struct {
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

func NewZhipuProvider(apiKey, baseURL, model string) *ZhipuProvider {
	return NewZhipuProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewZhipuProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *ZhipuProvider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_ZHIPU_BASE_URL.
		baseURL = settings.BaseURL("zhipu", ZhipuAPIURL)
	}
	if model == "" {
		// The compiled constant is a FALLBACK, not a decision this
		// library gets to keep making. A vendor retiring or rate-capping
		// a model must be answerable with an environment variable, not a
		// release. See pkg/settings: LLMPROVIDER_ZHIPU_MODEL.
		model = settings.Model("zhipu", ZhipuModel)
	}

	p := &ZhipuProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("zhipu", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "zhipu",
		ModelsEndpoint: ZhipuModelsURL,
		ModelsDevID:    "zhipu",
		APIKey:         apiKey,
		FallbackModels: []string{
			"glm-4-flash",
			"glm-4",
			"glm-4-plus",
			"glm-4-air",
			"glm-4-airx",
			"glm-4-long",
		},
	})

	return p
}

func (p *ZhipuProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("zhipu-%d", time.Now().UnixNano())
	}

	zReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, zReq)
	if err != nil {
		return nil, fmt.Errorf("Zhipu API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp ZhipuErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("Zhipu API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		return nil, fmt.Errorf("Zhipu API error: %d - %s", resp.StatusCode, string(body))
	}

	var zResp ZhipuResponse
	if err := json.Unmarshal(body, &zResp); err != nil {
		return nil, fmt.Errorf("failed to parse Zhipu response: %w", err)
	}

	if len(zResp.Choices) == 0 {
		return nil, fmt.Errorf("Zhipu API returned no choices")
	}

	return p.convertResponse(req, &zResp, startTime), nil
}

func (p *ZhipuProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	zReq := p.convertRequest(req)
	zReq.Stream = true

	resp, err := p.makeAPICall(ctx, zReq)
	if err != nil {
		return nil, fmt.Errorf("Zhipu streaming API call failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf("Zhipu API error: HTTP %d - %s", resp.StatusCode, string(body))
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
					ProviderID:   "zhipu",
					ProviderName: "Zhipu",
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

			var streamResp ZhipuStreamResponse
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
						ProviderID:   "zhipu",
						ProviderName: "Zhipu",
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
			ProviderID:   "zhipu",
			ProviderName: "Zhipu",
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

func (p *ZhipuProvider) convertRequest(req *models.LLMRequest) ZhipuRequest {
	messages := make([]ZhipuMessage, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, ZhipuMessage{Role: "system", Content: req.Prompt})
	}

	for _, msg := range req.Messages {
		messages = append(messages, ZhipuMessage{Role: msg.Role, Content: msg.Content})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = ZhipuMaxOutput
	} else if maxTokens > ZhipuMaxOutput {
		maxTokens = ZhipuMaxOutput
	}

	return ZhipuRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
	}
}

func (p *ZhipuProvider) convertResponse(req *models.LLMRequest, zResp *ZhipuResponse, startTime time.Time) *models.LLMResponse {
	var content, finishReason string
	if len(zResp.Choices) > 0 {
		content = zResp.Choices[0].Message.Content
		finishReason = zResp.Choices[0].FinishReason
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
		ID:           zResp.ID,
		RequestID:    req.ID,
		ProviderID:   "zhipu",
		ProviderName: "Zhipu",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   zResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             zResp.Model,
			"prompt_tokens":     zResp.Usage.PromptTokens,
			"completion_tokens": zResp.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *ZhipuProvider) makeAPICall(ctx context.Context, req ZhipuRequest) (*http.Response, error) {
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

func (p *ZhipuProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels:        p.discoverer.DiscoverModels(),
		SupportedFeatures:      []string{"text_completion", "chat", "streaming", "code_completion"},
		SupportsStreaming:      true,
		SupportsCodeCompletion: true,
		SupportsCodeAnalysis:   true,
		Limits: models.ModelLimits{
			MaxTokens:       ZhipuMaxOutput,
			MaxInputLength:  ZhipuMaxContext,
			MaxOutputLength: ZhipuMaxOutput,
		},
		Metadata: map[string]string{
			"provider": "Zhipu",
			"note":     i18n.Tr(context.Background(), "provider.zhipu.description", nil),
		},
	}
}

func (p *ZhipuProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}
	return len(errors) == 0, errors
}

func (p *ZhipuProvider) HealthCheck() error {
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

// modelsURL returns the Zhipu /models endpoint derived from the configured
// baseURL. Mirrors the Vulavula pattern: strip a trailing `/chat/completions`
// and append `/models` so the health check respects operator-supplied
// overrides (including httptest servers in unit tests).
func (p *ZhipuProvider) modelsURL() string {
	trimmed := strings.TrimSuffix(p.baseURL, "/chat/completions")
	return trimmed + "/models"
}
