package modal

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
func (p *ModalProvider) modelsURL() string {
	return strings.TrimSuffix(p.baseURL, "/chat/completions") + "/models"
}

const (
	ModalAPIURL     = "https://api.modal.com/v1/chat/completions"
	ModalModel      = "llama-3.1-8b"
	ModalModelsURL  = "https://api.modal.com/v1/models"
	ModalMaxContext = 131072
	ModalMaxOutput  = 4096
)

type ModalProvider struct {
	apiKey      string
	apiKeyID    string
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

type ModalRequest struct {
	Model       string         `json:"model"`
	Messages    []ModalMessage `json:"messages"`
	Temperature float64        `json:"temperature,omitempty"`
	MaxTokens   int            `json:"max_tokens,omitempty"`
	TopP        float64        `json:"top_p,omitempty"`
	Stream      bool           `json:"stream,omitempty"`
}

type ModalMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ModalResponse struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ModalChoice `json:"choices"`
	Usage   ModalUsage    `json:"usage"`
}

type ModalChoice struct {
	Index        int          `json:"index"`
	Message      ModalMessage `json:"message"`
	FinishReason string       `json:"finish_reason"`
}

type ModalUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ModalStreamResponse struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []ModalStreamChoice `json:"choices"`
}

type ModalStreamChoice struct {
	Index        int          `json:"index"`
	Delta        ModalMessage `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

type ModalErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

// DefaultHTTPTimeout is the compiled fallback for this adapter's HTTP client.
// Override with LLMPROVIDER_MODAL_TIMEOUT (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

func NewModalProvider(apiKey, apiKeyID, baseURL, model string) *ModalProvider {
	return NewModalProviderWithRetry(apiKey, apiKeyID, baseURL, model, DefaultRetryConfig())
}

func NewModalProviderWithRetry(apiKey, apiKeyID, baseURL, model string, retryConfig RetryConfig) *ModalProvider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_MODAL_BASE_URL.
		baseURL = settings.BaseURL("modal", ModalAPIURL)
	}
	if model == "" {
		// The compiled constant is a FALLBACK, not a decision this
		// library gets to keep making. A vendor retiring or rate-capping
		// a model must be answerable with an environment variable, not a
		// release. See pkg/settings: LLMPROVIDER_MODAL_MODEL.
		model = settings.Model("modal", ModalModel)
	}

	p := &ModalProvider{
		apiKey:   apiKey,
		apiKeyID: apiKeyID,
		baseURL:  baseURL,
		model:    model,
		// LLMPROVIDER_MODAL_TIMEOUT.
		httpClient:  &http.Client{Timeout: settings.Timeout("modal", DefaultHTTPTimeout)},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "modal",
		ModelsEndpoint: ModalModelsURL,
		ModelsDevID:    "modal",
		APIKey:         apiKey,
		FallbackModels: []string{
			"llama-3.1-8b",
			"llama-3.1-70b",
			"mistral-7b",
		},
	})

	return p
}

func (p *ModalProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("modal-%d", time.Now().UnixNano())
	}

	mReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, mReq)
	if err != nil {
		return nil, fmt.Errorf("Modal API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp ModalErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("Modal API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		return nil, fmt.Errorf("Modal API error: %d - %s", resp.StatusCode, string(body))
	}

	var mResp ModalResponse
	if err := json.Unmarshal(body, &mResp); err != nil {
		return nil, fmt.Errorf("failed to parse Modal response: %w", err)
	}

	if len(mResp.Choices) == 0 {
		return nil, fmt.Errorf("Modal API returned no choices")
	}

	return p.convertResponse(req, &mResp, startTime), nil
}

func (p *ModalProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	mReq := p.convertRequest(req)
	mReq.Stream = true

	resp, err := p.makeAPICall(ctx, mReq)
	if err != nil {
		return nil, fmt.Errorf("Modal streaming API call failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf("Modal API error: HTTP %d - %s", resp.StatusCode, string(body))
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
					ProviderID:   "modal",
					ProviderName: "Modal",
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

			var streamResp ModalStreamResponse
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
						ProviderID:   "modal",
						ProviderName: "Modal",
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
			ProviderID:   "modal",
			ProviderName: "Modal",
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

func (p *ModalProvider) convertRequest(req *models.LLMRequest) ModalRequest {
	messages := make([]ModalMessage, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, ModalMessage{Role: "system", Content: req.Prompt})
	}

	for _, msg := range req.Messages {
		messages = append(messages, ModalMessage{Role: msg.Role, Content: msg.Content})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = ModalMaxOutput
	} else if maxTokens > ModalMaxOutput {
		maxTokens = ModalMaxOutput
	}

	return ModalRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
	}
}

func (p *ModalProvider) convertResponse(req *models.LLMRequest, mResp *ModalResponse, startTime time.Time) *models.LLMResponse {
	var content, finishReason string
	if len(mResp.Choices) > 0 {
		content = mResp.Choices[0].Message.Content
		finishReason = mResp.Choices[0].FinishReason
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
		ID:           mResp.ID,
		RequestID:    req.ID,
		ProviderID:   "modal",
		ProviderName: "Modal",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   mResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             mResp.Model,
			"prompt_tokens":     mResp.Usage.PromptTokens,
			"completion_tokens": mResp.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *ModalProvider) makeAPICall(ctx context.Context, req ModalRequest) (*http.Response, error) {
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
		if p.apiKeyID != "" {
			httpReq.Header.Set("Modal-Key-ID", p.apiKeyID)
		}
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

func (p *ModalProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels:        p.discoverer.DiscoverModels(),
		SupportedFeatures:      []string{"text_completion", "chat", "streaming", "code_completion"},
		SupportsStreaming:      true,
		SupportsCodeCompletion: true,
		SupportsCodeAnalysis:   true,
		Limits: models.ModelLimits{
			MaxTokens:       ModalMaxOutput,
			MaxInputLength:  ModalMaxContext,
			MaxOutputLength: ModalMaxOutput,
		},
		Metadata: map[string]string{
			"provider": "Modal",
			"note":     i18n.Tr(context.Background(), "provider.modal.description", nil),
		},
	}
}

func (p *ModalProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}
	return len(errors) == 0, errors
}

func (p *ModalProvider) HealthCheck() error {
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
