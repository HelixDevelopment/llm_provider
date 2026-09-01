package sarvam

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
// LLMPROVIDER_SARVAM_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

// modelsURL derives the /models endpoint from the configured baseURL so
// health checks honor operator overrides (proxies, mirrors, httptest in
// unit tests) instead of always hitting the hardcoded production URL.
func (p *SarvamProvider) modelsURL() string {
	return strings.TrimSuffix(p.baseURL, "/chat/completions") + "/models"
}

const (
	SarvamAPIURL     = "https://api.sarvam.ai/v1/chat/completions"
	SarvamModel      = "sarvam-m"
	SarvamModelsURL  = "https://api.sarvam.ai/v1/models"
	SarvamMaxContext = 8192
	SarvamMaxOutput  = 4096
)

type SarvamProvider struct {
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

type SarvamRequest struct {
	Model       string          `json:"model"`
	Messages    []SarvamMessage `json:"messages"`
	Temperature float64         `json:"temperature,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	TopP        float64         `json:"top_p,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
}

type SarvamMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type SarvamResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []SarvamChoice `json:"choices"`
	Usage   SarvamUsage    `json:"usage"`
}

type SarvamChoice struct {
	Index        int           `json:"index"`
	Message      SarvamMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type SarvamUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type SarvamStreamResponse struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []SarvamStreamChoice `json:"choices"`
}

type SarvamStreamChoice struct {
	Index        int           `json:"index"`
	Delta        SarvamMessage `json:"delta"`
	FinishReason *string       `json:"finish_reason"`
}

type SarvamErrorResponse struct {
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

func NewSarvamProvider(apiKey, baseURL, model string) *SarvamProvider {
	return NewSarvamProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewSarvamProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *SarvamProvider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_SARVAM_BASE_URL.
		baseURL = settings.BaseURL("sarvam", SarvamAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_SARVAM_MODEL overrides this compiled fallback.
		model = settings.Model("sarvam", SarvamModel)
	}

	p := &SarvamProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("sarvam", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	// ModelsEndpoint is DERIVED from the configured base URL, not pinned to the
	// production constant — the same CONST-051(B) config-injection fix already
	// applied to HealthCheck via modelsURL() at the top of this file. Pinning it
	// meant a caller could point this provider at a mirror, a proxy or a test
	// double for completions while discovery silently kept talking to
	// api.sarvam.ai, which is both a surprise in production and the reason
	// TestGetCapabilities could only ever be an availability probe: no fixture
	// could reach the code path.
	//
	// Production behaviour is unchanged. With the default base URL this
	// evaluates to exactly SarvamModelsURL, and TestModelsURLMatchesConstant
	// asserts that equality rather than leaving it to inspection.
	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "sarvam",
		ModelsEndpoint: p.modelsURL(),
		ModelsDevID:    "sarvam",
		APIKey:         apiKey,
		FallbackModels: []string{
			"sarvam-m",
			"sarvam-2b",
			"sarvam-7b",
		},
	})

	return p
}

func (p *SarvamProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("sarvam-%d", time.Now().UnixNano())
	}

	sReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, sReq)
	if err != nil {
		return nil, fmt.Errorf("Sarvam API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp SarvamErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("Sarvam API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		return nil, fmt.Errorf("Sarvam API error: %d - %s", resp.StatusCode, string(body))
	}

	var sResp SarvamResponse
	if err := json.Unmarshal(body, &sResp); err != nil {
		return nil, fmt.Errorf("failed to parse Sarvam response: %w", err)
	}

	if len(sResp.Choices) == 0 {
		return nil, fmt.Errorf("Sarvam API returned no choices")
	}

	return p.convertResponse(req, &sResp, startTime), nil
}

func (p *SarvamProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	sReq := p.convertRequest(req)
	sReq.Stream = true

	resp, err := p.makeAPICall(ctx, sReq)
	if err != nil {
		return nil, fmt.Errorf("Sarvam streaming API call failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("Sarvam API error: HTTP %d (failed to read body: %v)", resp.StatusCode, readErr)
		}
		return nil, fmt.Errorf("Sarvam API error: HTTP %d - %s", resp.StatusCode, string(body))
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
					ProviderID:   "sarvam",
					ProviderName: "Sarvam",
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

			var streamResp SarvamStreamResponse
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
						ProviderID:   "sarvam",
						ProviderName: "Sarvam",
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
			ProviderID:   "sarvam",
			ProviderName: "Sarvam",
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

func (p *SarvamProvider) convertRequest(req *models.LLMRequest) SarvamRequest {
	messages := make([]SarvamMessage, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, SarvamMessage{Role: "system", Content: req.Prompt})
	}

	for _, msg := range req.Messages {
		messages = append(messages, SarvamMessage{Role: msg.Role, Content: msg.Content})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = SarvamMaxOutput
	} else if maxTokens > SarvamMaxOutput {
		maxTokens = SarvamMaxOutput
	}

	return SarvamRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
	}
}

func (p *SarvamProvider) convertResponse(req *models.LLMRequest, sResp *SarvamResponse, startTime time.Time) *models.LLMResponse {
	var content, finishReason string
	if len(sResp.Choices) > 0 {
		content = sResp.Choices[0].Message.Content
		finishReason = sResp.Choices[0].FinishReason
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
		ID:           sResp.ID,
		RequestID:    req.ID,
		ProviderID:   "sarvam",
		ProviderName: "Sarvam",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   sResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             sResp.Model,
			"prompt_tokens":     sResp.Usage.PromptTokens,
			"completion_tokens": sResp.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *SarvamProvider) makeAPICall(ctx context.Context, req SarvamRequest) (*http.Response, error) {
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

func (p *SarvamProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels:        p.discoverer.DiscoverModels(),
		SupportedFeatures:      []string{"text_completion", "chat", "streaming", "code_completion"},
		SupportsStreaming:      true,
		SupportsCodeCompletion: true,
		SupportsCodeAnalysis:   true,
		Limits: models.ModelLimits{
			MaxTokens:       SarvamMaxOutput,
			MaxInputLength:  SarvamMaxContext,
			MaxOutputLength: SarvamMaxOutput,
		},
		Metadata: map[string]string{
			"provider": "Sarvam",
			"note":     i18n.Tr(context.Background(), "provider.sarvam.description", nil),
		},
	}
}

func (p *SarvamProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}
	return len(errors) == 0, errors
}

func (p *SarvamProvider) HealthCheck() error {
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
