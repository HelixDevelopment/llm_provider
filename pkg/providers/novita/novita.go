package novita

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

const (
	NovitaAPIURL     = "https://api.novita.ai/v3/openai/chat/completions"
	NovitaModel      = "meta-llama/llama-3-8b-instruct"
	NovitaModelsURL  = "https://api.novita.ai/v3/openai/models"
	NovitaMaxContext = 8192
	NovitaMaxOutput  = 4096
)

type NovitaProvider struct {
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

type NovitaRequest struct {
	Model       string          `json:"model"`
	Messages    []NovitaMessage `json:"messages"`
	Temperature float64         `json:"temperature,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	TopP        float64         `json:"top_p,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
}

type NovitaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type NovitaResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []NovitaChoice `json:"choices"`
	Usage   NovitaUsage    `json:"usage"`
}

type NovitaChoice struct {
	Index        int           `json:"index"`
	Message      NovitaMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type NovitaUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type NovitaStreamResponse struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []NovitaStreamChoice `json:"choices"`
}

type NovitaStreamChoice struct {
	Index        int           `json:"index"`
	Delta        NovitaMessage `json:"delta"`
	FinishReason *string       `json:"finish_reason"`
}

type NovitaErrorResponse struct {
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

func NewNovitaProvider(apiKey, baseURL, model string) *NovitaProvider {
	return NewNovitaProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewNovitaProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *NovitaProvider {
	if baseURL == "" {
		baseURL = NovitaAPIURL
	}
	if model == "" {
		// The compiled constant is a FALLBACK, not a decision this
		// library gets to keep making. A vendor retiring or rate-capping
		// a model must be answerable with an environment variable, not a
		// release. See pkg/settings: LLMPROVIDER_NOVITA_MODEL.
		model = settings.Model("novita", NovitaModel)
	}

	p := &NovitaProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
		retryConfig: retryConfig,
	}

	// ModelsEndpoint is DERIVED from the configured base URL, not pinned to the
	// production constant — the same CONST-051(B) config-injection fix already
	// applied to HealthCheck via modelsURL() at the bottom of this file. Pinning
	// it meant a caller could point this provider at a mirror, a proxy or a test
	// double for completions while discovery silently kept talking to
	// api.novita.ai, which is both a surprise in production and the reason
	// TestGetCapabilities could only ever be an availability probe: no fixture
	// could reach the code path.
	//
	// Production behaviour is unchanged. With the default base URL this
	// evaluates to exactly NovitaModelsURL, and TestModelsURLMatchesConstant
	// asserts that equality rather than leaving it to inspection.
	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "novita",
		ModelsEndpoint: p.modelsURL(),
		ModelsDevID:    "novita",
		APIKey:         apiKey,
		FallbackModels: []string{
			"meta-llama/llama-3-8b-instruct",
			"meta-llama/llama-3-70b-instruct",
			"mistralai/mistral-7b-instruct",
		},
	})

	return p
}

func (p *NovitaProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("novita-%d", time.Now().UnixNano())
	}

	nReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, nReq)
	if err != nil {
		return nil, fmt.Errorf("Novita API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp NovitaErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("Novita API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		return nil, fmt.Errorf("Novita API error: %d - %s", resp.StatusCode, string(body))
	}

	var nResp NovitaResponse
	if err := json.Unmarshal(body, &nResp); err != nil {
		return nil, fmt.Errorf("failed to parse Novita response: %w", err)
	}

	if len(nResp.Choices) == 0 {
		return nil, fmt.Errorf("Novita API returned no choices")
	}

	return p.convertResponse(req, &nResp, startTime), nil
}

func (p *NovitaProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	nReq := p.convertRequest(req)
	nReq.Stream = true

	resp, err := p.makeAPICall(ctx, nReq)
	if err != nil {
		return nil, fmt.Errorf("Novita streaming API call failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf("Novita API error: HTTP %d - %s", resp.StatusCode, string(body))
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
					ProviderID:   "novita",
					ProviderName: "Novita",
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

			var streamResp NovitaStreamResponse
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
						ProviderID:   "novita",
						ProviderName: "Novita",
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
			ProviderID:   "novita",
			ProviderName: "Novita",
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

func (p *NovitaProvider) convertRequest(req *models.LLMRequest) NovitaRequest {
	messages := make([]NovitaMessage, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, NovitaMessage{Role: "system", Content: req.Prompt})
	}

	for _, msg := range req.Messages {
		messages = append(messages, NovitaMessage{Role: msg.Role, Content: msg.Content})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = NovitaMaxOutput
	} else if maxTokens > NovitaMaxOutput {
		maxTokens = NovitaMaxOutput
	}

	return NovitaRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
	}
}

func (p *NovitaProvider) convertResponse(req *models.LLMRequest, nResp *NovitaResponse, startTime time.Time) *models.LLMResponse {
	var content, finishReason string
	if len(nResp.Choices) > 0 {
		content = nResp.Choices[0].Message.Content
		finishReason = nResp.Choices[0].FinishReason
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
		ID:           nResp.ID,
		RequestID:    req.ID,
		ProviderID:   "novita",
		ProviderName: "Novita",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   nResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             nResp.Model,
			"prompt_tokens":     nResp.Usage.PromptTokens,
			"completion_tokens": nResp.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *NovitaProvider) makeAPICall(ctx context.Context, req NovitaRequest) (*http.Response, error) {
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

func (p *NovitaProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels:        p.discoverer.DiscoverModels(),
		SupportedFeatures:      []string{"text_completion", "chat", "streaming", "code_completion"},
		SupportsStreaming:      true,
		SupportsCodeCompletion: true,
		SupportsCodeAnalysis:   true,
		Limits: models.ModelLimits{
			MaxTokens:       NovitaMaxOutput,
			MaxInputLength:  NovitaMaxContext,
			MaxOutputLength: NovitaMaxOutput,
		},
		Metadata: map[string]string{
			"provider": "Novita",
			"note":     i18n.Tr(context.Background(), "provider.novita.description", nil),
		},
	}
}

func (p *NovitaProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}
	return len(errors) == 0, errors
}

func (p *NovitaProvider) HealthCheck() error {
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

// modelsURL derives the /models endpoint from the configured baseURL so
// health checks honor operator overrides (proxies, mirrors, httptest in
// unit tests) instead of always hitting the hardcoded production URL.
func (p *NovitaProvider) modelsURL() string {
	return strings.TrimSuffix(p.baseURL, "/chat/completions") + "/models"
}
