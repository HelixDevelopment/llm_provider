package vulavula

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
// LLMPROVIDER_VULAVULA_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

const (
	VulavulaAPIURL     = "https://api.vulavula.ai/v1/chat/completions"
	VulavulaModel      = "vulavula-1.5"
	VulavulaModelsURL  = "https://api.vulavula.ai/v1/models"
	VulavulaMaxContext = 32768
	VulavulaMaxOutput  = 4096
)

type VulavulaProvider struct {
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

type VulavulaRequest struct {
	Model       string            `json:"model"`
	Messages    []VulavulaMessage `json:"messages"`
	Temperature float64           `json:"temperature,omitempty"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	TopP        float64           `json:"top_p,omitempty"`
	Stream      bool              `json:"stream,omitempty"`
}

type VulavulaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type VulavulaResponse struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []VulavulaChoice `json:"choices"`
	Usage   VulavulaUsage    `json:"usage"`
}

type VulavulaChoice struct {
	Index        int             `json:"index"`
	Message      VulavulaMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type VulavulaUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type VulavulaStreamResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []VulavulaStreamChoice `json:"choices"`
}

type VulavulaStreamChoice struct {
	Index        int             `json:"index"`
	Delta        VulavulaMessage `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
}

type VulavulaErrorResponse struct {
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

func NewVulavulaProvider(apiKey, baseURL, model string) *VulavulaProvider {
	return NewVulavulaProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewVulavulaProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *VulavulaProvider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_VULAVULA_BASE_URL.
		baseURL = settings.BaseURL("vulavula", VulavulaAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_VULAVULA_MODEL overrides this compiled fallback.
		model = settings.Model("vulavula", VulavulaModel)
	}

	p := &VulavulaProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("vulavula", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "vulavula",
		ModelsEndpoint: VulavulaModelsURL,
		ModelsDevID:    "vulavula",
		APIKey:         apiKey,
		FallbackModels: []string{
			"vulavula-1.5",
			"vulavula-1",
		},
	})

	return p
}

func (p *VulavulaProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("vulavula-%d", time.Now().UnixNano())
	}

	vReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, vReq)
	if err != nil {
		return nil, fmt.Errorf("Vulavula API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp VulavulaErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("Vulavula API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		return nil, fmt.Errorf("Vulavula API error: %d - %s", resp.StatusCode, string(body))
	}

	var vResp VulavulaResponse
	if err := json.Unmarshal(body, &vResp); err != nil {
		return nil, fmt.Errorf("failed to parse Vulavula response: %w", err)
	}

	if len(vResp.Choices) == 0 {
		return nil, fmt.Errorf("Vulavula API returned no choices")
	}

	return p.convertResponse(req, &vResp, startTime), nil
}

func (p *VulavulaProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	vReq := p.convertRequest(req)
	vReq.Stream = true

	resp, err := p.makeAPICall(ctx, vReq)
	if err != nil {
		return nil, fmt.Errorf("Vulavula streaming API call failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf("Vulavula API error: HTTP %d - %s", resp.StatusCode, string(body))
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
					ProviderID:   "vulavula",
					ProviderName: "Vulavula",
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

			var streamResp VulavulaStreamResponse
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
						ProviderID:   "vulavula",
						ProviderName: "Vulavula",
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
			ProviderID:   "vulavula",
			ProviderName: "Vulavula",
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

func (p *VulavulaProvider) convertRequest(req *models.LLMRequest) VulavulaRequest {
	messages := make([]VulavulaMessage, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, VulavulaMessage{Role: "system", Content: req.Prompt})
	}

	for _, msg := range req.Messages {
		messages = append(messages, VulavulaMessage{Role: msg.Role, Content: msg.Content})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = VulavulaMaxOutput
	} else if maxTokens > VulavulaMaxOutput {
		maxTokens = VulavulaMaxOutput
	}

	return VulavulaRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
	}
}

func (p *VulavulaProvider) convertResponse(req *models.LLMRequest, vResp *VulavulaResponse, startTime time.Time) *models.LLMResponse {
	var content, finishReason string
	if len(vResp.Choices) > 0 {
		content = vResp.Choices[0].Message.Content
		finishReason = vResp.Choices[0].FinishReason
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
		ID:           vResp.ID,
		RequestID:    req.ID,
		ProviderID:   "vulavula",
		ProviderName: "Vulavula",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   vResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             vResp.Model,
			"prompt_tokens":     vResp.Usage.PromptTokens,
			"completion_tokens": vResp.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *VulavulaProvider) makeAPICall(ctx context.Context, req VulavulaRequest) (*http.Response, error) {
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

func (p *VulavulaProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels:        p.discoverer.DiscoverModels(),
		SupportedFeatures:      []string{"text_completion", "chat", "streaming", "code_completion"},
		SupportsStreaming:      true,
		SupportsCodeCompletion: true,
		SupportsCodeAnalysis:   true,
		Limits: models.ModelLimits{
			MaxTokens:       VulavulaMaxOutput,
			MaxInputLength:  VulavulaMaxContext,
			MaxOutputLength: VulavulaMaxOutput,
		},
		Metadata: map[string]string{
			"provider": "Vulavula",
			"note":     i18n.Tr(context.Background(), "provider.vulavula.description", nil),
		},
	}
}

func (p *VulavulaProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}
	return len(errors) == 0, errors
}

func (p *VulavulaProvider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", p.modelsURL(), nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}
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

// modelsURL returns the Vulavula /models endpoint derived from the
// configured baseURL. Derives it by stripping a trailing `/chat/completions`
// segment (production default) and appending `/models`; callers that pass a
// test-server URL get the test server's root plus `/models`, which works
// because the test handler responds to any path.
func (p *VulavulaProvider) modelsURL() string {
	trimmed := strings.TrimSuffix(p.baseURL, "/chat/completions")
	return trimmed + "/models"
}
