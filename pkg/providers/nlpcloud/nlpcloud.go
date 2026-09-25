package nlpcloud

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
// LLMPROVIDER_NLPCLOUD_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

// modelsURL derives the /models endpoint from the configured baseURL so
// health checks honor operator overrides (proxies, mirrors, httptest in
// unit tests) instead of always hitting the hardcoded production URL.
func (p *NLPCloudProvider) modelsURL() string {
	return strings.TrimSuffix(p.baseURL, "/chat/completions") + "/models"
}

const (
	NLPCloudAPIURL     = "https://api.nlpcloud.io/v1/gpu"
	NLPCloudModel      = "finetuned-llama-3-70b"
	NLPCloudModelsURL  = "https://api.nlpcloud.io/v1/models"
	NLPCloudMaxContext = 8192
	NLPCloudMaxOutput  = 4096
)

type NLPCloudProvider struct {
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

type NLPCloudRequest struct {
	Text        string  `json:"text"`
	Temperature float64 `json:"temperature,omitempty"`
	MaxLength   int     `json:"max_length,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
}

type NLPCloudResponse struct {
	GeneratedText string `json:"generated_text"`
}

type NLPCloudChatRequest struct {
	Model       string            `json:"model"`
	Messages    []NLPCloudMessage `json:"messages"`
	Temperature float64           `json:"temperature,omitempty"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	TopP        float64           `json:"top_p,omitempty"`
}

type NLPCloudMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type NLPCloudChatResponse struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []NLPCloudChoice `json:"choices"`
	Usage   NLPCloudUsage    `json:"usage"`
}

type NLPCloudChoice struct {
	Index        int             `json:"index"`
	Message      NLPCloudMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type NLPCloudUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type NLPCloudStreamResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []NLPCloudStreamChoice `json:"choices"`
}

type NLPCloudStreamChoice struct {
	Index        int             `json:"index"`
	Delta        NLPCloudMessage `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
}

type NLPCloudErrorResponse struct {
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

func NewNLPCloudProvider(apiKey, baseURL, model string) *NLPCloudProvider {
	return NewNLPCloudProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewNLPCloudProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *NLPCloudProvider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_NLPCLOUD_BASE_URL.
		baseURL = settings.BaseURL("nlpcloud", NLPCloudAPIURL)
	}
	if model == "" {
		// The compiled constant is a FALLBACK, not a decision this
		// library gets to keep making. A vendor retiring or rate-capping
		// a model must be answerable with an environment variable, not a
		// release. See pkg/settings: LLMPROVIDER_NLPCLOUD_MODEL.
		model = settings.Model("nlpcloud", NLPCloudModel)
	}

	p := &NLPCloudProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("nlpcloud", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "nlpcloud",
		ModelsEndpoint: NLPCloudModelsURL,
		ModelsDevID:    "nlpcloud",
		APIKey:         apiKey,
		FallbackModels: []string{
			"finetuned-llama-3-70b",
			"llama-3-70b",
			"mixtral-8x7b",
			"openchat-3-5",
		},
	})

	return p
}

func (p *NLPCloudProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("nlpcloud-%d", time.Now().UnixNano())
	}

	nReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, nReq)
	if err != nil {
		return nil, fmt.Errorf("NLPCloud API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp NLPCloudErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("NLPCloud API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		return nil, fmt.Errorf("NLPCloud API error: %d - %s", resp.StatusCode, string(body))
	}

	var nResp NLPCloudChatResponse
	if err := json.Unmarshal(body, &nResp); err != nil {
		return nil, fmt.Errorf("failed to parse NLPCloud response: %w", err)
	}

	if len(nResp.Choices) == 0 {
		return nil, fmt.Errorf("NLPCloud API returned no choices")
	}

	return p.convertResponse(req, &nResp, startTime), nil
}

func (p *NLPCloudProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	nReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, nReq)
	if err != nil {
		return nil, fmt.Errorf("NLPCloud streaming API call failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf("NLPCloud API error: HTTP %d - %s", resp.StatusCode, string(body))
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
					ProviderID:   "nlpcloud",
					ProviderName: "NLPCloud",
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

			var streamResp NLPCloudStreamResponse
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
						ProviderID:   "nlpcloud",
						ProviderName: "NLPCloud",
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
			ProviderID:   "nlpcloud",
			ProviderName: "NLPCloud",
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

func (p *NLPCloudProvider) convertRequest(req *models.LLMRequest) NLPCloudChatRequest {
	messages := make([]NLPCloudMessage, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, NLPCloudMessage{Role: "system", Content: req.Prompt})
	}

	for _, msg := range req.Messages {
		messages = append(messages, NLPCloudMessage{Role: msg.Role, Content: msg.Content})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = NLPCloudMaxOutput
	} else if maxTokens > NLPCloudMaxOutput {
		maxTokens = NLPCloudMaxOutput
	}

	return NLPCloudChatRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
	}
}

func (p *NLPCloudProvider) convertResponse(req *models.LLMRequest, nResp *NLPCloudChatResponse, startTime time.Time) *models.LLMResponse {
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
		ProviderID:   "nlpcloud",
		ProviderName: "NLPCloud",
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

func (p *NLPCloudProvider) makeAPICall(ctx context.Context, req NLPCloudChatRequest) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	apiURL := fmt.Sprintf("%s/%s/chat", p.baseURL, p.model)

	var lastErr error
	delay := p.retryConfig.InitialDelay

	for attempt := 0; attempt <= p.retryConfig.MaxRetries; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(body))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Token "+p.apiKey)
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

func (p *NLPCloudProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels:        p.discoverer.DiscoverModels(),
		SupportedFeatures:      []string{"text_completion", "chat", "streaming", "code_completion"},
		SupportsStreaming:      true,
		SupportsCodeCompletion: true,
		SupportsCodeAnalysis:   true,
		Limits: models.ModelLimits{
			MaxTokens:       NLPCloudMaxOutput,
			MaxInputLength:  NLPCloudMaxContext,
			MaxOutputLength: NLPCloudMaxOutput,
		},
		Metadata: map[string]string{
			"provider": "NLPCloud",
			"note":     i18n.Tr(context.Background(), "provider.nlpcloud.description", nil),
		},
	}
}

func (p *NLPCloudProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}
	return len(errors) == 0, errors
}

func (p *NLPCloudProvider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req, _ := http.NewRequestWithContext(ctx, "GET", p.modelsURL(), nil) //nolint:errcheck
	req.Header.Set("Authorization", "Token "+p.apiKey)

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
