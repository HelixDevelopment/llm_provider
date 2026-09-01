package nvidia

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
	NvidiaAPIURL     = "https://integrate.api.nvidia.com/v1/chat/completions"
	NvidiaModel      = "meta/llama-3.1-8b-instruct"
	NvidiaModelsURL  = "https://integrate.api.nvidia.com/v1/models"
	NvidiaMaxContext = 131072
	NvidiaMaxOutput  = 16384
)

type NvidiaProvider struct {
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

type NvidiaRequest struct {
	Model       string          `json:"model"`
	Messages    []NvidiaMessage `json:"messages"`
	Temperature float64         `json:"temperature,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	TopP        float64         `json:"top_p,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
}

type NvidiaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type NvidiaResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []NvidiaChoice `json:"choices"`
	Usage   NvidiaUsage    `json:"usage"`
}

type NvidiaChoice struct {
	Index        int           `json:"index"`
	Message      NvidiaMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type NvidiaUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type NvidiaStreamResponse struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []NvidiaStreamChoice `json:"choices"`
}

type NvidiaStreamChoice struct {
	Index        int           `json:"index"`
	Delta        NvidiaMessage `json:"delta"`
	FinishReason *string       `json:"finish_reason"`
}

type NvidiaErrorResponse struct {
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

func NewNvidiaProvider(apiKey, baseURL, model string) *NvidiaProvider {
	return NewNvidiaProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewNvidiaProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *NvidiaProvider {
	if baseURL == "" {
		baseURL = NvidiaAPIURL
	}
	if model == "" {
		// The compiled constant is a FALLBACK, not a decision this
		// library gets to keep making. A vendor retiring or rate-capping
		// a model must be answerable with an environment variable, not a
		// release. See pkg/settings: LLMPROVIDER_NVIDIA_MODEL.
		model = settings.Model("nvidia", NvidiaModel)
	}

	p := &NvidiaProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "nvidia",
		ModelsEndpoint: NvidiaModelsURL,
		ModelsDevID:    "nvidia",
		APIKey:         apiKey,
		FallbackModels: []string{
			"meta/llama-3.1-8b-instruct",
			"meta/llama-3.1-70b-instruct",
			"meta/llama-3.2-1b-instruct",
			"meta/llama-3.2-3b-instruct",
		},
	})

	return p
}

func (p *NvidiaProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("nvidia-%d", time.Now().UnixNano())
	}

	nvidiaReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, nvidiaReq)
	if err != nil {
		return nil, fmt.Errorf("NVIDIA API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp NvidiaErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("NVIDIA API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		return nil, fmt.Errorf("NVIDIA API error: %d - %s", resp.StatusCode, string(body))
	}

	var nvidiaResp NvidiaResponse
	if err := json.Unmarshal(body, &nvidiaResp); err != nil {
		return nil, fmt.Errorf("failed to parse NVIDIA response: %w", err)
	}

	if len(nvidiaResp.Choices) == 0 {
		return nil, fmt.Errorf("NVIDIA API returned no choices")
	}

	return p.convertResponse(req, &nvidiaResp, startTime), nil
}

func (p *NvidiaProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	nvidiaReq := p.convertRequest(req)
	nvidiaReq.Stream = true

	resp, err := p.makeAPICall(ctx, nvidiaReq)
	if err != nil {
		return nil, fmt.Errorf("NVIDIA streaming API call failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf("NVIDIA API error: HTTP %d - %s", resp.StatusCode, string(body))
	}

	ch := make(chan *models.LLMResponse)

	go func() {
		defer func() { _ = resp.Body.Close() }()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		var fullContent string

		// processLine handles one SSE line; returns stop=true when the stream
		// is complete (DONE marker or finish_reason set). It is invoked even
		// for the final partial line ReadBytes returns alongside io.EOF, so a
		// last "data: {...}" / "data: [DONE]" chunk written without a trailing
		// newline is never dropped (mirrors the openrouter reference loop).
		processLine := func(raw []byte) (stop bool) {
			line := bytes.TrimSpace(raw)
			if !bytes.HasPrefix(line, []byte("data: ")) {
				return false
			}
			line = bytes.TrimPrefix(line, []byte("data: "))

			if bytes.Equal(line, []byte("[DONE]")) {
				return true
			}

			var streamResp NvidiaStreamResponse
			if err := json.Unmarshal(line, &streamResp); err != nil {
				return false
			}

			if len(streamResp.Choices) > 0 {
				delta := streamResp.Choices[0].Delta.Content
				if delta != "" {
					fullContent += delta
					ch <- &models.LLMResponse{
						ID:           streamResp.ID,
						RequestID:    req.ID,
						ProviderID:   "nvidia",
						ProviderName: "NVIDIA NIM",
						Content:      delta,
						Confidence:   0.8,
						TokensUsed:   1,
						ResponseTime: time.Since(startTime).Milliseconds(),
						CreatedAt:    time.Now(),
					}
				}

				if streamResp.Choices[0].FinishReason != nil {
					return true
				}
			}
			return false
		}

		for {
			line, err := reader.ReadBytes('\n')

			// Process any complete line content returned alongside the error
			// (ReadBytes returns the data read so far on EOF without a newline).
			// break (not return) so the post-loop final frame is still emitted.
			if len(line) > 0 {
				if processLine(line) {
					break
				}
			}

			if err != nil {
				if err == io.EOF {
					break
				}
				ch <- &models.LLMResponse{
					ID:           "stream-error-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "nvidia",
					ProviderName: "NVIDIA NIM",
					FinishReason: "error",
					CreatedAt:    time.Now(),
				}
				return
			}
		}

		ch <- &models.LLMResponse{
			ID:           "stream-final-" + req.ID,
			RequestID:    req.ID,
			ProviderID:   "nvidia",
			ProviderName: "NVIDIA NIM",
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

func (p *NvidiaProvider) convertRequest(req *models.LLMRequest) NvidiaRequest {
	messages := make([]NvidiaMessage, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, NvidiaMessage{Role: "system", Content: req.Prompt})
	}

	for _, msg := range req.Messages {
		messages = append(messages, NvidiaMessage{Role: msg.Role, Content: msg.Content})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = NvidiaMaxOutput
	} else if maxTokens > NvidiaMaxOutput {
		maxTokens = NvidiaMaxOutput
	}

	return NvidiaRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
	}
}

func (p *NvidiaProvider) convertResponse(req *models.LLMRequest, nvidiaResp *NvidiaResponse, startTime time.Time) *models.LLMResponse {
	var content, finishReason string
	if len(nvidiaResp.Choices) > 0 {
		content = nvidiaResp.Choices[0].Message.Content
		finishReason = nvidiaResp.Choices[0].FinishReason
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
		ID:           nvidiaResp.ID,
		RequestID:    req.ID,
		ProviderID:   "nvidia",
		ProviderName: "NVIDIA NIM",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   nvidiaResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             nvidiaResp.Model,
			"prompt_tokens":     nvidiaResp.Usage.PromptTokens,
			"completion_tokens": nvidiaResp.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *NvidiaProvider) makeAPICall(ctx context.Context, req NvidiaRequest) (*http.Response, error) {
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

func (p *NvidiaProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels:        p.discoverer.DiscoverModels(),
		SupportedFeatures:      []string{"text_completion", "chat", "streaming", "code_completion"},
		SupportsStreaming:      true,
		SupportsCodeCompletion: true,
		SupportsCodeAnalysis:   true,
		Limits: models.ModelLimits{
			MaxTokens:       NvidiaMaxOutput,
			MaxInputLength:  NvidiaMaxContext,
			MaxOutputLength: NvidiaMaxOutput,
		},
		Metadata: map[string]string{
			"provider": "NVIDIA NIM",
			"note":     i18n.Tr(context.Background(), "provider.nvidia.description", nil),
		},
	}
}

func (p *NvidiaProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}
	return len(errors) == 0, errors
}

// modelsURL derives the models endpoint from the configured base URL so the
// health check honors a custom baseURL (CONST-051(B) config-injection) instead
// of a hardcoded production constant.
func (p *NvidiaProvider) modelsURL() string {
	return strings.TrimSuffix(p.baseURL, "/chat/completions") + "/models"
}

func (p *NvidiaProvider) HealthCheck() error {
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
