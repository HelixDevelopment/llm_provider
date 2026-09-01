package cloudflare

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"time"

	"digital.vasic.llmprovider/pkg/discovery"
	"digital.vasic.llmprovider/pkg/i18n"
	"digital.vasic.llmprovider/pkg/models"
	"digital.vasic.llmprovider/pkg/settings"
)

const (
	CloudflareAPIURL     = "https://api.cloudflare.com/client/v4/accounts/%s/ai/v1/chat/completions"
	CloudflareModel      = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
	CloudflareModelsURL  = "https://api.cloudflare.com/client/v4/accounts/%s/ai/models"
	CloudflareMaxContext = 131072
	CloudflareMaxOutput  = 8192
)

type CloudflareProvider struct {
	apiKey      string
	accountID   string
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

type CloudflareRequest struct {
	Model       string              `json:"model"`
	Messages    []CloudflareMessage `json:"messages"`
	Temperature float64             `json:"temperature,omitempty"`
	MaxTokens   int                 `json:"max_tokens,omitempty"`
	TopP        float64             `json:"top_p,omitempty"`
	Stream      bool                `json:"stream,omitempty"`
}

type CloudflareMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type CloudflareResponse struct {
	Result   CloudflareResult  `json:"result"`
	Success  bool              `json:"success"`
	Errors   []CloudflareError `json:"errors"`
	Messages []interface{}     `json:"messages"`
}

type CloudflareResult struct {
	Response string          `json:"response"`
	ID       string          `json:"id"`
	Model    string          `json:"model"`
	Usage    CloudflareUsage `json:"usage"`
}

type CloudflareUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type CloudflareError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type CloudflareStreamResponse struct {
	Response string `json:"response"`
	ID       string `json:"id"`
	Model    string `json:"model"`
	Done     bool   `json:"done"`
}

// DefaultHTTPTimeout is the compiled fallback for this adapter's HTTP client.
// Override with LLMPROVIDER_CLOUDFLARE_TIMEOUT (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

func NewCloudflareProvider(apiKey, accountID, baseURL, model string) *CloudflareProvider {
	return NewCloudflareProviderWithRetry(apiKey, accountID, baseURL, model, DefaultRetryConfig())
}

func NewCloudflareProviderWithRetry(apiKey, accountID, baseURL, model string, retryConfig RetryConfig) *CloudflareProvider {
	if accountID == "" {
		accountID = os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	}
	if model == "" {
		// The compiled constant is a FALLBACK, not a decision this
		// library gets to keep making. A vendor retiring or rate-capping
		// a model must be answerable with an environment variable, not a
		// release. See pkg/settings: LLMPROVIDER_CLOUDFLARE_MODEL.
		model = settings.Model("cloudflare", CloudflareModel)
	}

	p := &CloudflareProvider{
		apiKey:    apiKey,
		accountID: accountID,
		baseURL:   baseURL,
		model:     model,
		// LLMPROVIDER_CLOUDFLARE_TIMEOUT.
		httpClient:  &http.Client{Timeout: settings.Timeout("cloudflare", DefaultHTTPTimeout)},
		retryConfig: retryConfig,
	}

	if accountID != "" {
		modelsURL := fmt.Sprintf(CloudflareModelsURL, accountID)
		p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
			ProviderName:   "cloudflare",
			ModelsEndpoint: modelsURL,
			ModelsDevID:    "cloudflare",
			APIKey:         apiKey,
			FallbackModels: []string{
				"@cf/meta/llama-4-scout-17b-16e-instruct",
				"@cf/meta/llama-3.3-70b-instruct-fp8-fast",
				"@cf/meta/llama-3.1-70b-instruct",
				"@cf/meta/llama-3.1-8b-instruct-fast",
				"@cf/meta/llama-3.1-8b-instruct",
				"@cf/meta/llama-3.2-3b-instruct",
				"@cf/meta/llama-3.2-1b-instruct",
				"@cf/meta/llama-guard-3-8b",
				"@cf/nvidia/nemotron-3-120b-a12b",
				"@cf/openai/gpt-oss-120b",
				"@cf/openai/gpt-oss-20b",
				"@cf/mistral/mistral-small-3.1-24b-instruct",
				"@cf/mistral/mistral-7b-instruct",
				"@cf/google/gemma-3-12b-it",
				"@cf/qwen/qwen3-30b-a3b-fp8",
				"@cf/qwen/qwq-32b",
				"@cf/qwen/qwen2.5-coder-32b-instruct",
				"@cf/deepseek/deepseek-r1-distill-qwen-32b",
				"@cf/ibm/granite-4.0-h-micro",
				"@cf/zai/glm-4.7-flash",
			},
		})
	}

	return p
}

func (p *CloudflareProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("cloudflare-%d", time.Now().UnixNano())
	}

	cReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, cReq)
	if err != nil {
		return nil, fmt.Errorf("Cloudflare API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Cloudflare API error: %d - %s", resp.StatusCode, string(body))
	}

	var cResp CloudflareResponse
	if err := json.Unmarshal(body, &cResp); err != nil {
		return nil, fmt.Errorf("failed to parse Cloudflare response: %w", err)
	}

	if !cResp.Success && len(cResp.Errors) > 0 {
		return nil, fmt.Errorf("Cloudflare API error: %s", cResp.Errors[0].Message)
	}

	if cResp.Result.Response == "" {
		return nil, fmt.Errorf("Cloudflare API returned empty response")
	}

	return p.convertResponse(req, &cResp, startTime), nil
}

func (p *CloudflareProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	cReq := p.convertRequest(req)
	cReq.Stream = true

	resp, err := p.makeAPICall(ctx, cReq)
	if err != nil {
		return nil, fmt.Errorf("Cloudflare streaming API call failed: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf("Cloudflare API error: HTTP %d - %s", resp.StatusCode, string(body))
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
					ProviderID:   "cloudflare",
					ProviderName: "Cloudflare",
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

			var streamResp CloudflareStreamResponse
			if err := json.Unmarshal(line, &streamResp); err != nil {
				continue
			}

			if streamResp.Response != "" {
				fullContent += streamResp.Response
				ch <- &models.LLMResponse{
					ID:           streamResp.ID,
					RequestID:    req.ID,
					ProviderID:   "cloudflare",
					ProviderName: "Cloudflare",
					Content:      streamResp.Response,
					Confidence:   0.8,
					TokensUsed:   1,
					ResponseTime: time.Since(startTime).Milliseconds(),
					CreatedAt:    time.Now(),
				}
			}

			if streamResp.Done {
				break
			}
		}

		ch <- &models.LLMResponse{
			ID:           "stream-final-" + req.ID,
			RequestID:    req.ID,
			ProviderID:   "cloudflare",
			ProviderName: "Cloudflare",
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

func (p *CloudflareProvider) convertRequest(req *models.LLMRequest) CloudflareRequest {
	messages := make([]CloudflareMessage, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, CloudflareMessage{Role: "system", Content: req.Prompt})
	}

	for _, msg := range req.Messages {
		messages = append(messages, CloudflareMessage{Role: msg.Role, Content: msg.Content})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = CloudflareMaxOutput
	} else if maxTokens > CloudflareMaxOutput {
		maxTokens = CloudflareMaxOutput
	}

	return CloudflareRequest{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
	}
}

func (p *CloudflareProvider) convertResponse(req *models.LLMRequest, cResp *CloudflareResponse, startTime time.Time) *models.LLMResponse {
	content := cResp.Result.Response
	confidence := 0.8

	if len(content) > 100 {
		confidence += 0.05
	}
	if confidence > 1.0 {
		confidence = 1.0
	}

	return &models.LLMResponse{
		ID:           cResp.Result.ID,
		RequestID:    req.ID,
		ProviderID:   "cloudflare",
		ProviderName: "Cloudflare",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   cResp.Result.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: "stop",
		Metadata: map[string]any{
			"model":             cResp.Result.Model,
			"prompt_tokens":     cResp.Result.Usage.PromptTokens,
			"completion_tokens": cResp.Result.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *CloudflareProvider) makeAPICall(ctx context.Context, req CloudflareRequest) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var apiURL string
	if p.baseURL != "" {
		apiURL = p.baseURL
	} else if p.accountID != "" {
		apiURL = fmt.Sprintf(CloudflareAPIURL, p.accountID)
	} else {
		return nil, fmt.Errorf("Cloudflare account ID or base URL is required")
	}

	var lastErr error
	delay := p.retryConfig.InitialDelay

	for attempt := 0; attempt <= p.retryConfig.MaxRetries; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(body))
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

func (p *CloudflareProvider) GetCapabilities() *models.ProviderCapabilities {
	supportedModels := []string{CloudflareModel}
	if p.discoverer != nil {
		supportedModels = p.discoverer.DiscoverModels()
	}

	return &models.ProviderCapabilities{
		SupportedModels:        supportedModels,
		SupportedFeatures:      []string{"text_completion", "chat", "streaming", "code_completion", "reasoning"},
		SupportsStreaming:      true,
		SupportsCodeCompletion: true,
		SupportsCodeAnalysis:   true,
		Limits: models.ModelLimits{
			MaxTokens:       CloudflareMaxContext,
			MaxInputLength:  CloudflareMaxContext,
			MaxOutputLength: CloudflareMaxOutput,
		},
		Metadata: map[string]string{
			"provider": "Cloudflare",
			"note":     "Cloudflare Workers AI",
		},
	}
}

func (p *CloudflareProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}
	if p.accountID == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_account_id_required", nil))
	}
	return len(errors) == 0, errors
}

func (p *CloudflareProvider) HealthCheck() error {
	if p.accountID == "" {
		return fmt.Errorf("Cloudflare account ID is required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	modelsURL := fmt.Sprintf(CloudflareModelsURL, p.accountID)
	req, _ := http.NewRequestWithContext(ctx, "GET", modelsURL, nil) //nolint:errcheck
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
