package githubmodels

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
// LLMPROVIDER_GITHUBMODELS_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

const (
	// GitHubModelsAPIURL is the base URL for GitHub Models chat completions
	GitHubModelsAPIURL = "https://models.github.ai/inference/chat/completions"
	// GitHubModelsModelsURL is the URL for listing available models
	GitHubModelsModelsURL = "https://models.github.ai/inference/models"
	// GitHubModelsDefault is the default model
	GitHubModelsDefault = "openai/gpt-4.1"
)

// GitHubModelsProvider implements the LLMProvider interface for GitHub Models
type GitHubModelsProvider struct {
	apiKey      string
	baseURL     string
	model       string
	httpClient  *http.Client
	retryConfig RetryConfig
	discoverer  *discovery.Discoverer
}

// RetryConfig defines retry behavior for API calls
type RetryConfig struct {
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

// Request represents a GitHub Models chat completion request (OpenAI compatible)
type Request struct {
	Model            string          `json:"model"`
	Messages         []Message       `json:"messages"`
	Temperature      float64         `json:"temperature,omitempty"`
	MaxTokens        int             `json:"max_tokens,omitempty"`
	TopP             float64         `json:"top_p,omitempty"`
	Stream           bool            `json:"stream,omitempty"`
	Stop             []string        `json:"stop,omitempty"`
	PresencePenalty  float64         `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64         `json:"frequency_penalty,omitempty"`
	Tools            []Tool          `json:"tools,omitempty"`
	ToolChoice       any             `json:"tool_choice,omitempty"`
	ResponseFormat   *ResponseFormat `json:"response_format,omitempty"`
	Seed             *int            `json:"seed,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// Tool represents a tool definition
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function represents a function definition
type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ToolCall represents a tool call in a response
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall represents function call details
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ResponseFormat specifies the response format
type ResponseFormat struct {
	Type string `json:"type"` // "text" or "json_object"
}

// Response represents a GitHub Models chat completion response
type Response struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             Usage    `json:"usage"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}

// Choice represents a completion choice
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage represents token usage
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// StreamResponse represents a streaming response chunk
type StreamResponse struct {
	ID                string         `json:"id"`
	Object            string         `json:"object"`
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	Choices           []StreamChoice `json:"choices"`
	Usage             *Usage         `json:"usage,omitempty"`
	SystemFingerprint string         `json:"system_fingerprint,omitempty"`
}

// StreamChoice represents a streaming choice
type StreamChoice struct {
	Index        int     `json:"index"`
	Delta        Message `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// DefaultRetryConfig returns sensible defaults for GitHub Models
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

// NewGitHubModelsProvider creates a new GitHub Models provider
func NewGitHubModelsProvider(apiKey, baseURL, model string) *GitHubModelsProvider {
	return NewGitHubModelsProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

// NewGitHubModelsProviderWithRetry creates a new GitHub Models provider with
// custom retry config
func NewGitHubModelsProviderWithRetry(
	apiKey, baseURL, model string,
	retryConfig RetryConfig,
) *GitHubModelsProvider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_GITHUBMODELS_BASE_URL.
		baseURL = settings.BaseURL("githubmodels", GitHubModelsAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_GITHUBMODELS_MODEL overrides this compiled fallback.
		model = settings.Model("githubmodels", GitHubModelsDefault)
	}

	p := &GitHubModelsProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("githubmodels", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "github-models",
		ModelsEndpoint: GitHubModelsModelsURL,
		ModelsDevID:    "", // No models.dev entry for GitHub Models
		APIKey:         apiKey,
		ExtraHeaders: map[string]string{
			"X-GitHub-Api-Version": "2022-11-28",
		},
		FallbackModels: []string{
			// OpenAI models
			"openai/gpt-5",
			"openai/gpt-4.1",
			"openai/gpt-4.1-mini",
			"openai/gpt-4.1-nano",
			"openai/gpt-4o",
			"openai/gpt-4o-mini",
			"openai/o4-mini",
			"openai/o3",
			"openai/o3-mini",
			"openai/o1",
			"openai/o1-mini",
			// DeepSeek models
			"DeepSeek/DeepSeek-V3-0324",
			"DeepSeek/DeepSeek-R1",
			// Meta models
			"Meta/Llama-4-Scout-17B-16E-Instruct",
			"Meta/Llama-4-Maverick-17B-128E-Instruct",
			"Meta/Llama-3.3-70B-Instruct",
			"Meta/Llama-3.1-405B-Instruct",
			"Meta/Llama-3.1-70B-Instruct",
			// Microsoft models
			"Microsoft/Phi-4-reasoning",
			"Microsoft/Phi-4-multimodal-instruct",
			"Microsoft/Phi-4",
			// Mistral models
			"Mistral/Mistral-Large-2",
			"Mistral/Mistral-Small-3.1-24B-Instruct",
			// Cohere models
			"Cohere/Command-A",
			// AI21 models
			"AI21-Labs/AI21-Jamba-1.5-Large",
		},
		// GitHub Models uses publisher/model-name format; accept all
		ModelFilter: func(modelID string) bool {
			return modelID != ""
		},
	})

	return p
}

// Complete sends a completion request to GitHub Models
func (p *GitHubModelsProvider) Complete(
	ctx context.Context,
	req *models.LLMRequest,
) (*models.LLMResponse, error) {
	startTime := time.Now()
	apiReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, apiReq)
	if err != nil {
		return nil, fmt.Errorf("GitHub Models API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"GitHub Models API error: %d - %s",
			resp.StatusCode, string(body),
		)
	}

	var apiResp Response
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return p.convertResponse(req, &apiResp, startTime), nil
}

// CompleteStream sends a streaming completion request to GitHub Models
func (p *GitHubModelsProvider) CompleteStream(
	ctx context.Context,
	req *models.LLMRequest,
) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()
	apiReq := p.convertRequest(req)
	apiReq.Stream = true

	resp, err := p.makeAPICall(ctx, apiReq)
	if err != nil {
		return nil, fmt.Errorf(
			"GitHub Models streaming API call failed: %w", err,
		)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf(
			"GitHub Models API error: %d - %s",
			resp.StatusCode, string(body),
		)
	}

	ch := make(chan *models.LLMResponse)
	go func() {
		defer func() { _ = resp.Body.Close() }()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		var fullContent strings.Builder

		for {
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				ch <- &models.LLMResponse{
					ID:           "stream-error-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "github-models",
					ProviderName: "GitHub Models",
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

			if string(line) == "[DONE]" {
				ch <- &models.LLMResponse{
					ID:           "stream-final-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "github-models",
					ProviderName: "GitHub Models",
					Content:      fullContent.String(),
					Confidence:   0.9,
					ResponseTime: time.Since(startTime).Milliseconds(),
					FinishReason: "stop",
					CreatedAt:    time.Now(),
				}
				return
			}

			var streamResp StreamResponse
			if err := json.Unmarshal(line, &streamResp); err != nil {
				continue
			}

			if len(streamResp.Choices) > 0 {
				delta := streamResp.Choices[0].Delta
				if delta.Content != "" {
					fullContent.WriteString(delta.Content)
					ch <- &models.LLMResponse{
						ID:           streamResp.ID,
						RequestID:    req.ID,
						ProviderID:   "github-models",
						ProviderName: "GitHub Models",
						Content:      delta.Content,
						FinishReason: "",
						CreatedAt:    time.Now(),
					}
				}
			}
		}
	}()

	return ch, nil
}

// HealthCheck verifies provider connectivity by querying the models endpoint
func (p *GitHubModelsProvider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(
		ctx, "GET", GitHubModelsModelsURL, nil,
	)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	httpReq.Header.Set("X-GitHub-Api-Version", "2022-11-28")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed: status %d", resp.StatusCode)
	}
	return nil
}

// GetCapabilities returns provider capabilities
func (p *GitHubModelsProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: p.discoverer.DiscoverModels(),
		SupportedFeatures: []string{
			"chat", "streaming", "tools", "vision", "json_mode",
			"code_completion", "reasoning", "multi_provider",
		},
		SupportedRequestTypes:   []string{"chat", "completion"},
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          true,
		SupportsTools:           true,
		SupportsReasoning:       true,
		SupportsCodeCompletion:  true,
		SupportsCodeAnalysis:    true,
		Limits: models.ModelLimits{
			MaxTokens:             128000,
			MaxInputLength:        128000,
			MaxOutputLength:       16384,
			MaxConcurrentRequests: 50,
		},
		Metadata: map[string]string{
			"provider":       "github-models",
			"multi_provider": "true",
		},
	}
}

// ValidateConfig validates provider configuration
func (p *GitHubModelsProvider) ValidateConfig(
	config map[string]interface{},
) (bool, []string) {
	var errors []string

	if p.apiKey == "" {
		// CONST-046 round-441: validation error routed through i18n.
		errors = append(errors, i18n.Tr(context.Background(),
			"llmprovider_validate_api_key_required_githubmodels", nil))
	}

	return len(errors) == 0, errors
}

// GetName returns the provider name
func (p *GitHubModelsProvider) GetName() string {
	return "github-models"
}

// GetProviderType returns the provider type identifier
func (p *GitHubModelsProvider) GetProviderType() string {
	return "github-models"
}

// GetModel returns the current model
func (p *GitHubModelsProvider) GetModel() string {
	return p.model
}

// SetModel sets the model
func (p *GitHubModelsProvider) SetModel(model string) {
	p.model = model
}

// convertRequest converts LLMRequest to GitHub Models API format
func (p *GitHubModelsProvider) convertRequest(req *models.LLMRequest) Request {
	messages := make([]Message, 0, len(req.Messages)+1)

	// Add system prompt
	if req.Prompt != "" {
		messages = append(messages, Message{
			Role:    "system",
			Content: req.Prompt,
		})
	}

	// Add conversation messages
	for _, msg := range req.Messages {
		messages = append(messages, Message{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	// Get max tokens with default
	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096
	}

	apiReq := Request{
		Model:       p.model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stop:        req.ModelParams.StopSequences,
	}

	// Override model if specified in request params
	if req.ModelParams.Model != "" {
		apiReq.Model = req.ModelParams.Model
	}

	// Convert tools
	if len(req.Tools) > 0 {
		apiReq.Tools = make([]Tool, len(req.Tools))
		for i, t := range req.Tools {
			apiReq.Tools[i] = Tool{
				Type: t.Type,
				Function: Function{
					Name:        t.Function.Name,
					Description: t.Function.Description,
					Parameters:  t.Function.Parameters,
				},
			}
		}
		apiReq.ToolChoice = req.ToolChoice
	}

	return apiReq
}

// convertResponse converts GitHub Models API response to LLMResponse
func (p *GitHubModelsProvider) convertResponse(
	req *models.LLMRequest,
	resp *Response,
	startTime time.Time,
) *models.LLMResponse {
	var content string
	var finishReason string
	var toolCalls []models.ToolCall

	if len(resp.Choices) > 0 {
		content = resp.Choices[0].Message.Content
		finishReason = resp.Choices[0].FinishReason

		// Parse tool calls
		if len(resp.Choices[0].Message.ToolCalls) > 0 {
			toolCalls = make(
				[]models.ToolCall,
				len(resp.Choices[0].Message.ToolCalls),
			)
			for i, tc := range resp.Choices[0].Message.ToolCalls {
				toolCalls[i] = models.ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: models.ToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
			if finishReason == "" || finishReason == "stop" {
				finishReason = "tool_calls"
			}
		}
	}

	confidence := p.calculateConfidence(content, finishReason)

	metadata := map[string]any{
		"model":             resp.Model,
		"prompt_tokens":     resp.Usage.PromptTokens,
		"completion_tokens": resp.Usage.CompletionTokens,
	}

	return &models.LLMResponse{
		ID:           resp.ID,
		RequestID:    req.ID,
		ProviderID:   "github-models",
		ProviderName: "GitHub Models",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   resp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		ToolCalls:    toolCalls,
		Metadata:     metadata,
		CreatedAt:    time.Now(),
	}
}

func (p *GitHubModelsProvider) calculateConfidence(
	content, finishReason string,
) float64 {
	confidence := 0.85
	switch finishReason {
	case "stop":
		confidence += 0.1
	case "length":
		confidence -= 0.1
	case "content_filter":
		confidence -= 0.3
	}
	if len(content) > 100 {
		confidence += 0.03
	}
	if confidence > 1.0 {
		confidence = 1.0
	}
	if confidence < 0.0 {
		confidence = 0.0
	}
	return confidence
}

func (p *GitHubModelsProvider) makeAPICall(
	ctx context.Context,
	req Request,
) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var lastErr error
	for attempt := 0; attempt <= p.retryConfig.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := p.calculateBackoff(attempt)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		httpReq, err := http.NewRequestWithContext(
			ctx, "POST", p.baseURL, bytes.NewReader(body),
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
		httpReq.Header.Set("X-GitHub-Api-Version", "2022-11-28")

		resp, err := p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = err
			continue
		}

		// Check for retryable status codes
		if resp.StatusCode == 429 ||
			resp.StatusCode == 500 ||
			resp.StatusCode == 502 ||
			resp.StatusCode == 503 ||
			resp.StatusCode == 504 {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf(
				"retryable error: status %d", resp.StatusCode,
			)
			continue
		}

		return resp, nil
	}

	return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

func (p *GitHubModelsProvider) calculateBackoff(attempt int) time.Duration {
	delay := p.retryConfig.InitialDelay
	for i := 1; i < attempt; i++ {
		delay = time.Duration(
			float64(delay) * p.retryConfig.Multiplier,
		)
		if delay > p.retryConfig.MaxDelay {
			delay = p.retryConfig.MaxDelay
			break
		}
	}
	// Add jitter (10% of delay)
	jitter := time.Duration(
		rand.Float64() * float64(delay) * 0.1, // #nosec G404
	)
	return delay + jitter
}
