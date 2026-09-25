package together

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
// LLMPROVIDER_TOGETHER_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

const (
	// TogetherAPIURL is the base URL for Together AI API
	TogetherAPIURL = "https://api.together.xyz/v1/chat/completions"
	// TogetherModelsURL is the URL for listing models
	TogetherModelsURL = "https://api.together.xyz/v1/models"
	// DefaultModel is the default Together model
	DefaultModel = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
)

// Provider implements the LLMProvider interface for Together AI
type Provider struct {
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

// Request represents a Together AI chat completion request (OpenAI compatible)
type Request struct {
	Model             string          `json:"model"`
	Messages          []Message       `json:"messages"`
	Temperature       float64         `json:"temperature,omitempty"`
	MaxTokens         int             `json:"max_tokens,omitempty"`
	TopP              float64         `json:"top_p,omitempty"`
	TopK              int             `json:"top_k,omitempty"`
	Stream            bool            `json:"stream,omitempty"`
	Stop              []string        `json:"stop,omitempty"`
	RepetitionPenalty float64         `json:"repetition_penalty,omitempty"`
	Tools             []Tool          `json:"tools,omitempty"`
	ToolChoice        any             `json:"tool_choice,omitempty"`
	ResponseFormat    *ResponseFormat `json:"response_format,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
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
	Type   string         `json:"type"` // "text" or "json_object"
	Schema map[string]any `json:"schema,omitempty"`
}

// Response represents a Together AI chat completion response
type Response struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
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
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
	Usage   *Usage         `json:"usage,omitempty"`
}

// StreamChoice represents a streaming choice
type StreamChoice struct {
	Index        int     `json:"index"`
	Delta        Message `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// DefaultRetryConfig returns sensible defaults
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

// NewProvider creates a new Together AI provider
func NewProvider(apiKey, baseURL, model string) *Provider {
	return NewProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

// NewProviderWithRetry creates a new Together AI provider with custom retry config
func NewProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *Provider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_TOGETHER_BASE_URL.
		baseURL = settings.BaseURL("together", TogetherAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_TOGETHER_MODEL overrides this compiled fallback.
		model = settings.Model("together", DefaultModel)
	}

	p := &Provider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("together", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "together",
		ModelsEndpoint: TogetherModelsURL,
		ModelsDevID:    "together",
		APIKey:         apiKey,
		FallbackModels: []string{
			// Llama models
			"meta-llama/Llama-3.3-70B-Instruct-Turbo",
			"meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
			"meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
			"meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
			"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
			"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
			// Qwen models
			"Qwen/Qwen2.5-72B-Instruct-Turbo",
			"Qwen/Qwen2.5-7B-Instruct-Turbo",
			"Qwen/QwQ-32B-Preview",
			// Mistral models
			"mistralai/Mistral-Small-24B-Instruct-2501",
			"mistralai/Mixtral-8x22B-Instruct-v0.1",
			"mistralai/Mixtral-8x7B-Instruct-v0.1",
			// DeepSeek models
			"deepseek-ai/DeepSeek-R1",
			"deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
			"deepseek-ai/DeepSeek-V3",
			// Google models
			"google/gemma-2-27b-it",
			"google/gemma-2-9b-it",
			// NVIDIA models
			"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
			// Databricks models
			"databricks/dbrx-instruct",
		},
	})

	return p
}

// Complete sends a completion request
func (p *Provider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	apiReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, apiReq)
	if err != nil {
		return nil, fmt.Errorf("Together AI API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Together AI API error: %d - %s", resp.StatusCode, string(body))
	}

	var apiResp Response
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(apiResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in Together AI response")
	}

	return p.convertResponse(req, &apiResp, startTime), nil
}

// CompleteStream sends a streaming completion request
func (p *Provider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()
	apiReq := p.convertRequest(req)
	apiReq.Stream = true

	resp, err := p.makeAPICall(ctx, apiReq)
	if err != nil {
		return nil, fmt.Errorf("Together AI streaming API call failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("Together AI API error: %d (failed to read body: %v)", resp.StatusCode, readErr)
		}
		return nil, fmt.Errorf("Together AI API error: %d - %s", resp.StatusCode, string(body))
	}

	ch := make(chan *models.LLMResponse)
	go func() {
		defer func() { _ = resp.Body.Close() }()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		var fullContent strings.Builder

		for {
			line, err := reader.ReadBytes('\n')
			// On EOF, ReadBytes returns any bytes read so far (a final
			// "data: {...}" or "data: [DONE]" line that arrived without a
			// trailing newline). Process that line BEFORE breaking, otherwise
			// the last chunk is silently dropped.
			eof := false
			if err != nil {
				if err == io.EOF {
					eof = true
				} else {
					ch <- &models.LLMResponse{
						ID:           "stream-error-" + req.ID,
						RequestID:    req.ID,
						ProviderID:   "together",
						ProviderName: "Together AI",
						FinishReason: "error",
						CreatedAt:    time.Now(),
					}
					return
				}
			}

			line = bytes.TrimSpace(line)
			if !bytes.HasPrefix(line, []byte("data: ")) {
				if eof {
					break
				}
				continue
			}
			line = bytes.TrimPrefix(line, []byte("data: "))

			if string(line) == "[DONE]" {
				ch <- &models.LLMResponse{
					ID:           "stream-final-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "together",
					ProviderName: "Together AI",
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
				if eof {
					break
				}
				continue
			}

			if len(streamResp.Choices) > 0 {
				delta := streamResp.Choices[0].Delta
				if delta.Content != "" {
					fullContent.WriteString(delta.Content)
					ch <- &models.LLMResponse{
						ID:           streamResp.ID,
						RequestID:    req.ID,
						ProviderID:   "together",
						ProviderName: "Together AI",
						Content:      delta.Content,
						FinishReason: "",
						CreatedAt:    time.Now(),
					}
				}
			}

			if eof {
				break
			}
		}
	}()

	return ch, nil
}

// HealthCheck verifies provider connectivity
// modelsURL derives the models endpoint from the configured base URL so the
// health check honors a custom baseURL (CONST-051(B) config-injection) instead
// of a hardcoded production constant.
func (p *Provider) modelsURL() string {
	return strings.TrimSuffix(p.baseURL, "/chat/completions") + "/models"
}

func (p *Provider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(ctx, "GET", p.modelsURL(), nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

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
func (p *Provider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: p.discoverer.DiscoverModels(),
		SupportedFeatures: []string{
			"chat", "streaming", "tools", "vision", "json_mode",
			"code_completion", "embeddings",
		},
		SupportedRequestTypes:   []string{"chat", "completion", "embed"},
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          true,
		SupportsTools:           true,
		SupportsReasoning:       true,
		SupportsCodeCompletion:  true,
		SupportsCodeAnalysis:    true,
		Limits: models.ModelLimits{
			MaxTokens:             131072,
			MaxInputLength:        131072,
			MaxOutputLength:       4096,
			MaxConcurrentRequests: 100,
		},
		Metadata: map[string]string{
			"provider": "together",
		},
	}
}

// ValidateConfig validates provider configuration
func (p *Provider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		// CONST-046 round-425: validation error routed through i18n.
		errors = append(errors, i18n.Tr(context.Background(),
			"llmprovider_validate_api_key_required", nil))
	}
	return len(errors) == 0, errors
}

// convertRequest converts LLMRequest to Together AI format
func (p *Provider) convertRequest(req *models.LLMRequest) Request {
	messages := make([]Message, 0, len(req.Messages)+1)

	// Add system prompt
	if req.Prompt != "" {
		messages = append(messages, Message{Role: "system", Content: req.Prompt})
	}

	// Add conversation messages
	for _, msg := range req.Messages {
		messages = append(messages, Message{Role: msg.Role, Content: msg.Content})
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

	// Override model if specified
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

// convertResponse converts Together AI response to LLMResponse
func (p *Provider) convertResponse(req *models.LLMRequest, resp *Response, startTime time.Time) *models.LLMResponse {
	var content string
	var finishReason string
	var toolCalls []models.ToolCall

	if len(resp.Choices) > 0 {
		content = resp.Choices[0].Message.Content
		finishReason = resp.Choices[0].FinishReason

		// Parse tool calls
		if len(resp.Choices[0].Message.ToolCalls) > 0 {
			toolCalls = make([]models.ToolCall, len(resp.Choices[0].Message.ToolCalls))
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

	return &models.LLMResponse{
		ID:           resp.ID,
		RequestID:    req.ID,
		ProviderID:   "together",
		ProviderName: "Together AI",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   resp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		ToolCalls:    toolCalls,
		Metadata: map[string]any{
			"model":             resp.Model,
			"prompt_tokens":     resp.Usage.PromptTokens,
			"completion_tokens": resp.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *Provider) calculateConfidence(content, finishReason string) float64 {
	confidence := 0.85
	switch finishReason {
	case "stop", "end_turn":
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

func (p *Provider) makeAPICall(ctx context.Context, req Request) (*http.Response, error) {
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

		httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

		resp, err := p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = err
			continue
		}

		// Check for retryable status codes
		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("retryable error: status %d", resp.StatusCode)
			continue
		}

		return resp, nil
	}

	return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

func (p *Provider) calculateBackoff(attempt int) time.Duration {
	delay := p.retryConfig.InitialDelay
	for i := 1; i < attempt; i++ {
		delay = time.Duration(float64(delay) * p.retryConfig.Multiplier)
		if delay > p.retryConfig.MaxDelay {
			delay = p.retryConfig.MaxDelay
			break
		}
	}
	jitter := time.Duration(rand.Float64() * float64(delay) * 0.1) // #nosec G404
	return delay + jitter
}

// GetModel returns the current model
func (p *Provider) GetModel() string {
	return p.model
}

// SetModel sets the model
func (p *Provider) SetModel(model string) {
	p.model = model
}

// GetName returns the provider name
func (p *Provider) GetName() string {
	return "together"
}
