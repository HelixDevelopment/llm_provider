package anthropic

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
// LLMPROVIDER_ANTHROPIC_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 300 * time.Second

const (
	// AnthropicAPIURL is the base URL for Anthropic API
	AnthropicAPIURL = "https://api.anthropic.com/v1/messages"
	// DefaultModel is the default Anthropic model
	DefaultModel = "claude-sonnet-4-20250514"
	// APIVersion is the Anthropic API version
	APIVersion = "2023-06-01"
)

// Provider implements the LLMProvider interface for Anthropic
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

// Request represents an Anthropic messages request
type Request struct {
	Model         string       `json:"model"`
	Messages      []Message    `json:"messages"`
	MaxTokens     int          `json:"max_tokens"`
	System        string       `json:"system,omitempty"`
	Temperature   float64      `json:"temperature,omitempty"`
	TopP          float64      `json:"top_p,omitempty"`
	TopK          int          `json:"top_k,omitempty"`
	Stream        bool         `json:"stream,omitempty"`
	StopSequences []string     `json:"stop_sequences,omitempty"`
	Tools         []Tool       `json:"tools,omitempty"`
	ToolChoice    *ToolChoice  `json:"tool_choice,omitempty"`
	Metadata      *RequestMeta `json:"metadata,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content,omitempty"`
}

// ContentBlock represents a content block in a message
type ContentBlock struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	ID        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
	Input     any    `json:"input,omitempty"`
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
}

// Tool represents a tool definition
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

// ToolChoice represents tool choice configuration
type ToolChoice struct {
	Type string `json:"type"` // "auto", "any", or "tool"
	Name string `json:"name,omitempty"`
}

// RequestMeta contains request metadata
type RequestMeta struct {
	UserID string `json:"user_id,omitempty"`
}

// Response represents an Anthropic messages response
type Response struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Content      []ContentBlock `json:"content"`
	Model        string         `json:"model"`
	StopReason   string         `json:"stop_reason"`
	StopSequence string         `json:"stop_sequence,omitempty"`
	Usage        Usage          `json:"usage"`
}

// Usage represents token usage
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// StreamEvent represents a streaming event
type StreamEvent struct {
	Type         string        `json:"type"`
	Index        int           `json:"index,omitempty"`
	ContentBlock *ContentBlock `json:"content_block,omitempty"`
	Delta        *StreamDelta  `json:"delta,omitempty"`
	Message      *Response     `json:"message,omitempty"`
	Usage        *Usage        `json:"usage,omitempty"`
}

// StreamDelta represents streaming delta content
type StreamDelta struct {
	Type        string `json:"type,omitempty"`
	Text        string `json:"text,omitempty"`
	StopReason  string `json:"stop_reason,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
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

// NewProvider creates a new Anthropic provider
func NewProvider(apiKey, baseURL, model string) *Provider {
	return NewProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

// NewProviderWithRetry creates a new Anthropic provider with custom retry config
func NewProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *Provider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_ANTHROPIC_BASE_URL.
		baseURL = settings.BaseURL("anthropic", AnthropicAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_ANTHROPIC_MODEL overrides this compiled fallback.
		model = settings.Model("anthropic", DefaultModel)
	}

	p := &Provider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("anthropic", DefaultHTTPTimeout), // Anthropic can have long responses
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "anthropic",
		ModelsEndpoint: "https://api.anthropic.com/v1/models",
		ModelsDevID:    "anthropic",
		APIKey:         apiKey,
		AuthHeader:     "x-api-key",
		AuthPrefix:     "",
		ExtraHeaders: map[string]string{
			"anthropic-version": "2023-06-01",
		},
		FallbackModels: []string{
			// Claude 4.6 (Latest - February 2026)
			"claude-opus-4-6",
			// Claude 4.5 (November 2025)
			"claude-opus-4-5-20251101",
			"claude-sonnet-4-5-20250929",
			"claude-haiku-4-5-20251001",
			// Claude 4 models (May 2025)
			"claude-sonnet-4-20250514",
			"claude-opus-4-20250514",
			"claude-haiku-4-20250514",
			// Claude 3.5 models
			"claude-3-5-sonnet-20241022",
			"claude-3-5-haiku-20241022",
			// Claude 3 models
			"claude-3-opus-20240229",
			"claude-3-sonnet-20240229",
			"claude-3-haiku-20240307",
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
		return nil, fmt.Errorf("Anthropic API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Anthropic API error: %d - %s", resp.StatusCode, string(body))
	}

	var apiResp Response
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
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
		return nil, fmt.Errorf("Anthropic streaming API call failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf("Anthropic API error: %d - %s", resp.StatusCode, string(body))
	}

	ch := make(chan *models.LLMResponse)
	go func() {
		defer func() { _ = resp.Body.Close() }()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		var fullContent strings.Builder
		var stopReason string

		for {
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				ch <- &models.LLMResponse{
					ID:           "stream-error-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "anthropic",
					ProviderName: "Anthropic",
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

			var event StreamEvent
			if err := json.Unmarshal(line, &event); err != nil {
				continue
			}

			switch event.Type {
			case "content_block_delta":
				if event.Delta != nil && event.Delta.Text != "" {
					fullContent.WriteString(event.Delta.Text)
					ch <- &models.LLMResponse{
						ID:           "stream-chunk-" + req.ID,
						RequestID:    req.ID,
						ProviderID:   "anthropic",
						ProviderName: "Anthropic",
						Content:      event.Delta.Text,
						FinishReason: "",
						CreatedAt:    time.Now(),
					}
				}
			case "message_delta":
				if event.Delta != nil && event.Delta.StopReason != "" {
					stopReason = event.Delta.StopReason
				}
			case "message_stop":
				ch <- &models.LLMResponse{
					ID:           "stream-final-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "anthropic",
					ProviderName: "Anthropic",
					Content:      fullContent.String(),
					Confidence:   0.9,
					ResponseTime: time.Since(startTime).Milliseconds(),
					FinishReason: stopReason,
					CreatedAt:    time.Now(),
				}
				return
			}
		}
	}()

	return ch, nil
}

// HealthCheck verifies provider connectivity
func (p *Provider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Make a minimal request to check API connectivity
	req := &models.LLMRequest{
		ID:       "health-check",
		Messages: []models.Message{{Role: "user", Content: "Hi"}},
		ModelParams: models.ModelParameters{
			MaxTokens: 5,
		},
	}

	_, err := p.Complete(ctx, req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	return nil
}

// GetCapabilities returns provider capabilities
func (p *Provider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: p.discoverer.DiscoverModels(),
		SupportedFeatures: []string{
			"chat", "streaming", "tools", "vision", "extended_thinking",
			"code_completion", "system_prompts", "computer_use",
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
			MaxTokens:             200000,
			MaxInputLength:        200000,
			MaxOutputLength:       8192,
			MaxConcurrentRequests: 50,
		},
		Metadata: map[string]string{
			"provider":    "anthropic",
			"api_version": APIVersion,
		},
	}
}

// ValidateConfig validates provider configuration
func (p *Provider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}
	return len(errors) == 0, errors
}

// convertRequest converts LLMRequest to Anthropic format
func (p *Provider) convertRequest(req *models.LLMRequest) Request {
	messages := make([]Message, 0, len(req.Messages))

	// Convert messages - Anthropic uses content blocks
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			continue // System messages handled separately
		}
		messages = append(messages, Message{
			Role: msg.Role,
			Content: []ContentBlock{
				{Type: "text", Text: msg.Content},
			},
		})
	}

	// Get max tokens with default
	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096
	}

	apiReq := Request{
		Model:         p.model,
		Messages:      messages,
		MaxTokens:     maxTokens,
		System:        req.Prompt,
		Temperature:   req.ModelParams.Temperature,
		TopP:          req.ModelParams.TopP,
		StopSequences: req.ModelParams.StopSequences,
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
				Name:        t.Function.Name,
				Description: t.Function.Description,
				InputSchema: t.Function.Parameters,
			}
		}
		// Set tool choice
		if req.ToolChoice != nil {
			switch v := req.ToolChoice.(type) {
			case string:
				if v == "auto" {
					apiReq.ToolChoice = &ToolChoice{Type: "auto"}
				} else if v == "any" {
					apiReq.ToolChoice = &ToolChoice{Type: "any"}
				}
			}
		}
	}

	return apiReq
}

// convertResponse converts Anthropic response to LLMResponse
func (p *Provider) convertResponse(req *models.LLMRequest, resp *Response, startTime time.Time) *models.LLMResponse {
	var content strings.Builder
	var toolCalls []models.ToolCall

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			content.WriteString(block.Text)
		case "tool_use":
			inputBytes, _ := json.Marshal(block.Input) //nolint:errcheck
			toolCalls = append(toolCalls, models.ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: models.ToolCallFunction{
					Name:      block.Name,
					Arguments: string(inputBytes),
				},
			})
		}
	}

	finishReason := resp.StopReason
	if len(toolCalls) > 0 && finishReason == "end_turn" {
		finishReason = "tool_calls"
	}

	confidence := p.calculateConfidence(content.String(), finishReason)
	totalTokens := resp.Usage.InputTokens + resp.Usage.OutputTokens

	return &models.LLMResponse{
		ID:           resp.ID,
		RequestID:    req.ID,
		ProviderID:   "anthropic",
		ProviderName: "Anthropic",
		Content:      content.String(),
		Confidence:   confidence,
		TokensUsed:   totalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		ToolCalls:    toolCalls,
		Metadata: map[string]any{
			"model":         resp.Model,
			"input_tokens":  resp.Usage.InputTokens,
			"output_tokens": resp.Usage.OutputTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *Provider) calculateConfidence(content, finishReason string) float64 {
	confidence := 0.85
	switch finishReason {
	case "end_turn", "stop":
		confidence += 0.1
	case "max_tokens":
		confidence -= 0.1
	case "stop_sequence":
		confidence += 0.05
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
		httpReq.Header.Set("x-api-key", p.apiKey)
		httpReq.Header.Set("anthropic-version", APIVersion)

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
	return "anthropic"
}
