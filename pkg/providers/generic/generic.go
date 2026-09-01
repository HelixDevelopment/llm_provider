package generic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"digital.vasic.llmprovider/pkg/i18n"
	"digital.vasic.llmprovider/pkg/models"
	"digital.vasic.llmprovider/pkg/settings"
)

const (
	// DefaultTimeout is the default HTTP timeout for generic providers
	DefaultTimeout = 120 * time.Second
	// DefaultMaxTokens is the default max tokens
	DefaultMaxTokens = 4096
	// MaxTokensCap is the maximum token cap for generic providers
	MaxTokensCap = 16384
)

// Provider implements LLMProvider for any OpenAI-compatible chat completions endpoint.
// This enables verification of providers that have OpenAI-compatible APIs but no
// dedicated parent-project provider implementation (e.g., nvidia, sambanova, hyperbolic).
type Provider struct {
	apiKey     string
	baseURL    string
	model      string
	name       string
	httpClient *http.Client
}

// Request represents an OpenAI-compatible chat completion request
type Request struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	TopP        float64   `json:"top_p,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Response represents an OpenAI-compatible chat completion response
type Response struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

// Choice represents a response choice
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// StreamChoice represents a streaming response choice
type StreamChoice struct {
	Index        int     `json:"index"`
	Delta        Message `json:"delta"`
	FinishReason string  `json:"finish_reason"`
}

// StreamResponse represents an OpenAI-compatible streaming response chunk
type StreamResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

// Usage represents token usage
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// NewGenericProvider creates a new generic OpenAI-compatible provider.
// name: provider identifier (e.g., "nvidia", "sambanova")
// apiKey: API key for authentication (Bearer token)
// baseURL: full chat completions URL (e.g., "https://api.nvidia.com/v1/chat/completions")
// model: default model ID to use
func NewGenericProvider(name, apiKey, baseURL, model string) *Provider {
	return &Provider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		name:    name,
		httpClient: &http.Client{
			// The settings key is the CALLER'S provider name, not "generic":
			// this adapter is a shim in front of many backends, so a single
			// LLMPROVIDER_GENERIC_TIMEOUT would tie every backend reached
			// through it to one wait. `name` gives each its own —
			// LLMPROVIDER_NVIDIA_TIMEOUT, LLMPROVIDER_SAMBANOVA_TIMEOUT, …
			// baseURL and model are NOT resolved here on purpose: this
			// constructor has no compiled default for either, so there is
			// nothing to unfreeze — the caller supplies both.
			Timeout: settings.Timeout(name, DefaultTimeout),
		},
	}
}

// Complete sends a completion request to the OpenAI-compatible endpoint
func (p *Provider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()

	apiReq := p.convertRequest(req)

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("%s: failed to marshal request: %w", p.name, err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%s: failed to create request: %w", p.name, err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%s: API request failed: %w", p.name, err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("%s: failed to read response: %w", p.name, err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: API error: %d - %s", p.name, resp.StatusCode, string(respBody))
	}

	var apiResp Response
	if err := json.Unmarshal(respBody, &apiResp); err != nil {
		return nil, fmt.Errorf("%s: failed to parse response: %w", p.name, err)
	}

	return p.convertResponse(req, &apiResp, startTime), nil
}

// CompleteStream sends a streaming completion request
func (p *Provider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	apiReq := p.convertRequest(req)
	apiReq.Stream = true

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("%s: failed to marshal stream request: %w", p.name, err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%s: failed to create stream request: %w", p.name, err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%s: stream request failed: %w", p.name, err)
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf("%s: stream API error: %d - %s", p.name, resp.StatusCode, string(respBody))
	}

	startTime := time.Now()
	ch := make(chan *models.LLMResponse)

	go func() {
		defer func() { _ = resp.Body.Close() }()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		var fullContent strings.Builder

		for {
			line, err := reader.ReadBytes('\n')
			// On io.EOF, ReadBytes may return a final partial line (no trailing
			// newline) that MUST still be processed before exiting; only a
			// non-EOF error is a true stream failure.
			if err != nil && err != io.EOF {
				ch <- &models.LLMResponse{
					ID:           "stream-error-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   p.name,
					ProviderName: p.name,
					FinishReason: "error",
					CreatedAt:    time.Now(),
				}
				return
			}
			eof := err == io.EOF

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
					ProviderID:   p.name,
					ProviderName: p.name,
					Content:      fullContent.String(),
					Confidence:   0.85,
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
						ProviderID:   p.name,
						ProviderName: p.name,
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

// HealthCheck verifies provider connectivity by making a minimal API call
func (p *Provider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Make a minimal completion request to verify the endpoint works
	apiReq := Request{
		Model: p.model,
		Messages: []Message{
			{Role: "user", Content: "hi"},
		},
		MaxTokens: 1,
	}

	body, err := json.Marshal(apiReq)
	if err != nil {
		return fmt.Errorf("%s: failed to create health check request: %w", p.name, err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("%s: failed to create request: %w", p.name, err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("%s: health check failed: %w", p.name, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: health check failed: status %d", p.name, resp.StatusCode)
	}
	return nil
}

// GetCapabilities returns provider capabilities
func (p *Provider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels:         []string{p.model},
		SupportedFeatures:       []string{"chat", "streaming"},
		SupportedRequestTypes:   []string{"chat"},
		SupportsStreaming:       true,
		SupportsFunctionCalling: false,
		SupportsVision:          false,
		SupportsTools:           false,
		Limits: models.ModelLimits{
			MaxTokens:             MaxTokensCap,
			MaxInputLength:        MaxTokensCap,
			MaxOutputLength:       MaxTokensCap,
			MaxConcurrentRequests: 10,
		},
		Metadata: map[string]string{"provider": p.name, "type": "generic_openai_compatible"},
	}
}

// ValidateConfig validates provider configuration
func (p *Provider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string
	if p.apiKey == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
	}
	if p.baseURL == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_base_url_required", nil))
	}
	if p.model == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_model_required", nil))
	}
	return len(errors) == 0, errors
}

// convertRequest converts LLMRequest to generic OpenAI format
func (p *Provider) convertRequest(req *models.LLMRequest) Request {
	messages := make([]Message, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		messages = append(messages, Message{Role: "system", Content: req.Prompt})
	}

	for _, msg := range req.Messages {
		messages = append(messages, Message{Role: msg.Role, Content: msg.Content})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	} else if maxTokens > MaxTokensCap {
		maxTokens = MaxTokensCap
	}

	model := p.model
	if req.ModelParams.Model != "" {
		model = req.ModelParams.Model
	}

	return Request{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: req.ModelParams.Temperature,
		TopP:        req.ModelParams.TopP,
		Stop:        req.ModelParams.StopSequences,
	}
}

// convertResponse converts OpenAI-compatible response to LLMResponse
func (p *Provider) convertResponse(req *models.LLMRequest, resp *Response, startTime time.Time) *models.LLMResponse {
	var content string
	var finishReason string

	if len(resp.Choices) > 0 {
		content = resp.Choices[0].Message.Content
		finishReason = resp.Choices[0].FinishReason
	}

	confidence := 0.85
	if finishReason == "stop" {
		confidence = 0.9
	} else if finishReason == "length" {
		confidence = 0.75
	}

	var tokensUsed int
	if resp.Usage != nil {
		tokensUsed = resp.Usage.TotalTokens
	}

	return &models.LLMResponse{
		ID:           resp.ID,
		RequestID:    req.ID,
		ProviderID:   p.name,
		ProviderName: p.name,
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   tokensUsed,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		CreatedAt:    time.Now(),
	}
}
