package perplexity

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
// LLMPROVIDER_PERPLEXITY_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

const (
	// PerplexityAPIURL is the base URL for Perplexity API
	PerplexityAPIURL = "https://api.perplexity.ai/chat/completions"
	// DefaultModel is the default Perplexity model
	DefaultModel = "llama-3.1-sonar-large-128k-online"
)

// Provider implements the LLMProvider interface for Perplexity
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

// Request represents a Perplexity chat completion request (OpenAI compatible)
type Request struct {
	Model                  string    `json:"model"`
	Messages               []Message `json:"messages"`
	Temperature            float64   `json:"temperature,omitempty"`
	MaxTokens              int       `json:"max_tokens,omitempty"`
	TopP                   float64   `json:"top_p,omitempty"`
	TopK                   int       `json:"top_k,omitempty"`
	Stream                 bool      `json:"stream,omitempty"`
	FrequencyPenalty       float64   `json:"frequency_penalty,omitempty"`
	PresencePenalty        float64   `json:"presence_penalty,omitempty"`
	SearchDomainFilter     []string  `json:"search_domain_filter,omitempty"`
	ReturnImages           bool      `json:"return_images,omitempty"`
	ReturnRelatedQuestions bool      `json:"return_related_questions,omitempty"`
	SearchRecencyFilter    string    `json:"search_recency_filter,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Response represents a Perplexity chat completion response
type Response struct {
	ID        string   `json:"id"`
	Object    string   `json:"object"`
	Created   int64    `json:"created"`
	Model     string   `json:"model"`
	Choices   []Choice `json:"choices"`
	Usage     Usage    `json:"usage"`
	Citations []string `json:"citations,omitempty"`
}

// Choice represents a completion choice
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
	Delta        Message `json:"delta,omitempty"`
}

// Usage represents token usage
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// StreamResponse represents a streaming response chunk
type StreamResponse struct {
	ID        string         `json:"id"`
	Object    string         `json:"object"`
	Created   int64          `json:"created"`
	Model     string         `json:"model"`
	Choices   []StreamChoice `json:"choices"`
	Citations []string       `json:"citations,omitempty"`
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

// NewProvider creates a new Perplexity provider
func NewProvider(apiKey, baseURL, model string) *Provider {
	return NewProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

// NewProviderWithRetry creates a new Perplexity provider with custom retry config
func NewProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *Provider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_PERPLEXITY_BASE_URL.
		baseURL = settings.BaseURL("perplexity", PerplexityAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_PERPLEXITY_MODEL overrides this compiled fallback.
		model = settings.Model("perplexity", DefaultModel)
	}

	p := &Provider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("perplexity", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName: "perplexity",
		ModelsDevID:  "perplexity",
		APIKey:       apiKey,
		FallbackModels: []string{
			// Sonar models (online search)
			"llama-3.1-sonar-small-128k-online",
			"llama-3.1-sonar-large-128k-online",
			"llama-3.1-sonar-huge-128k-online",
			// Sonar models (chat)
			"llama-3.1-sonar-small-128k-chat",
			"llama-3.1-sonar-large-128k-chat",
			// Open models
			"llama-3.1-8b-instruct",
			"llama-3.1-70b-instruct",
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
		return nil, fmt.Errorf("Perplexity API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Perplexity API error: %d - %s", resp.StatusCode, string(body))
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
		return nil, fmt.Errorf("Perplexity streaming API call failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		bodyStr := string(body)
		if err != nil {
			bodyStr = fmt.Sprintf("failed to read response body: %v", err)
		}
		return nil, fmt.Errorf("Perplexity API error: %d - %s", resp.StatusCode, bodyStr)
	}

	ch := make(chan *models.LLMResponse)
	go func() {
		defer func() { _ = resp.Body.Close() }()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		var fullContent strings.Builder
		var citations []string

		for {
			line, err := reader.ReadBytes('\n')
			// On io.EOF, ReadBytes may return a final partial line (no trailing
			// newline) that MUST still be processed before exiting; only a
			// non-EOF error is a true stream failure.
			if err != nil && err != io.EOF {
				ch <- &models.LLMResponse{
					ID:           "stream-error-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "perplexity",
					ProviderName: "Perplexity",
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
				metadata := map[string]any{}
				if len(citations) > 0 {
					metadata["citations"] = citations
				}
				ch <- &models.LLMResponse{
					ID:           "stream-final-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "perplexity",
					ProviderName: "Perplexity",
					Content:      fullContent.String(),
					Confidence:   0.9,
					ResponseTime: time.Since(startTime).Milliseconds(),
					FinishReason: "stop",
					Metadata:     metadata,
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

			if len(streamResp.Citations) > 0 {
				citations = streamResp.Citations
			}

			if len(streamResp.Choices) > 0 {
				delta := streamResp.Choices[0].Delta
				if delta.Content != "" {
					fullContent.WriteString(delta.Content)
					ch <- &models.LLMResponse{
						ID:           streamResp.ID,
						RequestID:    req.ID,
						ProviderID:   "perplexity",
						ProviderName: "Perplexity",
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
func (p *Provider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Perplexity doesn't have a models endpoint, so we do a minimal completion
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
			"chat", "streaming", "online_search", "citations",
			"search_domain_filter", "search_recency_filter",
		},
		SupportedRequestTypes:   []string{"chat", "completion"},
		SupportsStreaming:       true,
		SupportsFunctionCalling: false,
		SupportsVision:          false,
		SupportsTools:           false,
		SupportsReasoning:       true,
		SupportsCodeCompletion:  true,
		SupportsCodeAnalysis:    true,
		Limits: models.ModelLimits{
			MaxTokens:             128000,
			MaxInputLength:        127000,
			MaxOutputLength:       4096,
			MaxConcurrentRequests: 50,
		},
		Metadata: map[string]string{
			"provider":       "perplexity",
			"specialization": "search",
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

// convertRequest converts LLMRequest to Perplexity format
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
	}

	// Override model if specified
	if req.ModelParams.Model != "" {
		apiReq.Model = req.ModelParams.Model
	}

	// Check for provider-specific options
	if req.ModelParams.ProviderSpecific != nil {
		if domains, ok := req.ModelParams.ProviderSpecific["search_domain_filter"].([]string); ok {
			apiReq.SearchDomainFilter = domains
		}
		if recency, ok := req.ModelParams.ProviderSpecific["search_recency_filter"].(string); ok {
			apiReq.SearchRecencyFilter = recency
		}
		if returnImages, ok := req.ModelParams.ProviderSpecific["return_images"].(bool); ok {
			apiReq.ReturnImages = returnImages
		}
	}

	return apiReq
}

// convertResponse converts Perplexity response to LLMResponse
func (p *Provider) convertResponse(req *models.LLMRequest, resp *Response, startTime time.Time) *models.LLMResponse {
	var content string
	var finishReason string

	if len(resp.Choices) > 0 {
		content = resp.Choices[0].Message.Content
		finishReason = resp.Choices[0].FinishReason
	}

	confidence := p.calculateConfidence(content, finishReason)

	metadata := map[string]any{
		"model":             resp.Model,
		"prompt_tokens":     resp.Usage.PromptTokens,
		"completion_tokens": resp.Usage.CompletionTokens,
	}

	// Add citations if present
	if len(resp.Citations) > 0 {
		metadata["citations"] = resp.Citations
	}

	return &models.LLMResponse{
		ID:           resp.ID,
		RequestID:    req.ID,
		ProviderID:   "perplexity",
		ProviderName: "Perplexity",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   resp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata:     metadata,
		CreatedAt:    time.Now(),
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
	return "perplexity"
}
