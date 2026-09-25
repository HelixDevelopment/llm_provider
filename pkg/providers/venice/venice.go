package venice

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
// LLMPROVIDER_VENICE_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

const (
	// VeniceAPIURL is the base URL for Venice AI chat completions
	VeniceAPIURL = "https://api.venice.ai/api/v1/chat/completions"
	// VeniceModelsURL is the URL for listing models
	VeniceModelsURL = "https://api.venice.ai/api/v1/models"
	// VeniceEmbeddingsURL is the URL for embeddings generation
	VeniceEmbeddingsURL = "https://api.venice.ai/api/v1/embeddings"
	// VeniceImageURL is the URL for image generation
	VeniceImageURL = "https://api.venice.ai/api/v1/image/generate"
	// VeniceAudioURL is the URL for text-to-speech
	VeniceAudioURL = "https://api.venice.ai/api/v1/audio/speech"
	// VeniceTranscribeURL is the URL for speech-to-text transcriptions
	VeniceTranscribeURL = "https://api.venice.ai/api/v1/audio/transcriptions"
	// VeniceDefault is the default Venice model
	VeniceDefault = "llama-3.3-70b"
)

// Provider implements the LLMProvider interface for Venice AI
type Provider struct {
	apiKey      string
	baseURL     string
	modelsURL   string
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

// Request represents a Venice AI chat completion request (OpenAI compatible)
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
	Reasoning        string          `json:"reasoning,omitempty"`
	VeniceParameters *VeniceParams   `json:"venice_parameters,omitempty"`
}

// VeniceParams contains Venice-specific request parameters
type VeniceParams struct {
	EnableWebSearch       string `json:"enable_web_search,omitempty"`
	EnableWebCitations    bool   `json:"enable_web_citations,omitempty"`
	StripThinkingResponse bool   `json:"strip_thinking_response,omitempty"`
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

// Response represents a Venice AI chat completion response
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

// DefaultRetryConfig returns sensible defaults for Venice AI
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

// NewProvider creates a new Venice AI provider
func NewProvider(apiKey, baseURL, model string) *Provider {
	return NewProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

// NewProviderWithRetry creates a new Venice AI provider with custom retry config
func NewProviderWithRetry(
	apiKey, baseURL, model string,
	retryConfig RetryConfig,
) *Provider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_VENICE_BASE_URL.
		baseURL = settings.BaseURL("venice", VeniceAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_VENICE_MODEL overrides this compiled fallback.
		model = settings.Model("venice", VeniceDefault)
	}

	modelsURL := VeniceModelsURL
	// Derive models URL from custom base URL if provided
	if baseURL != VeniceAPIURL {
		modelsURL = strings.TrimSuffix(baseURL, "/chat/completions") +
			"/models"
	}

	p := &Provider{
		apiKey:    apiKey,
		baseURL:   baseURL,
		modelsURL: modelsURL,
		model:     model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("venice", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName: "venice",
		// Honour the derived modelsURL (computed from the caller-supplied
		// baseURL) so a custom baseURL — e.g. an httptest server in tests
		// or a corporate proxy in production — actually reaches the
		// discoverer. Previously this field was hard-wired to the
		// production VeniceModelsURL, making the modelsURL derivation
		// above dead-code from the discoverer's perspective.
		ModelsEndpoint: modelsURL,
		ModelsDevID:    "venice",
		APIKey:         apiKey,
		FallbackModels: []string{
			"llama-3.3-70b",
			"zai-org-glm-4.7",
			"venice-uncensored",
			"qwen3-vl-235b-a22b",
			"qwen-2.5-vl",
			"deepseek-r1-671b",
			"llama-3.1-405b",
		},
	})

	return p
}

// Complete sends a completion request to Venice AI
func (p *Provider) Complete(
	ctx context.Context,
	req *models.LLMRequest,
) (*models.LLMResponse, error) {
	startTime := time.Now()
	apiReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, apiReq)
	if err != nil {
		return nil, fmt.Errorf("Venice API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"Venice API error: %d - %s",
			resp.StatusCode, string(body),
		)
	}

	var apiResp Response
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return p.convertResponse(req, &apiResp, startTime), nil
}

// CompleteStream sends a streaming completion request to Venice AI
func (p *Provider) CompleteStream(
	ctx context.Context,
	req *models.LLMRequest,
) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()
	apiReq := p.convertRequest(req)
	apiReq.Stream = true

	resp, err := p.makeAPICall(ctx, apiReq)
	if err != nil {
		return nil, fmt.Errorf(
			"Venice streaming API call failed: %w", err,
		)
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		return nil, fmt.Errorf(
			"Venice API error: %d - %s",
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
					ProviderID:   "venice",
					ProviderName: "Venice",
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
					ProviderID:   "venice",
					ProviderName: "Venice",
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
						ProviderID:   "venice",
						ProviderName: "Venice",
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
func (p *Provider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(
		context.Background(), 10*time.Second,
	)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(
		ctx, "GET", p.modelsURL, nil,
	)
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
			"chat", "streaming", "tools", "vision", "reasoning",
			"web_search", "code_completion", "uncensored",
			"embeddings", "image_generation",
			"text_to_speech", "speech_to_text",
		},
		SupportedRequestTypes:   []string{"chat", "completion"},
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          true,
		SupportsTools:           true,
		SupportsSearch:          true,
		SupportsReasoning:       true,
		SupportsCodeCompletion:  true,
		SupportsCodeAnalysis:    true,
		Limits: models.ModelLimits{
			MaxTokens:             131072,
			MaxInputLength:        131072,
			MaxOutputLength:       32768,
			MaxConcurrentRequests: 50,
		},
		Metadata: map[string]string{
			"provider":            "venice",
			"supported_endpoints": "chat,embeddings,image,audio,models",
			"web_search":          "true",
			"e2ee":                "true",
			"uncensored_models":   "true",
		},
	}
}

// GetSupportedEndpoints returns all Venice AI API endpoints
func (p *Provider) GetSupportedEndpoints() []string {
	return []string{
		"chat/completions",
		"embeddings",
		"image/generate",
		"image/edit",
		"image/upscale",
		"audio/speech",
		"audio/transcriptions",
		"models",
	}
}

// ValidateConfig validates provider configuration
func (p *Provider) ValidateConfig(
	config map[string]interface{},
) (bool, []string) {
	var errors []string

	if p.apiKey == "" {
		// CONST-046 round-441: validation error routed through i18n.
		errors = append(errors, i18n.Tr(context.Background(),
			"llmprovider_validate_api_key_required_venice", nil))
	}

	return len(errors) == 0, errors
}

// GetName returns the provider name
func (p *Provider) GetName() string {
	return "venice"
}

// GetProviderType returns the provider type identifier
func (p *Provider) GetProviderType() string {
	return "venice"
}

// GetModel returns the current model
func (p *Provider) GetModel() string {
	return p.model
}

// SetModel sets the model
func (p *Provider) SetModel(model string) {
	p.model = model
}

// convertRequest converts LLMRequest to Venice API format
func (p *Provider) convertRequest(req *models.LLMRequest) Request {
	messages := make([]Message, 0, len(req.Messages)+1)

	// Add system prompt
	if req.Prompt != "" {
		messages = append(messages, Message{
			Role: "system", Content: req.Prompt,
		})
	}

	// Add conversation messages
	for _, msg := range req.Messages {
		messages = append(messages, Message{
			Role: msg.Role, Content: msg.Content,
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

	// Override model if specified
	if req.ModelParams.Model != "" {
		apiReq.Model = req.ModelParams.Model
	}

	// Add Venice-specific parameters from ProviderSpecific
	if req.ModelParams.ProviderSpecific != nil {
		vp := &VeniceParams{}
		hasVeniceParams := false

		if v, ok := req.ModelParams.ProviderSpecific["enable_web_search"]; ok {
			if s, ok := v.(string); ok {
				vp.EnableWebSearch = s
				hasVeniceParams = true
			}
		}
		if v, ok := req.ModelParams.ProviderSpecific["enable_web_citations"]; ok {
			if b, ok := v.(bool); ok {
				vp.EnableWebCitations = b
				hasVeniceParams = true
			}
		}
		if v, ok := req.ModelParams.ProviderSpecific["strip_thinking_response"]; ok {
			if b, ok := v.(bool); ok {
				vp.StripThinkingResponse = b
				hasVeniceParams = true
			}
		}
		if hasVeniceParams {
			apiReq.VeniceParameters = vp
		}

		if v, ok := req.ModelParams.ProviderSpecific["reasoning"]; ok {
			if s, ok := v.(string); ok {
				apiReq.Reasoning = s
			}
		}
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

// convertResponse converts Venice API response to LLMResponse
func (p *Provider) convertResponse(
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
		ProviderID:   "venice",
		ProviderName: "Venice",
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

func (p *Provider) calculateConfidence(
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

func (p *Provider) makeAPICall(
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
			return nil, fmt.Errorf(
				"failed to create request: %w", err,
			)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

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

func (p *Provider) calculateBackoff(attempt int) time.Duration {
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
	// #nosec G404
	jitter := time.Duration(rand.Float64() * float64(delay) * 0.1)
	return delay + jitter
}
