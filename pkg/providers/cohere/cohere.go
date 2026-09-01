package cohere

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
// LLMPROVIDER_COHERE_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

const (
	// CohereAPIURL is the base URL for Cohere API
	CohereAPIURL = "https://api.cohere.com/v2/chat"
	// CohereModelsURL is the URL for listing models
	CohereModelsURL = "https://api.cohere.com/v1/models"
	// DefaultModel is the default Cohere model
	DefaultModel = "command-a-03-2025"
)

// Provider implements the LLMProvider interface for Cohere
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

// Request represents a Cohere chat request
type Request struct {
	Model            string           `json:"model"`
	Messages         []Message        `json:"messages"`
	Temperature      float64          `json:"temperature,omitempty"`
	MaxTokens        int              `json:"max_tokens,omitempty"`
	TopP             float64          `json:"p,omitempty"`
	TopK             int              `json:"k,omitempty"`
	Stream           bool             `json:"stream,omitempty"`
	StopSequences    []string         `json:"stop_sequences,omitempty"`
	Tools            []Tool           `json:"tools,omitempty"`
	Documents        []Document       `json:"documents,omitempty"`
	ResponseFormat   *ResponseFormat  `json:"response_format,omitempty"`
	SafetyMode       string           `json:"safety_mode,omitempty"`
	ConnectorIDs     []string         `json:"connector_ids,omitempty"`
	SearchQueryOnly  bool             `json:"search_query_only,omitempty"`
	Preamble         string           `json:"preamble,omitempty"`
	PromptTruncation string           `json:"prompt_truncation,omitempty"`
	CitationOptions  *CitationOptions `json:"citation_options,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// Tool represents a tool definition
type Tool struct {
	Type        string         `json:"type"`
	Function    *FunctionDef   `json:"function,omitempty"`
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameter_definitions,omitempty"`
}

// FunctionDef represents a function definition
type FunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ToolCall represents a tool call in a response
type ToolCall struct {
	ID         string         `json:"id"`
	Type       string         `json:"type"`
	Function   FunctionCall   `json:"function,omitempty"`
	Name       string         `json:"name,omitempty"`
	Parameters map[string]any `json:"parameters,omitempty"`
}

// FunctionCall represents function call details
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Document represents a document for RAG
type Document struct {
	ID    string `json:"id"`
	Text  string `json:"text"`
	Title string `json:"title,omitempty"`
}

// ResponseFormat specifies the response format
type ResponseFormat struct {
	Type       string         `json:"type"` // "text" or "json_object"
	JSONSchema map[string]any `json:"json_schema,omitempty"`
}

// CitationOptions configures citation behavior
type CitationOptions struct {
	Mode string `json:"mode"` // "FAST", "ACCURATE", "OFF"
}

// Response represents a Cohere chat response
type Response struct {
	ID           string        `json:"id"`
	Message      MessageOutput `json:"message"`
	FinishReason string        `json:"finish_reason"`
	Usage        Usage         `json:"usage"`
}

// MessageOutput represents the assistant message in response
type MessageOutput struct {
	Role      string        `json:"role"`
	Content   []ContentPart `json:"content"`
	ToolCalls []ToolCall    `json:"tool_calls,omitempty"`
	ToolPlan  string        `json:"tool_plan,omitempty"`
	Citations []Citation    `json:"citations,omitempty"`
}

// ContentPart represents a part of the message content
type ContentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// Citation represents a citation in the response
type Citation struct {
	Start       int      `json:"start"`
	End         int      `json:"end"`
	Text        string   `json:"text"`
	DocumentIDs []string `json:"document_ids"`
}

// Usage represents token usage
type Usage struct {
	BilledUnits BilledUnits `json:"billed_units"`
	Tokens      TokenUsage  `json:"tokens"`
}

// BilledUnits represents billed token units
type BilledUnits struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// TokenUsage represents detailed token usage
type TokenUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// StreamResponse represents a streaming response chunk
type StreamResponse struct {
	Type         string       `json:"type"`
	ID           string       `json:"id,omitempty"`
	Delta        *StreamDelta `json:"delta,omitempty"`
	FinishReason string       `json:"finish_reason,omitempty"`
	Usage        *Usage       `json:"usage,omitempty"`
}

// StreamDelta represents streaming delta content
type StreamDelta struct {
	Message *StreamMessage `json:"message,omitempty"`
}

// StreamMessage represents streaming message content
type StreamMessage struct {
	Content *StreamContent `json:"content,omitempty"`
}

// StreamContent represents streaming content
type StreamContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
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

// NewProvider creates a new Cohere provider
func NewProvider(apiKey, baseURL, model string) *Provider {
	return NewProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

// NewProviderWithRetry creates a new Cohere provider with custom retry config
func NewProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *Provider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_COHERE_BASE_URL.
		baseURL = settings.BaseURL("cohere", CohereAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_COHERE_MODEL overrides this compiled fallback.
		model = settings.Model("cohere", DefaultModel)
	}

	p := &Provider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("cohere", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "cohere",
		ModelsEndpoint: CohereModelsURL,
		ModelsDevID:    "cohere",
		APIKey:         apiKey,
		ResponseParser: discovery.ParseCohereModelsResponse,
		FallbackModels: []string{
			"command-a-03-2025",
			"command-a-reasoning-08-2025",
			"command-a-vision-07-2025",
			"command-a-translate-08-2025",
			"command-r7b-12-2024",
			"command-r-plus-08-2024",
			"command-r-08-2024",
			"c4ai-aya-expanse-32b",
			"c4ai-aya-vision-32b",
			"c4ai-aya-expanse-8b",
			"c4ai-aya-vision-8b",
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
		return nil, fmt.Errorf("Cohere API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Cohere API error: %d - %s", resp.StatusCode, string(body))
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
		return nil, fmt.Errorf("Cohere streaming API call failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("Cohere API error: %d - failed to read error body: %w", resp.StatusCode, err)
		}
		return nil, fmt.Errorf("Cohere API error: %d - %s", resp.StatusCode, string(body))
	}

	ch := make(chan *models.LLMResponse)
	go func() {
		defer func() { _ = resp.Body.Close() }()
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		var fullContent strings.Builder

		for {
			line, err := reader.ReadBytes('\n')
			// On io.EOF, ReadBytes may still return a final partial line (the last
			// data chunk without a trailing newline). Process it before breaking.
			eof := err == io.EOF
			if err != nil && !eof {
				ch <- &models.LLMResponse{
					ID:           "stream-error-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "cohere",
					ProviderName: "Cohere",
					FinishReason: "error",
					CreatedAt:    time.Now(),
				}
				return
			}

			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				if eof {
					break
				}
				continue
			}

			// Skip "data: " prefix if present
			if bytes.HasPrefix(line, []byte("data: ")) {
				line = bytes.TrimPrefix(line, []byte("data: "))
			}

			var streamResp StreamResponse
			if err := json.Unmarshal(line, &streamResp); err != nil {
				if eof {
					break
				}
				continue
			}

			switch streamResp.Type {
			case "content-delta":
				if streamResp.Delta != nil && streamResp.Delta.Message != nil &&
					streamResp.Delta.Message.Content != nil {
					text := streamResp.Delta.Message.Content.Text
					fullContent.WriteString(text)
					ch <- &models.LLMResponse{
						ID:           streamResp.ID,
						RequestID:    req.ID,
						ProviderID:   "cohere",
						ProviderName: "Cohere",
						Content:      text,
						FinishReason: "",
						CreatedAt:    time.Now(),
					}
				}
			case "message-end":
				ch <- &models.LLMResponse{
					ID:           "stream-final-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "cohere",
					ProviderName: "Cohere",
					Content:      fullContent.String(),
					Confidence:   0.9,
					ResponseTime: time.Since(startTime).Milliseconds(),
					FinishReason: streamResp.FinishReason,
					CreatedAt:    time.Now(),
				}
				return
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
// of a hardcoded production constant. Cohere's chat base ("/v2/chat") and models
// path ("/v1/models") differ in version, so the host root is taken from baseURL.
func (p *Provider) modelsURL() string {
	root := strings.TrimSuffix(p.baseURL, "/v2/chat")
	return root + "/v1/models"
}

func (p *Provider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(ctx, "GET", p.modelsURL(), nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	httpReq.Header.Set("Accept", "application/json")

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
			"chat", "streaming", "tools", "rag", "embeddings",
			"rerank", "classify", "summarize", "json_mode",
			"vision", "reasoning",
		},
		SupportedRequestTypes:   []string{"chat", "completion", "embed", "rerank"},
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          true,
		SupportsTools:           true,
		SupportsReasoning:       true,
		SupportsCodeCompletion:  true,
		SupportsCodeAnalysis:    true,
		Limits: models.ModelLimits{
			MaxTokens:             256000,
			MaxInputLength:        256000,
			MaxOutputLength:       8192,
			MaxConcurrentRequests: 100,
		},
		Metadata: map[string]string{
			"provider":    "cohere",
			"api_version": "v2",
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

// convertRequest converts LLMRequest to Cohere format
func (p *Provider) convertRequest(req *models.LLMRequest) Request {
	messages := make([]Message, 0, len(req.Messages)+1)

	// Add system prompt as preamble if present
	preamble := ""
	if req.Prompt != "" {
		preamble = req.Prompt
	}

	// Convert messages - Cohere v2 uses different role names
	for _, msg := range req.Messages {
		role := msg.Role
		// Map roles to Cohere format
		switch role {
		case "system":
			preamble = msg.Content
			continue
		case "assistant":
			role = "assistant"
		case "user":
			role = "user"
		case "tool":
			role = "tool"
		}
		messages = append(messages, Message{Role: role, Content: msg.Content})
	}

	// Get max tokens with default
	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096
	}

	apiReq := Request{
		Model:         p.model,
		Messages:      messages,
		Temperature:   req.ModelParams.Temperature,
		MaxTokens:     maxTokens,
		TopP:          req.ModelParams.TopP,
		StopSequences: req.ModelParams.StopSequences,
		Preamble:      preamble,
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
				Type: "function",
				Function: &FunctionDef{
					Name:        t.Function.Name,
					Description: t.Function.Description,
					Parameters:  t.Function.Parameters,
				},
			}
		}
	}

	return apiReq
}

// convertResponse converts Cohere response to LLMResponse
func (p *Provider) convertResponse(req *models.LLMRequest, resp *Response, startTime time.Time) *models.LLMResponse {
	var content string
	var toolCalls []models.ToolCall

	// Extract content from response
	for _, part := range resp.Message.Content {
		if part.Type == "text" {
			content += part.Text
		}
	}

	// Parse tool calls
	if len(resp.Message.ToolCalls) > 0 {
		toolCalls = make([]models.ToolCall, len(resp.Message.ToolCalls))
		for i, tc := range resp.Message.ToolCalls {
			args := ""
			if tc.Parameters != nil {
				argsBytes, _ := json.Marshal(tc.Parameters) //nolint:errcheck
				args = string(argsBytes)
			} else if tc.Function.Arguments != "" {
				args = tc.Function.Arguments
			}

			toolCalls[i] = models.ToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: models.ToolCallFunction{
					Name:      tc.Name,
					Arguments: args,
				},
			}
		}
	}

	finishReason := resp.FinishReason
	if len(toolCalls) > 0 && (finishReason == "" || finishReason == "COMPLETE") {
		finishReason = "tool_calls"
	}

	confidence := p.calculateConfidence(content, finishReason)
	totalTokens := resp.Usage.Tokens.InputTokens + resp.Usage.Tokens.OutputTokens

	return &models.LLMResponse{
		ID:           resp.ID,
		RequestID:    req.ID,
		ProviderID:   "cohere",
		ProviderName: "Cohere",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   totalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		ToolCalls:    toolCalls,
		Metadata: map[string]any{
			"input_tokens":  resp.Usage.Tokens.InputTokens,
			"output_tokens": resp.Usage.Tokens.OutputTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *Provider) calculateConfidence(content, finishReason string) float64 {
	confidence := 0.85
	switch finishReason {
	case "COMPLETE", "stop":
		confidence += 0.1
	case "MAX_TOKENS", "length":
		confidence -= 0.1
	case "ERROR":
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
		httpReq.Header.Set("Accept", "application/json")

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
	return "cohere"
}
