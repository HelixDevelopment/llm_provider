package huggingface

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
// LLMPROVIDER_HUGGINGFACE_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 120 * time.Second

const (
	// HuggingFaceInferenceURL is the base URL for HuggingFace Inference API (legacy)
	HuggingFaceInferenceURL = "https://router.huggingface.co/hf-inference/models/"
	// HuggingFaceProURL is the base URL for HuggingFace Inference Router (chat completions)
	// Updated from api-inference.huggingface.co which returns 410 Gone as of Nov 2025
	HuggingFaceProURL = "https://router.huggingface.co/v1/chat/completions"
	// DefaultModel is the default HuggingFace model
	DefaultModel = "meta-llama/Llama-3.3-70B-Instruct"
)

// Provider implements the LLMProvider interface for HuggingFace
type Provider struct {
	apiKey      string
	baseURL     string
	model       string
	httpClient  *http.Client
	retryConfig RetryConfig
	usePro      bool
	discoverer  *discovery.Discoverer
}

// RetryConfig defines retry behavior for API calls
type RetryConfig struct {
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

// InferenceRequest represents a HuggingFace inference request
type InferenceRequest struct {
	Inputs     string            `json:"inputs"`
	Parameters InferenceParams   `json:"parameters,omitempty"`
	Options    *InferenceOptions `json:"options,omitempty"`
}

// InferenceParams represents inference parameters
type InferenceParams struct {
	MaxNewTokens      int      `json:"max_new_tokens,omitempty"`
	Temperature       float64  `json:"temperature,omitempty"`
	TopP              float64  `json:"top_p,omitempty"`
	TopK              int      `json:"top_k,omitempty"`
	RepetitionPenalty float64  `json:"repetition_penalty,omitempty"`
	DoSample          bool     `json:"do_sample,omitempty"`
	StopSequences     []string `json:"stop_sequences,omitempty"`
	ReturnFullText    bool     `json:"return_full_text,omitempty"`
}

// InferenceOptions represents inference options
type InferenceOptions struct {
	UseCache     bool `json:"use_cache,omitempty"`
	WaitForModel bool `json:"wait_for_model,omitempty"`
}

// InferenceResponse represents a HuggingFace inference response
type InferenceResponse struct {
	GeneratedText string `json:"generated_text"`
}

// ChatRequest represents a HuggingFace chat completions request (OpenAI compatible)
type ChatRequest struct {
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

// ChatResponse represents a HuggingFace chat completions response
type ChatResponse struct {
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
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
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

// NewProvider creates a new HuggingFace provider
func NewProvider(apiKey, baseURL, model string) *Provider {
	return NewProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

// NewProviderWithRetry creates a new HuggingFace provider with custom retry config
func NewProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *Provider {
	usePro := false
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_HUGGINGFACE_BASE_URL.
		baseURL = settings.BaseURL("huggingface", HuggingFaceProURL)
		usePro = true
	} else if strings.Contains(baseURL, "chat/completions") {
		usePro = true
	}
	if model == "" {
		// LLMPROVIDER_HUGGINGFACE_MODEL overrides this compiled fallback.
		model = settings.Model("huggingface", DefaultModel)
	}

	p := &Provider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("huggingface", DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
		usePro:      usePro,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName: "huggingface",
		ModelsDevID:  "huggingface",
		APIKey:       apiKey,
		FallbackModels: []string{
			// Meta Llama models (current)
			"meta-llama/Llama-3.3-70B-Instruct",
			"meta-llama/Llama-3.2-3B-Instruct",
			"meta-llama/Llama-3.1-8B-Instruct",
			"meta-llama/Llama-3.1-70B-Instruct",
			// Mistral models
			"mistralai/Mistral-7B-Instruct-v0.3",
			"mistralai/Mixtral-8x7B-Instruct-v0.1",
			// Google models
			"google/gemma-2-9b-it",
			"google/gemma-2-27b-it",
			// Microsoft models
			"microsoft/Phi-3-mini-4k-instruct",
			"microsoft/Phi-3-medium-4k-instruct",
			// Other models
			"Qwen/Qwen2.5-72B-Instruct",
			"bigcode/starcoder2-15b",
		},
	})

	return p
}

// Complete sends a completion request
func (p *Provider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()

	if p.usePro {
		return p.completePro(ctx, req, startTime)
	}
	return p.completeInference(ctx, req, startTime)
}

func (p *Provider) completePro(ctx context.Context, req *models.LLMRequest, startTime time.Time) (*models.LLMResponse, error) {
	apiReq := p.convertChatRequest(req)

	resp, err := p.makeAPICall(ctx, p.baseURL, apiReq)
	if err != nil {
		return nil, fmt.Errorf("HuggingFace API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HuggingFace API error: %d - %s", resp.StatusCode, string(body))
	}

	var apiResp ChatResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return p.convertChatResponse(req, &apiResp, startTime), nil
}

func (p *Provider) completeInference(ctx context.Context, req *models.LLMRequest, startTime time.Time) (*models.LLMResponse, error) {
	apiReq := p.convertInferenceRequest(req)
	url := p.baseURL + p.model
	if p.baseURL == HuggingFaceProURL {
		url = HuggingFaceInferenceURL + p.model
	}

	resp, err := p.makeAPICall(ctx, url, apiReq)
	if err != nil {
		return nil, fmt.Errorf("HuggingFace API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HuggingFace API error: %d - %s", resp.StatusCode, string(body))
	}

	// Response can be array or single object
	var responses []InferenceResponse
	if err := json.Unmarshal(body, &responses); err != nil {
		// Try single object
		var single InferenceResponse
		if err := json.Unmarshal(body, &single); err != nil {
			return nil, fmt.Errorf("failed to parse response: %w", err)
		}
		responses = []InferenceResponse{single}
	}

	return p.convertInferenceResponse(req, responses, startTime), nil
}

// CompleteStream sends a streaming completion request
func (p *Provider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	if !p.usePro {
		// Inference API doesn't support streaming, fall back to polling
		ch := make(chan *models.LLMResponse, 1)
		go func() {
			defer close(ch)
			resp, err := p.Complete(ctx, req)
			if err != nil {
				ch <- &models.LLMResponse{
					ID:           "stream-error-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "huggingface",
					ProviderName: "HuggingFace",
					FinishReason: "error",
					CreatedAt:    time.Now(),
				}
				return
			}
			ch <- resp
		}()
		return ch, nil
	}

	startTime := time.Now()
	apiReq := p.convertChatRequest(req)
	apiReq.Stream = true

	resp, err := p.makeAPICall(ctx, p.baseURL, apiReq)
	if err != nil {
		return nil, fmt.Errorf("HuggingFace streaming API call failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("HuggingFace API error: %d - failed to read response body: %w", resp.StatusCode, err)
		}
		return nil, fmt.Errorf("HuggingFace API error: %d - %s", resp.StatusCode, string(body))
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
					ProviderID:   "huggingface",
					ProviderName: "HuggingFace",
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
					ProviderID:   "huggingface",
					ProviderName: "HuggingFace",
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
						ProviderID:   "huggingface",
						ProviderName: "HuggingFace",
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

// HealthCheck verifies provider connectivity
func (p *Provider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	url := "https://huggingface.co/api/models/" + p.model
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
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
			"chat", "streaming", "text_generation", "embeddings",
			"code_completion", "classification", "translation",
		},
		SupportedRequestTypes:   []string{"chat", "completion", "embed"},
		SupportsStreaming:       true,
		SupportsFunctionCalling: false,
		SupportsVision:          true,
		SupportsTools:           false,
		SupportsReasoning:       true,
		SupportsCodeCompletion:  true,
		SupportsCodeAnalysis:    true,
		Limits: models.ModelLimits{
			MaxTokens:             8192,
			MaxInputLength:        8192,
			MaxOutputLength:       4096,
			MaxConcurrentRequests: 50,
		},
		Metadata: map[string]string{
			"provider":       "huggingface",
			"specialization": "open_models",
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

func (p *Provider) convertChatRequest(req *models.LLMRequest) ChatRequest {
	messages := make([]Message, 0, len(req.Messages)+1)

	// Add system prompt
	if req.Prompt != "" {
		messages = append(messages, Message{Role: "system", Content: req.Prompt})
	}

	// Add conversation messages
	for _, msg := range req.Messages {
		messages = append(messages, Message{Role: msg.Role, Content: msg.Content})
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1024
	}

	model := p.model
	if req.ModelParams.Model != "" {
		model = req.ModelParams.Model
	}

	return ChatRequest{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: req.ModelParams.Temperature,
		TopP:        req.ModelParams.TopP,
		Stop:        req.ModelParams.StopSequences,
	}
}

func (p *Provider) convertInferenceRequest(req *models.LLMRequest) InferenceRequest {
	// Build prompt
	var prompt strings.Builder
	if req.Prompt != "" {
		prompt.WriteString("System: ")
		prompt.WriteString(req.Prompt)
		prompt.WriteString("\n\n")
	}
	for _, msg := range req.Messages {
		prompt.WriteString(msg.Role)
		prompt.WriteString(": ")
		prompt.WriteString(msg.Content)
		prompt.WriteString("\n\n")
	}
	prompt.WriteString("Assistant: ")

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	return InferenceRequest{
		Inputs: prompt.String(),
		Parameters: InferenceParams{
			MaxNewTokens:   maxTokens,
			Temperature:    req.ModelParams.Temperature,
			TopP:           req.ModelParams.TopP,
			DoSample:       req.ModelParams.Temperature > 0,
			ReturnFullText: false,
			StopSequences:  req.ModelParams.StopSequences,
		},
		Options: &InferenceOptions{
			WaitForModel: true,
		},
	}
}

func (p *Provider) convertChatResponse(req *models.LLMRequest, resp *ChatResponse, startTime time.Time) *models.LLMResponse {
	var content string
	var finishReason string

	if len(resp.Choices) > 0 {
		content = resp.Choices[0].Message.Content
		finishReason = resp.Choices[0].FinishReason
	}

	confidence := p.calculateConfidence(content, finishReason)

	return &models.LLMResponse{
		ID:           resp.ID,
		RequestID:    req.ID,
		ProviderID:   "huggingface",
		ProviderName: "HuggingFace",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   resp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             resp.Model,
			"prompt_tokens":     resp.Usage.PromptTokens,
			"completion_tokens": resp.Usage.CompletionTokens,
		},
		CreatedAt: time.Now(),
	}
}

func (p *Provider) convertInferenceResponse(req *models.LLMRequest, responses []InferenceResponse, startTime time.Time) *models.LLMResponse {
	var content strings.Builder
	for _, resp := range responses {
		content.WriteString(resp.GeneratedText)
	}

	confidence := p.calculateConfidence(content.String(), "stop")

	return &models.LLMResponse{
		ID:           "hf-" + req.ID,
		RequestID:    req.ID,
		ProviderID:   "huggingface",
		ProviderName: "HuggingFace",
		Content:      content.String(),
		Confidence:   confidence,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: "stop",
		Metadata: map[string]any{
			"model": p.model,
		},
		CreatedAt: time.Now(),
	}
}

func (p *Provider) calculateConfidence(content, finishReason string) float64 {
	confidence := 0.85
	switch finishReason {
	case "stop", "end_turn", "eos_token":
		confidence += 0.1
	case "length":
		confidence -= 0.1
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

func (p *Provider) makeAPICall(ctx context.Context, url string, req interface{}) (*http.Response, error) {
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

		httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
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
		if resp.StatusCode == 429 || resp.StatusCode >= 500 || resp.StatusCode == 503 {
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
	return "huggingface"
}
