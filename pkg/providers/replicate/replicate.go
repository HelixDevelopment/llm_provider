package replicate

import (
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
// LLMPROVIDER_REPLICATE_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 300 * time.Second

const (
	// ReplicateAPIURL is the base URL for Replicate API
	ReplicateAPIURL = "https://api.replicate.com/v1/predictions"
	// ReplicateModelsURL is the URL for models
	ReplicateModelsURL = "https://api.replicate.com/v1/models"
	// DefaultModel is the default Replicate model
	DefaultModel = "meta/llama-2-70b-chat"
)

// Provider implements the LLMProvider interface for Replicate
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

// PredictionRequest represents a Replicate prediction request
type PredictionRequest struct {
	Version string          `json:"version,omitempty"`
	Model   string          `json:"model,omitempty"`
	Input   PredictionInput `json:"input"`
	Stream  bool            `json:"stream,omitempty"`
	Webhook string          `json:"webhook,omitempty"`
}

// PredictionInput represents input for a prediction
type PredictionInput struct {
	Prompt            string  `json:"prompt"`
	SystemPrompt      string  `json:"system_prompt,omitempty"`
	MaxNewTokens      int     `json:"max_new_tokens,omitempty"`
	MaxTokens         int     `json:"max_tokens,omitempty"`
	Temperature       float64 `json:"temperature,omitempty"`
	TopP              float64 `json:"top_p,omitempty"`
	TopK              int     `json:"top_k,omitempty"`
	RepetitionPenalty float64 `json:"repetition_penalty,omitempty"`
	StopSequences     string  `json:"stop_sequences,omitempty"`
}

// PredictionResponse represents a Replicate prediction response
type PredictionResponse struct {
	ID          string          `json:"id"`
	Model       string          `json:"model"`
	Version     string          `json:"version"`
	Status      string          `json:"status"`
	Input       PredictionInput `json:"input"`
	Output      any             `json:"output"`
	Error       string          `json:"error,omitempty"`
	Logs        string          `json:"logs,omitempty"`
	Metrics     *Metrics        `json:"metrics,omitempty"`
	URLs        *URLs           `json:"urls,omitempty"`
	CreatedAt   string          `json:"created_at"`
	StartedAt   string          `json:"started_at,omitempty"`
	CompletedAt string          `json:"completed_at,omitempty"`
}

// Metrics represents prediction metrics
type Metrics struct {
	PredictTime float64 `json:"predict_time"`
	TotalTime   float64 `json:"total_time"`
}

// URLs represents URLs for async operations
type URLs struct {
	Get    string `json:"get"`
	Cancel string `json:"cancel"`
	Stream string `json:"stream,omitempty"`
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

// NewProvider creates a new Replicate provider
func NewProvider(apiKey, baseURL, model string) *Provider {
	return NewProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

// NewProviderWithRetry creates a new Replicate provider with custom retry config
func NewProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *Provider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// keeps making for the operator: LLMPROVIDER_REPLICATE_BASE_URL.
		baseURL = settings.BaseURL("replicate", ReplicateAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_REPLICATE_MODEL overrides this compiled fallback.
		model = settings.Model("replicate", DefaultModel)
	}

	p := &Provider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: settings.Timeout("replicate", DefaultHTTPTimeout), // Replicate can have cold starts
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "replicate",
		ModelsEndpoint: ReplicateModelsURL,
		ModelsDevID:    "replicate",
		APIKey:         apiKey,
		AuthHeader:     "Authorization",
		AuthPrefix:     "Bearer ",
		ResponseParser: discovery.ParseReplicateModelsResponse,
		FallbackModels: []string{
			// Meta Llama models
			"meta/llama-2-70b-chat",
			"meta/llama-2-13b-chat",
			"meta/llama-2-7b-chat",
			"meta/meta-llama-3-70b-instruct",
			"meta/meta-llama-3-8b-instruct",
			"meta/meta-llama-3.1-405b-instruct",
			// Mistral models
			"mistralai/mistral-7b-instruct-v0.2",
			"mistralai/mixtral-8x7b-instruct-v0.1",
			// Other models
			"stability-ai/stable-diffusion",
			"openai/whisper",
			"replicate/all-mpnet-base-v2",
		},
	})

	return p
}

// Complete sends a completion request
func (p *Provider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	apiReq := p.convertRequest(req)

	// Create prediction
	resp, err := p.createPrediction(ctx, apiReq)
	if err != nil {
		return nil, fmt.Errorf("Replicate API call failed: %w", err)
	}

	// Poll for completion if not streaming
	if resp.Status == "starting" || resp.Status == "processing" {
		resp, err = p.pollPrediction(ctx, resp.URLs.Get)
		if err != nil {
			return nil, fmt.Errorf("failed to poll prediction: %w", err)
		}
	}

	if resp.Status == "failed" {
		return nil, fmt.Errorf("prediction failed: %s", resp.Error)
	}

	return p.convertResponse(req, resp, startTime), nil
}

// CompleteStream sends a streaming completion request
func (p *Provider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()
	apiReq := p.convertRequest(req)
	apiReq.Stream = true

	// Create prediction
	resp, err := p.createPrediction(ctx, apiReq)
	if err != nil {
		return nil, fmt.Errorf("Replicate streaming API call failed: %w", err)
	}

	ch := make(chan *models.LLMResponse)
	go func() {
		defer close(ch)

		// For Replicate, streaming is done by polling the prediction
		var fullContent strings.Builder
		prevOutput := ""

		for {
			select {
			case <-ctx.Done():
				return
			case <-time.After(500 * time.Millisecond):
			}

			predResp, err := p.getPrediction(ctx, resp.URLs.Get)
			if err != nil {
				ch <- &models.LLMResponse{
					ID:           "stream-error-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "replicate",
					ProviderName: "Replicate",
					FinishReason: "error",
					CreatedAt:    time.Now(),
				}
				return
			}

			// Get current output
			currentOutput := p.extractOutput(predResp.Output)
			if len(currentOutput) > len(prevOutput) {
				delta := currentOutput[len(prevOutput):]
				fullContent.WriteString(delta)
				ch <- &models.LLMResponse{
					ID:           predResp.ID,
					RequestID:    req.ID,
					ProviderID:   "replicate",
					ProviderName: "Replicate",
					Content:      delta,
					FinishReason: "",
					CreatedAt:    time.Now(),
				}
				prevOutput = currentOutput
			}

			if predResp.Status == "succeeded" || predResp.Status == "failed" || predResp.Status == "canceled" {
				ch <- &models.LLMResponse{
					ID:           "stream-final-" + req.ID,
					RequestID:    req.ID,
					ProviderID:   "replicate",
					ProviderName: "Replicate",
					Content:      fullContent.String(),
					Confidence:   0.9,
					ResponseTime: time.Since(startTime).Milliseconds(),
					FinishReason: predResp.Status,
					CreatedAt:    time.Now(),
				}
				return
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
	return strings.TrimSuffix(p.baseURL, "/predictions") + "/models"
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
			"chat", "streaming", "image_generation", "speech_to_text",
			"text_to_image", "embeddings",
		},
		SupportedRequestTypes:   []string{"chat", "completion", "image", "audio"},
		SupportsStreaming:       true,
		SupportsFunctionCalling: false,
		SupportsVision:          true,
		SupportsTools:           false,
		SupportsReasoning:       true,
		SupportsCodeCompletion:  true,
		SupportsCodeAnalysis:    true,
		Limits: models.ModelLimits{
			MaxTokens:             4096,
			MaxInputLength:        4096,
			MaxOutputLength:       4096,
			MaxConcurrentRequests: 50,
		},
		Metadata: map[string]string{
			"provider":       "replicate",
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

func (p *Provider) convertRequest(req *models.LLMRequest) PredictionRequest {
	// Build prompt from messages
	var prompt strings.Builder
	for _, msg := range req.Messages {
		switch msg.Role {
		case "user":
			prompt.WriteString("[INST] ")
			prompt.WriteString(msg.Content)
			prompt.WriteString(" [/INST]")
		case "assistant":
			prompt.WriteString(msg.Content)
		}
	}

	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	input := PredictionInput{
		Prompt:       prompt.String(),
		SystemPrompt: req.Prompt,
		MaxNewTokens: maxTokens,
		Temperature:  req.ModelParams.Temperature,
		TopP:         req.ModelParams.TopP,
	}

	if len(req.ModelParams.StopSequences) > 0 {
		input.StopSequences = strings.Join(req.ModelParams.StopSequences, ",")
	}

	model := p.model
	if req.ModelParams.Model != "" {
		model = req.ModelParams.Model
	}

	return PredictionRequest{
		Model: model,
		Input: input,
	}
}

func (p *Provider) convertResponse(req *models.LLMRequest, resp *PredictionResponse, startTime time.Time) *models.LLMResponse {
	content := p.extractOutput(resp.Output)
	confidence := p.calculateConfidence(content, resp.Status)

	return &models.LLMResponse{
		ID:           resp.ID,
		RequestID:    req.ID,
		ProviderID:   "replicate",
		ProviderName: "Replicate",
		Content:      content,
		Confidence:   confidence,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: resp.Status,
		Metadata: map[string]any{
			"model":   resp.Model,
			"version": resp.Version,
		},
		CreatedAt: time.Now(),
	}
}

func (p *Provider) extractOutput(output any) string {
	switch v := output.(type) {
	case string:
		return v
	case []interface{}:
		var result strings.Builder
		for _, item := range v {
			if str, ok := item.(string); ok {
				result.WriteString(str)
			}
		}
		return result.String()
	default:
		if output != nil {
			data, _ := json.Marshal(output) //nolint:errcheck
			return string(data)
		}
		return ""
	}
}

func (p *Provider) calculateConfidence(content, status string) float64 {
	confidence := 0.85
	switch status {
	case "succeeded":
		confidence += 0.1
	case "failed":
		confidence -= 0.3
	case "canceled":
		confidence -= 0.2
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

func (p *Provider) createPrediction(ctx context.Context, req PredictionRequest) (*PredictionResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(respBody))
	}

	var predResp PredictionResponse
	if err := json.Unmarshal(respBody, &predResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &predResp, nil
}

func (p *Provider) getPrediction(ctx context.Context, url string) (*PredictionResponse, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var predResp PredictionResponse
	if err := json.Unmarshal(respBody, &predResp); err != nil {
		return nil, err
	}

	return &predResp, nil
}

func (p *Provider) pollPrediction(ctx context.Context, url string) (*PredictionResponse, error) {
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1 * time.Second):
		}

		resp, err := p.getPrediction(ctx, url)
		if err != nil {
			return nil, err
		}

		if resp.Status == "succeeded" || resp.Status == "failed" || resp.Status == "canceled" {
			return resp, nil
		}
	}
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
	return "replicate"
}
