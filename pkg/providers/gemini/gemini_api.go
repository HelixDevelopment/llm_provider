package gemini

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

// DefaultHTTPTimeout is the compiled fallback for this transport's HTTP client.
// It is deliberately NOT DefaultUnifiedTimeout (180s, declared in gemini.go):
// a plain HTTPS round trip and a CLI/ACP sub-provider round trip are not the
// same wait. Override with LLMPROVIDER_GEMINI_TIMEOUT.
const DefaultHTTPTimeout = 120 * time.Second

const (
	// GeminiDefaultModel is the default model for the Gemini API provider.
	GeminiDefaultModel = "gemini-2.5-flash"

	// thinkingBudgetDefault is the default thinking budget for extended-thinking models.
	thinkingBudgetDefault = 8192

	// maxOutputTokensLegacy is the safe maximum for older Gemini models.
	maxOutputTokensLegacy = 8192

	// maxOutputTokensExtended is the maximum for Gemini 2.5+ models.
	maxOutputTokensExtended = 65536
)

// thinkingModels lists models that support extended thinking.
var thinkingModels = map[string]bool{
	"gemini-2.5-pro":         true,
	"gemini-3-pro-preview":   true,
	"gemini-3.1-pro-preview": true,
}

// extendedTokenModels lists model prefixes that support the higher output token cap.
var extendedTokenPrefixes = []string{
	"gemini-2.5",
	"gemini-3",
}

// GeminiAPIProvider implements the LLMProvider interface for the Google Gemini
// API using direct REST calls. It supports non-streaming and streaming
// completions, function calling, vision, and extended thinking for supported
// models.
type GeminiAPIProvider struct {
	apiKey      string
	baseURL     string
	streamURL   string
	healthURL   string
	model       string
	httpClient  *http.Client
	retryConfig RetryConfig
	discoverer  *discovery.Discoverer
}

// GeminiAPIRequest represents a request to the Gemini API with extended tool
// support including Google Search grounding.
type GeminiAPIRequest struct {
	Contents         []GeminiContent           `json:"contents"`
	GenerationConfig GeminiAPIGenerationConfig `json:"generationConfig,omitempty"`
	SafetySettings   []GeminiSafetySetting     `json:"safetySettings,omitempty"`
	Tools            []GeminiToolDefExtended   `json:"tools,omitempty"`
	ToolConfig       *GeminiToolConfig         `json:"toolConfig,omitempty"`
}

// GeminiToolDefExtended represents a tool definition that supports both
// function declarations and Google Search grounding.
type GeminiToolDefExtended struct {
	FunctionDeclarations []GeminiFunctionDeclaration `json:"functionDeclarations,omitempty"`
	GoogleSearch         *GeminiGoogleSearch         `json:"googleSearch,omitempty"`
}

// GeminiGoogleSearch enables Google Search grounding in Gemini responses.
type GeminiGoogleSearch struct{}

// GeminiAPIGenerationConfig extends the base generation config with thinking
// support for extended-thinking models.
type GeminiAPIGenerationConfig struct {
	Temperature     float64               `json:"temperature,omitempty"`
	TopP            float64               `json:"topP,omitempty"`
	TopK            int                   `json:"topK,omitempty"`
	MaxOutputTokens int                   `json:"maxOutputTokens,omitempty"`
	StopSequences   []string              `json:"stopSequences,omitempty"`
	ThinkingConfig  *GeminiThinkingConfig `json:"thinkingConfig,omitempty"`
}

// GeminiThinkingConfig configures extended thinking for supported Gemini models.
type GeminiThinkingConfig struct {
	ThinkingBudget int `json:"thinkingBudget,omitempty"`
}

// NewGeminiAPIProvider creates a new GeminiAPIProvider with default retry
// configuration.
func NewGeminiAPIProvider(apiKey, baseURL, model string) *GeminiAPIProvider {
	return NewGeminiAPIProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

// NewGeminiAPIProviderWithRetry creates a new GeminiAPIProvider with the
// specified retry configuration.
func NewGeminiAPIProviderWithRetry(
	apiKey, baseURL, model string,
	retryConfig RetryConfig,
) *GeminiAPIProvider {
	if baseURL == "" {
		// LLMPROVIDER_GEMINI_BASE_URL.
		baseURL = settings.BaseURL(SettingsProvider, GeminiAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_GEMINI_MODEL.
		model = settings.Model(SettingsProvider, GeminiDefaultModel)
	}

	// Derive streaming URL from base URL
	streamURL := GeminiStreamAPIURL
	if baseURL != GeminiAPIURL {
		// Custom base URL - try to derive stream URL by replacing
		// generateContent with streamGenerateContent
		streamURL = baseURL
		if len(streamURL) > 15 &&
			streamURL[len(streamURL)-15:] == ":generateContent" {
			streamURL = streamURL[:len(streamURL)-15] +
				":streamGenerateContent"
		}
	}

	p := &GeminiAPIProvider{
		apiKey:    apiKey,
		baseURL:   baseURL,
		streamURL: streamURL,
		model:     model,
		httpClient: &http.Client{
			// LLMPROVIDER_GEMINI_TIMEOUT.
			Timeout: settings.Timeout(SettingsProvider, DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "gemini",
		ModelsEndpoint: "https://generativelanguage.googleapis.com/v1beta/models",
		ModelsDevID:    "google",
		APIKey:         apiKey,
		AuthHeader:     "x-goog-api-key",
		AuthPrefix:     "",
		ResponseParser: discovery.ParseGeminiModelsResponse,
		FallbackModels: []string{
			"gemini-2.0-flash",
			"gemini-2.5-flash",
			"gemini-2.5-flash-lite",
			"gemini-2.5-pro",
			"gemini-3-flash-preview",
			"gemini-3-pro-preview",
			"gemini-3.1-pro-preview",
		},
	})

	return p
}

// GetName returns the unique identifier for this provider instance.
func (p *GeminiAPIProvider) GetName() string {
	return "gemini-api"
}

// GetProviderType returns the provider family type.
func (p *GeminiAPIProvider) GetProviderType() string {
	return "gemini"
}

// Complete performs a non-streaming completion request against the Gemini API.
func (p *GeminiAPIProvider) Complete(
	ctx context.Context,
	req *models.LLMRequest,
) (*models.LLMResponse, error) {
	startTime := time.Now()

	geminiReq := p.convertRequest(req)

	resp, err := p.makeAPICall(ctx, geminiReq)
	if err != nil {
		return nil, fmt.Errorf("Gemini API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"Gemini API error: %d - %s",
			resp.StatusCode, string(body),
		)
	}

	var geminiResp GeminiResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini response: %w", err)
	}

	return p.convertResponse(req, &geminiResp, startTime), nil
}

// CompleteStream performs a streaming completion request against the Gemini
// API. Responses are sent on the returned channel as they arrive.
func (p *GeminiAPIProvider) CompleteStream(
	ctx context.Context,
	req *models.LLMRequest,
) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	geminiReq := p.convertRequest(req)

	if geminiReq.GenerationConfig.MaxOutputTokens == 0 {
		geminiReq.GenerationConfig.MaxOutputTokens = 2048
	}

	resp, err := p.makeStreamAPICall(ctx, geminiReq)
	if err != nil {
		return nil, fmt.Errorf(
			"Gemini streaming API call failed: %w", err,
		)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf(
				"Gemini API error: HTTP %d - failed to read response "+
					"body: %v",
				resp.StatusCode, readErr,
			)
		}
		return nil, fmt.Errorf(
			"Gemini API error: HTTP %d - %s",
			resp.StatusCode, string(body),
		)
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
				errorResp := &models.LLMResponse{
					ID:             "stream-error-" + req.ID,
					RequestID:      req.ID,
					ProviderID:     "gemini-api",
					ProviderName:   "Gemini",
					Content:        "",
					Confidence:     0.0,
					TokensUsed:     0,
					ResponseTime:   time.Since(startTime).Milliseconds(),
					FinishReason:   "error",
					Selected:       false,
					SelectionScore: 0.0,
					CreatedAt:      time.Now(),
				}
				ch <- errorResp
				return
			}

			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}

			if bytes.HasPrefix(line, []byte("data: ")) {
				line = bytes.TrimPrefix(line, []byte("data: "))
			}

			if bytes.Equal(line, []byte("[DONE]")) {
				break
			}

			var streamResp GeminiStreamResponse

			lineStr := string(line)
			if len(lineStr) > 0 && lineStr[0] == '[' {
				lineStr = strings.TrimPrefix(lineStr, "[")
				lineStr = strings.TrimSuffix(lineStr, "]")
				lineStr = strings.TrimPrefix(lineStr, ",")
				lineStr = strings.TrimSpace(lineStr)
				line = []byte(lineStr)
			}

			if err := json.Unmarshal(line, &streamResp); err != nil {
				var fullResp GeminiResponse
				if err2 := json.Unmarshal(line, &fullResp); err2 == nil &&
					len(fullResp.Candidates) > 0 {
					streamResp.Candidates = fullResp.Candidates
				} else {
					continue
				}
			}

			if len(streamResp.Candidates) > 0 {
				candidate := streamResp.Candidates[0]
				if len(candidate.Content.Parts) > 0 {
					for _, part := range candidate.Content.Parts {
						if part.Text != "" {
							fullContent += part.Text

							chunkResp := &models.LLMResponse{
								ID:           "gemini-api-stream-" + req.ID,
								RequestID:    req.ID,
								ProviderID:   "gemini-api",
								ProviderName: "Gemini",
								Content:      part.Text,
								Confidence:   0.85,
								TokensUsed:   1,
								ResponseTime: time.Since(
									startTime,
								).Milliseconds(),
								FinishReason:   "",
								Selected:       false,
								SelectionScore: 0.0,
								CreatedAt:      time.Now(),
							}
							ch <- chunkResp
						}
					}
				}

				if candidate.FinishReason != "" {
					break
				}
			}
		}

		finalResp := &models.LLMResponse{
			ID:             "gemini-api-final-" + req.ID,
			RequestID:      req.ID,
			ProviderID:     "gemini-api",
			ProviderName:   "Gemini",
			Content:        "",
			Confidence:     0.85,
			TokensUsed:     len(fullContent) / 4,
			ResponseTime:   time.Since(startTime).Milliseconds(),
			FinishReason:   "stop",
			Selected:       false,
			SelectionScore: 0.0,
			CreatedAt:      time.Now(),
		}
		ch <- finalResp
	}()

	return ch, nil
}

// convertRequest transforms an internal LLMRequest into a GeminiAPIRequest,
// applying extended thinking for supported models and always enabling Google
// Search grounding.
func (p *GeminiAPIProvider) convertRequest(
	req *models.LLMRequest,
) GeminiAPIRequest {
	contents := make([]GeminiContent, 0, len(req.Messages)+1)

	if req.Prompt != "" {
		contents = append(contents, GeminiContent{
			Parts: []GeminiPart{
				{Text: req.Prompt},
			},
			Role: "user",
		})
	}

	for _, msg := range req.Messages {
		role := "user"
		if msg.Role == "assistant" {
			role = "model"
		}

		contents = append(contents, GeminiContent{
			Parts: []GeminiPart{
				{Text: msg.Content},
			},
			Role: role,
		})
	}

	maxTokens := p.resolveMaxTokens(req.ModelParams.MaxTokens)

	genConfig := GeminiAPIGenerationConfig{
		Temperature:     req.ModelParams.Temperature,
		TopP:            req.ModelParams.TopP,
		MaxOutputTokens: maxTokens,
		StopSequences:   req.ModelParams.StopSequences,
	}

	// Enable extended thinking for supported models
	if thinkingModels[p.model] {
		genConfig.ThinkingConfig = &GeminiThinkingConfig{
			ThinkingBudget: thinkingBudgetDefault,
		}
	}

	geminiReq := GeminiAPIRequest{
		Contents:         contents,
		GenerationConfig: genConfig,
		SafetySettings: []GeminiSafetySetting{
			{
				Category:  "HARM_CATEGORY_HARASSMENT",
				Threshold: "BLOCK_NONE",
			},
			{
				Category:  "HARM_CATEGORY_HATE_SPEECH",
				Threshold: "BLOCK_NONE",
			},
			{
				Category:  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
				Threshold: "BLOCK_NONE",
			},
			{
				Category:  "HARM_CATEGORY_DANGEROUS_CONTENT",
				Threshold: "BLOCK_NONE",
			},
		},
	}

	// Build tools list — always include Google Search grounding
	var tools []GeminiToolDefExtended

	// Add function declarations if provided
	if len(req.Tools) > 0 {
		funcDecls := make(
			[]GeminiFunctionDeclaration, len(req.Tools),
		)
		for i, tool := range req.Tools {
			funcDecls[i] = GeminiFunctionDeclaration{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			}
		}
		tools = append(tools, GeminiToolDefExtended{
			FunctionDeclarations: funcDecls,
		})

		// Set tool config based on ToolChoice
		if req.ToolChoice != "" {
			mode := "AUTO"
			switch req.ToolChoice {
			case "none":
				mode = "NONE"
			case "auto":
				mode = "AUTO"
			case "required":
				mode = "ANY"
			}
			geminiReq.ToolConfig = &GeminiToolConfig{
				FunctionCallingConfig: &GeminiFunctionCallingConfig{
					Mode: mode,
				},
			}
		}
	}

	// Always add Google Search grounding
	tools = append(tools, GeminiToolDefExtended{
		GoogleSearch: &GeminiGoogleSearch{},
	})

	geminiReq.Tools = tools

	return geminiReq
}

// resolveMaxTokens determines the appropriate max output tokens cap based on
// the configured model.
func (p *GeminiAPIProvider) resolveMaxTokens(requested int) int {
	cap := maxOutputTokensLegacy
	for _, prefix := range extendedTokenPrefixes {
		if strings.HasPrefix(p.model, prefix) {
			cap = maxOutputTokensExtended
			break
		}
	}

	if requested <= 0 {
		return 4096
	}
	if requested > cap {
		return cap
	}
	return requested
}

// convertResponse transforms a GeminiResponse into the internal LLMResponse
// format, extracting thinking content into metadata when present.
func (p *GeminiAPIProvider) convertResponse(
	req *models.LLMRequest,
	geminiResp *GeminiResponse,
	startTime time.Time,
) *models.LLMResponse {
	var content string
	var thinkingContent string
	var finishReason string
	var tokensUsed int
	var toolCalls []models.ToolCall

	if len(geminiResp.Candidates) > 0 {
		candidate := geminiResp.Candidates[0]
		if len(candidate.Content.Parts) > 0 {
			for i, part := range candidate.Content.Parts {
				// Gemini returns thinking in parts that have a
				// "thought" field set to true. Since GeminiPart is
				// parsed from JSON, we check if the raw part
				// contains a thought marker via the Thought field.
				if part.Thought {
					thinkingContent += part.Text
					continue
				}
				if part.Text != "" {
					content += part.Text
				}
				if part.FunctionCall != nil {
					name, ok := part.FunctionCall["name"].(string)
					if !ok {
						name = ""
					}
					args, ok2 := part.FunctionCall["args"].(map[string]interface{})
					if !ok2 {
						args = map[string]interface{}{}
					}

					argsJSON, _ := json.Marshal(args) //nolint:errcheck

					toolCalls = append(toolCalls, models.ToolCall{
						ID:   fmt.Sprintf("call_%d", i),
						Type: "function",
						Function: models.ToolCallFunction{
							Name:      name,
							Arguments: string(argsJSON),
						},
					})
				}
			}
		}
		finishReason = candidate.FinishReason

		if len(toolCalls) > 0 &&
			(finishReason == "" || finishReason == "STOP") {
			finishReason = "tool_calls"
		}
	}

	if geminiResp.UsageMetadata != nil {
		tokensUsed = geminiResp.UsageMetadata.TotalTokenCount
	}

	confidence := p.calculateConfidence(content, finishReason)

	metadata := map[string]any{
		"model": p.model,
	}
	if thinkingContent != "" {
		metadata["thinking"] = thinkingContent
	}

	return &models.LLMResponse{
		ID:             "gemini-api-" + req.ID,
		RequestID:      req.ID,
		ProviderID:     "gemini-api",
		ProviderName:   "Gemini",
		Content:        content,
		Confidence:     confidence,
		TokensUsed:     tokensUsed,
		ResponseTime:   time.Since(startTime).Milliseconds(),
		FinishReason:   finishReason,
		ToolCalls:      toolCalls,
		Metadata:       metadata,
		Selected:       false,
		SelectionScore: 0.0,
		CreatedAt:      time.Now(),
	}
}

// calculateConfidence derives a confidence score from the response content and
// the Gemini finish reason.
func (p *GeminiAPIProvider) calculateConfidence(
	content, finishReason string,
) float64 {
	confidence := 0.85

	switch finishReason {
	case "STOP":
		confidence += 0.1
	case "MAX_TOKENS":
		confidence -= 0.1
	case "SAFETY":
		confidence -= 0.3
	case "RECITATION":
		confidence -= 0.2
	}

	if len(content) > 100 {
		confidence += 0.03
	}
	if len(content) > 500 {
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

// makeAPICall performs a non-streaming API call with retry and auth-retry
// support.
func (p *GeminiAPIProvider) makeAPICall(
	ctx context.Context,
	req GeminiAPIRequest,
) (*http.Response, error) {
	return p.makeAPICallWithAuthRetry(ctx, req, true)
}

// makeAPICallWithAuthRetry performs the API call with optional 401 retry to
// handle transient auth issues.
func (p *GeminiAPIProvider) makeAPICallWithAuthRetry(
	ctx context.Context,
	req GeminiAPIRequest,
	allowAuthRetry bool,
) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf(p.baseURL, p.model)

	var lastErr error
	delay := p.retryConfig.InitialDelay

	for attempt := 0; attempt <= p.retryConfig.MaxRetries; attempt++ {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
		default:
		}

		httpReq, err := http.NewRequestWithContext(
			ctx, "POST", url, bytes.NewBuffer(body),
		)
		if err != nil {
			return nil, fmt.Errorf(
				"failed to create request: %w", err,
			)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("x-goog-api-key", p.apiKey)
		httpReq.Header.Set("User-Agent", "LLMProvider/1.0")

		resp, err := p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("HTTP request failed: %w", err)
			if attempt < p.retryConfig.MaxRetries {
				p.waitWithJitter(ctx, delay)
				delay = p.nextDelay(delay)
				continue
			}
			return nil, lastErr
		}

		if isAuthRetryableStatus(resp.StatusCode) && allowAuthRetry {
			_ = resp.Body.Close()
			authRetryDelay := 500 * time.Millisecond
			p.waitWithJitter(ctx, authRetryDelay)
			return p.makeAPICallWithAuthRetry(ctx, req, false)
		}

		if isRetryableStatus(resp.StatusCode) &&
			attempt < p.retryConfig.MaxRetries {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf(
				"HTTP %d: retryable error", resp.StatusCode,
			)
			p.waitWithJitter(ctx, delay)
			delay = p.nextDelay(delay)
			continue
		}

		return resp, nil
	}

	return nil, fmt.Errorf(
		"all %d retry attempts failed: %w",
		p.retryConfig.MaxRetries+1, lastErr,
	)
}

// makeStreamAPICall performs a streaming API call with retry support.
func (p *GeminiAPIProvider) makeStreamAPICall(
	ctx context.Context,
	req GeminiAPIRequest,
) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf(p.streamURL, p.model)
	url += "?alt=sse&key=" + p.apiKey

	var lastErr error
	delay := p.retryConfig.InitialDelay

	for attempt := 0; attempt <= p.retryConfig.MaxRetries; attempt++ {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
		default:
		}

		httpReq, err := http.NewRequestWithContext(
			ctx, "POST", url, bytes.NewBuffer(body),
		)
		if err != nil {
			return nil, fmt.Errorf(
				"failed to create request: %w", err,
			)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")
		httpReq.Header.Set("User-Agent", "LLMProvider/1.0")

		resp, err := p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("HTTP request failed: %w", err)
			if attempt < p.retryConfig.MaxRetries {
				p.waitWithJitter(ctx, delay)
				delay = p.nextDelay(delay)
				continue
			}
			return nil, lastErr
		}

		if isRetryableStatus(resp.StatusCode) &&
			attempt < p.retryConfig.MaxRetries {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf(
				"HTTP %d: retryable error", resp.StatusCode,
			)
			p.waitWithJitter(ctx, delay)
			delay = p.nextDelay(delay)
			continue
		}

		return resp, nil
	}

	return nil, fmt.Errorf(
		"all %d retry attempts failed: %w",
		p.retryConfig.MaxRetries+1, lastErr,
	)
}

// waitWithJitter waits for the specified duration plus random jitter to avoid
// thundering herd effects on retries.
func (p *GeminiAPIProvider) waitWithJitter(
	ctx context.Context,
	delay time.Duration,
) {
	// Add 10% jitter - using math/rand is acceptable for non-security jitter
	jitter := time.Duration(
		rand.Float64() * 0.1 * float64(delay), // #nosec G404
	)
	select {
	case <-ctx.Done():
	case <-time.After(delay + jitter):
	}
}

// nextDelay calculates the next delay using exponential backoff capped at
// MaxDelay.
func (p *GeminiAPIProvider) nextDelay(
	currentDelay time.Duration,
) time.Duration {
	next := time.Duration(
		float64(currentDelay) * p.retryConfig.Multiplier,
	)
	if next > p.retryConfig.MaxDelay {
		next = p.retryConfig.MaxDelay
	}
	return next
}

// GetCapabilities returns the capabilities of the Gemini API provider.
func (p *GeminiAPIProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: p.discoverer.DiscoverModels(),
		SupportedFeatures: []string{
			"text_completion",
			"chat",
			"function_calling",
			"streaming",
			"vision",
			"search_grounding",
			"extended_thinking",
		},
		SupportedRequestTypes: []string{
			"text_completion",
			"chat",
		},
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          true,
		SupportsTools:           true,
		SupportsSearch:          true,
		SupportsReasoning:       true,
		SupportsCodeCompletion:  true,
		SupportsCodeAnalysis:    true,
		SupportsRefactoring:     false,
		Limits: models.ModelLimits{
			MaxTokens:             1048576,
			MaxInputLength:        1048576,
			MaxOutputLength:       maxOutputTokensExtended,
			MaxConcurrentRequests: 10,
		},
		Metadata: map[string]string{
			"provider":     "Google",
			"model_family": "Gemini",
			"api_version":  "v1beta",
		},
	}
}

// ValidateConfig validates the provider configuration.
func (p *GeminiAPIProvider) ValidateConfig(
	config map[string]interface{},
) (bool, []string) {
	var errors []string

	if p.apiKey == "" {
		// CONST-046 round-425: validation error routed through i18n.
		errors = append(errors, i18n.Tr(context.Background(),
			"llmprovider_validate_api_key_required", nil))
	}

	if p.baseURL == "" {
		// CONST-046 round-425: validation error routed through i18n.
		errors = append(errors, i18n.Tr(context.Background(),
			"llmprovider_validate_base_url_required", nil))
	}

	if p.model == "" {
		// CONST-046 round-425: validation error routed through i18n.
		errors = append(errors, i18n.Tr(context.Background(),
			"llmprovider_validate_model_required", nil))
	}

	return len(errors) == 0, errors
}

// HealthCheck performs a health check against the Gemini API by listing
// available models.
func (p *GeminiAPIProvider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(
		context.Background(), 10*time.Second,
	)
	defer cancel()

	healthURL := p.healthURL
	if healthURL == "" {
		// LLMPROVIDER_GEMINI_MODELS_BASE_URL — keyed apart from the
		// generateContent endpoint; see SettingsProviderModels in gemini.go.
		healthURL = settings.BaseURL(SettingsProviderModels, GeminiModelsURL)
	}

	req, err := http.NewRequestWithContext(
		ctx, "GET", healthURL+"?key="+p.apiKey, nil,
	)
	if err != nil {
		return fmt.Errorf(
			"failed to create health check request: %w", err,
		)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf(
			"health check failed with status: %d", resp.StatusCode,
		)
	}

	return nil
}
