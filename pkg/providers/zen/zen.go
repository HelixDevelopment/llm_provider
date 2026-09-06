package zen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand" // Used for non-security operations (jitter)
	"net/http"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"digital.vasic.llmprovider/pkg/i18n"
	"digital.vasic.llmprovider/pkg/models"
	"digital.vasic.llmprovider/pkg/settings"
)

// Env-var provider keys for this file's settings.
//
// WHY THIS FILE USES A DIFFERENT KEY FOR THE ENDPOINT THAN zen_http.go DOES.
// The two files are two TRANSPORTS for the same vendor, and they do not talk to
// the same place: this one calls the hosted Zen API at ZenAPIURL, while
// zen_http.go talks to a locally running opencode server at
// DefaultZenBaseURL ("http://localhost:4096"). Sharing one
// LLMPROVIDER_ZEN_BASE_URL between them would mean an operator pointing the
// local provider at their own server would SILENTLY redirect the cloud provider
// to a host that does not speak that protocol — precisely the quiet
// misconfiguration this whole settings layer exists to prevent. So the endpoint
// and the transport timeout are keyed to "zen_api", while the MODEL stays on
// "zen": the model id genuinely is shared between the two transports, and
// zen_http.go already resolves it under that key.
const (
	// SettingsProviderAPI keys the hosted-API endpoint and timeout:
	// LLMPROVIDER_ZEN_API_BASE_URL and LLMPROVIDER_ZEN_API_TIMEOUT.
	SettingsProviderAPI = "zen_api"
	// SettingsProviderModel keys the shared default model:
	// LLMPROVIDER_ZEN_MODEL. Same key zen_http.go uses, deliberately.
	SettingsProviderModel = "zen"
)

// DefaultZenAPITimeout is the compiled fallback for the hosted-API transport.
// It is deliberately NOT DefaultZenTimeout (180s, declared in zen_http.go for
// the local opencode server): a cloud round-trip and a localhost round-trip are
// not the same wait. Override with LLMPROVIDER_ZEN_API_TIMEOUT.
const DefaultZenAPITimeout = 120 * time.Second

// SettingsProviderModels keys the model-discovery probe:
// LLMPROVIDER_ZEN_MODELS_TIMEOUT.
const SettingsProviderModels = "zen_models"

// DefaultZenModelsTimeout is the compiled fallback for that probe. It is
// deliberately far shorter than either completion timeout: listing models is
// a cheap GET, and a slow one should fail fast rather than stall a caller.
const DefaultZenModelsTimeout = 10 * time.Second

var log = logrus.New()

const (
	// ZenAPIURL is the base URL for OpenCode Zen API
	ZenAPIURL = "https://opencode.ai/zen/v1/chat/completions"
	// ZenModelsURL is the endpoint to fetch available models
	ZenModelsURL = "https://opencode.ai/zen/v1/models"

	// Free models - available WITHOUT API key (verified 2026-02-22)
	// NOTE: Zen API requires model names WITHOUT "opencode/" prefix
	// These models work with anonymous access (X-Device-ID header):
	//   - big-pickle (Minimax M2.5 routing)
	//   - glm-5-free (GLM-5 Free tier)
	//   - minimax-m2.5-free (Minimax M2.5 Free)
	//   - minimax-m2.1-free (Minimax M2.1 Free)
	//   - trinity-large-preview-free (Arcee AI Trinity)
	ModelBigPickle               = "big-pickle"
	ModelGLM5Free                = "glm-5-free"
	ModelMinimaxM25Free          = "minimax-m2.5-free"
	ModelMinimaxM21Free          = "minimax-m2.1-free"
	ModelTrinityLargePreviewFree = "trinity-large-preview-free"

	// Paid models - require API key (listed for reference)
	ModelGLM47   = "glm-4.7"
	ModelKimiK2  = "kimi-k2"
	ModelGemini3 = "gemini-3-flash"

	// Deprecated: ModelQwen3 was removed from the free tier in 2026-02.
	// Use ModelBigPickle or ModelGLM5Free instead.
	ModelQwen3 = "qwen3-coder"
	// Deprecated: ModelGrokCodeFast is no longer available on the Zen platform.
	// Use ModelBigPickle or ModelTrinityLargePreviewFree instead.
	ModelGrokCodeFast = "grok-code"
	// Deprecated: ModelGLM47Free has been superseded by ModelGLM5Free.
	ModelGLM47Free = "glm-4.7-free"

	// Legacy model IDs with prefix (for backward compatibility in configs).
	ModelBigPickleFull = "opencode/big-pickle"
	// Deprecated: ModelGrokCodeFastFull is the prefixed form of the deprecated
	// ModelGrokCodeFast. Use ModelBigPickleFull instead.
	ModelGrokCodeFastFull = "opencode/grok-code"
	// Deprecated: ModelGLM47FreeFull is the prefixed form of the deprecated
	// ModelGLM47Free. Use ModelGLM5Free instead.
	ModelGLM47FreeFull = "opencode/glm-4.7-free"
	// Deprecated: ModelGPT5NanoFull is not available on the Zen platform.
	// It was listed speculatively and never worked.
	ModelGPT5NanoFull = "opencode/gpt-5-nano"

	// Default model for Zen provider - using verified working model
	DefaultZenModel = ModelBigPickle

	// Anonymous access identifier for free models
	AnonymousDeviceHeader = "X-Device-ID"

	// Model discovery cache TTL
	ModelDiscoveryCacheTTL = 6 * time.Hour
)

var (
	// Cached discovered models
	discoveredModels      []string
	discoveredModelsTime  time.Time
	discoveredModelsMutex sync.RWMutex

	// Known free models as fallback (verified 2026-02-22)
	// These models work with anonymous X-Device-ID access
	knownFreeModels = []string{
		ModelBigPickle,
		ModelGLM5Free,
		ModelMinimaxM25Free,
		ModelMinimaxM21Free,
		ModelTrinityLargePreviewFree,
	}
)

// FreeModels returns the list of free models available on Zen.
// Merges dynamically discovered models with the static known list
// so that known models aren't dropped when the API returns a partial response.
func FreeModels() []string {
	discovered := DiscoverFreeModels()
	// Merge: start with discovered, add any missing known models
	seen := make(map[string]struct{}, len(discovered)+len(knownFreeModels))
	merged := make([]string, 0, len(discovered)+len(knownFreeModels))
	for _, m := range discovered {
		if _, ok := seen[m]; !ok {
			seen[m] = struct{}{}
			merged = append(merged, m)
		}
	}
	for _, m := range knownFreeModels {
		if _, ok := seen[m]; !ok {
			seen[m] = struct{}{}
			merged = append(merged, m)
		}
	}
	if len(merged) > 0 {
		return merged
	}
	return knownFreeModels
}

// DiscoverFreeModels dynamically discovers available free models
// Uses cached results if fresh (< 6 hours old)
// Discovery order: 1) Zen API, 2) OpenCode CLI, 3) Known list
func DiscoverFreeModels() []string {
	// Check cache first
	discoveredModelsMutex.RLock()
	if time.Since(discoveredModelsTime) < ModelDiscoveryCacheTTL && len(discoveredModels) > 0 {
		models := make([]string, len(discoveredModels))
		copy(models, discoveredModels)
		discoveredModelsMutex.RUnlock()
		return models
	}
	discoveredModelsMutex.RUnlock()

	// Try discovery methods
	var models []string

	// Method 1: Fetch from Zen API
	models = discoverModelsFromAPI()
	if len(models) > 0 {
		cacheDiscoveredModels(models)
		return models
	}

	// Method 2: Use OpenCode CLI
	models = discoverModelsFromCLI()
	if len(models) > 0 {
		cacheDiscoveredModels(models)
		return models
	}

	// Method 3: Return known models
	cacheDiscoveredModels(knownFreeModels)
	return knownFreeModels
}

// discoverModelsFromAPI fetches models from Zen API
func discoverModelsFromAPI() []string {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", ZenModelsURL, nil)
	if err != nil {
		log.WithError(err).Debug("Failed to create Zen API request")
		return nil
	}

	// Anonymous access with device ID
	req.Header.Set(AnonymousDeviceHeader, generateDeviceID())
	req.Header.Set("Content-Type", "application/json")

	// A short probe, not a completion: LLMPROVIDER_ZEN_MODELS_TIMEOUT keys it
	// apart from the completion timeouts so shortening a discovery probe does
	// not shorten a generation, and vice versa.
	client := &http.Client{Timeout: settings.Timeout(SettingsProviderModels, DefaultZenModelsTimeout)}
	resp, err := client.Do(req)
	if err != nil {
		log.WithError(err).Debug("Failed to fetch models from Zen API")
		return nil
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		log.WithField("status", resp.StatusCode).Debug("Zen API returned non-200 status")
		return nil
	}

	var modelsResp ZenModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelsResp); err != nil {
		log.WithError(err).Debug("Failed to decode Zen API response")
		return nil
	}

	// Extract model IDs, filter for free models (opencode/ prefix or in known list)
	models := make([]string, 0)
	for _, model := range modelsResp.Data {
		modelID := normalizeModelID(model.ID)
		// Include if it's a known free model or has opencode prefix
		if strings.HasPrefix(model.ID, "opencode/") || contains(knownFreeModels, modelID) {
			if !contains(models, modelID) {
				models = append(models, modelID)
			}
		}
	}

	if len(models) > 0 {
		log.WithField("count", len(models)).Info("Discovered Zen models from API")
	}
	return models
}

// discoverModelsFromCLI uses OpenCode CLI to discover models
func discoverModelsFromCLI() []string {
	// Check if opencode CLI is available
	path, err := exec.LookPath("opencode")
	if err != nil {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, path, "models")
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.WithError(err).Debug("Failed to run opencode CLI")
		return nil
	}

	// Parse output - one model per line
	lines := strings.Split(string(output), "\n")
	models := make([]string, 0)
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Normalize model ID (remove opencode/ prefix)
		modelID := normalizeModelID(line)
		// Only include if it looks like a free model
		if strings.HasPrefix(line, "opencode/") || contains(knownFreeModels, modelID) {
			if !contains(models, modelID) {
				models = append(models, modelID)
			}
		}
	}

	if len(models) > 0 {
		log.WithField("count", len(models)).Info("Discovered Zen models from CLI")
	}
	return models
}

// cacheDiscoveredModels stores models in cache
func cacheDiscoveredModels(models []string) {
	discoveredModelsMutex.Lock()
	defer discoveredModelsMutex.Unlock()
	discoveredModels = make([]string, len(models))
	copy(discoveredModels, models)
	discoveredModelsTime = time.Now()
}

// contains checks if a string is in a slice
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// generateDeviceID generates a cryptographically secure unique device identifier for anonymous access
func generateDeviceID() string {
	return "helix-" + fmt.Sprintf("%d", time.Now().UnixNano())
}

// IsAnonymousAccessAllowed checks if a model can be used without API key
func IsAnonymousAccessAllowed(model string) bool {
	return isFreeModel(model)
}

// ZenProvider implements the LLM provider interface for OpenCode Zen
// Supports both authenticated (API key) and anonymous (free models only) access
type ZenProvider struct {
	apiKey        string
	baseURL       string
	model         string
	httpClient    *http.Client
	retryConfig   RetryConfig
	deviceID      string // For anonymous access to free models
	anonymousMode bool   // True when using free models without API key
}

// RetryConfig defines retry behavior for API calls
type RetryConfig struct {
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

// ZenRequest represents a request to Zen API
type ZenRequest struct {
	Model       string       `json:"model"`
	Messages    []ZenMessage `json:"messages"`
	Temperature float64      `json:"temperature,omitempty"`
	MaxTokens   int          `json:"max_tokens,omitempty"`
	TopP        float64      `json:"top_p,omitempty"`
	Stream      bool         `json:"stream,omitempty"`
}

// ZenMessage represents a message in Zen API
type ZenMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ZenResponse represents a response from Zen API
type ZenResponse struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created"`
	Model   string      `json:"model"`
	Choices []ZenChoice `json:"choices"`
	Usage   ZenUsage    `json:"usage"`
}

// ZenChoice represents a choice in the response
type ZenChoice struct {
	Index        int        `json:"index"`
	Message      ZenMessage `json:"message"`
	FinishReason string     `json:"finish_reason"`
}

// ZenUsage represents token usage
type ZenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ZenStreamResponse represents a streaming response chunk
type ZenStreamResponse struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []ZenStreamChoice `json:"choices"`
}

// ZenStreamChoice represents a choice in a streaming response
type ZenStreamChoice struct {
	Index        int        `json:"index"`
	Delta        ZenMessage `json:"delta"`
	FinishReason *string    `json:"finish_reason"`
}

// ZenErrorResponse represents an error response
type ZenErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

// ZenModelInfo represents model information from the models endpoint
type ZenModelInfo struct {
	ID              string `json:"id"`
	Object          string `json:"object"`
	OwnedBy         string `json:"owned_by"`
	Created         int64  `json:"created"`
	ContextWindow   int    `json:"context_window"`
	MaxOutputTokens int    `json:"max_output_tokens"`
	Pricing         struct {
		Input       float64 `json:"input"`
		Output      float64 `json:"output"`
		CachedRead  float64 `json:"cached_read"`
		CachedWrite float64 `json:"cached_write"`
	} `json:"pricing"`
	Capabilities []string `json:"capabilities"`
}

// ZenModelsResponse represents the response from the models endpoint
type ZenModelsResponse struct {
	Object string         `json:"object"`
	Data   []ZenModelInfo `json:"data"`
}

// DefaultRetryConfig returns sensible defaults for Zen API retry behavior
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

// NewZenProvider creates a new Zen provider instance
func NewZenProvider(apiKey, baseURL, model string) *ZenProvider {
	return NewZenProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

// NewZenProviderWithRetry creates a new Zen provider with custom retry config
// If apiKey is empty and model is a free model, anonymous mode is enabled
func NewZenProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *ZenProvider {
	if baseURL == "" {
		// The compiled constant is a FALLBACK, not a decision this library
		// gets to keep making. See pkg/settings:
		// LLMPROVIDER_ZEN_API_BASE_URL.
		baseURL = settings.BaseURL(SettingsProviderAPI, ZenAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_ZEN_MODEL — shared with zen_http.go by design.
		model = settings.Model(SettingsProviderModel, DefaultZenModel)
	}

	// Determine if we're in anonymous mode (no API key, free model)
	anonymousMode := apiKey == "" && isFreeModel(model)
	deviceID := ""
	if anonymousMode {
		deviceID = generateDeviceID()
		log.WithFields(logrus.Fields{
			"provider": "zen",
			"model":    model,
			"mode":     "anonymous",
		}).Info("Zen provider initialized in anonymous mode for free model")
	}

	return &ZenProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			// LLMPROVIDER_ZEN_API_TIMEOUT.
			Timeout: settings.Timeout(SettingsProviderAPI, DefaultZenAPITimeout),
		},
		retryConfig:   retryConfig,
		deviceID:      deviceID,
		anonymousMode: anonymousMode,
	}
}

// NewZenProviderAnonymous creates a Zen provider for anonymous access (free models only)
func NewZenProviderAnonymous(model string) *ZenProvider {
	if model == "" {
		// LLMPROVIDER_ZEN_MODEL. An operator may name a DIFFERENT free model
		// here; the clamp below still enforces the free-only invariant.
		model = settings.Model(SettingsProviderModel, DefaultZenModel)
	}
	// Ensure only free models can be used anonymously.
	//
	// This clamp deliberately does NOT consult the environment. It is a safety
	// invariant, not a default: its whole job is to refuse a model that would
	// bill an anonymous caller, and resolving it through settings would let the
	// same variable that caused the violation also choose the remedy — so a
	// non-free LLMPROVIDER_ZEN_MODEL would land right back here. The compiled
	// free model is the only value guaranteed to satisfy the invariant.
	if !isFreeModel(model) {
		log.WithField("model", model).Warn("Non-free model requested for anonymous access, defaulting to free model")
		model = DefaultZenModel
	}
	// Pass an EMPTY baseURL rather than ZenAPIURL. Handing the constant in
	// positionally made this call bypass the resolution in
	// NewZenProviderWithRetry entirely, so LLMPROVIDER_ZEN_API_BASE_URL was
	// honoured for every construction path except this one — the frozen value
	// won silently, which is the worst shape this defect takes.
	return NewZenProviderWithRetry("", "", model, DefaultRetryConfig())
}

// Complete performs a non-streaming completion request
func (p *ZenProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()
	requestID := req.ID
	if requestID == "" {
		requestID = fmt.Sprintf("zen-%d", time.Now().UnixNano())
	}

	log.WithFields(logrus.Fields{
		"provider":   "zen",
		"model":      p.model,
		"request_id": requestID,
		"messages":   len(req.Messages),
	}).Debug("Starting Zen API call")

	// Convert internal request to Zen format
	zenReq := p.convertRequest(req)

	log.WithFields(logrus.Fields{
		"provider":   "zen",
		"request_id": requestID,
		"max_tokens": zenReq.MaxTokens,
		"temp":       zenReq.Temperature,
	}).Debug("Request converted, making API call")

	// Make API call
	resp, err := p.makeAPICall(ctx, zenReq)
	if err != nil {
		log.WithFields(logrus.Fields{
			"provider":   "zen",
			"request_id": requestID,
			"error":      err.Error(),
			"duration":   time.Since(startTime).String(),
		}).Error("Zen API call failed")
		return nil, fmt.Errorf("Zen API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Parse response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.WithFields(logrus.Fields{
			"provider":   "zen",
			"request_id": requestID,
			"error":      err.Error(),
		}).Error("Failed to read response body")
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	log.WithFields(logrus.Fields{
		"provider":    "zen",
		"request_id":  requestID,
		"status_code": resp.StatusCode,
		"body_size":   len(body),
	}).Debug("Received API response")

	if resp.StatusCode != http.StatusOK {
		var errResp ZenErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			log.WithFields(logrus.Fields{
				"provider":    "zen",
				"request_id":  requestID,
				"status_code": resp.StatusCode,
				"error_msg":   errResp.Error.Message,
				"error_type":  errResp.Error.Type,
			}).Error("Zen API returned error")
			return nil, fmt.Errorf("Zen API error: %d - %s", resp.StatusCode, errResp.Error.Message)
		}
		log.WithFields(logrus.Fields{
			"provider":    "zen",
			"request_id":  requestID,
			"status_code": resp.StatusCode,
			"body":        string(body[:min(500, len(body))]),
		}).Error("Zen API error response")
		return nil, fmt.Errorf("Zen API error: %d - %s", resp.StatusCode, string(body))
	}

	var zenResp ZenResponse
	if err := json.Unmarshal(body, &zenResp); err != nil {
		log.WithFields(logrus.Fields{
			"provider":   "zen",
			"request_id": requestID,
			"error":      err.Error(),
			"body":       string(body[:min(200, len(body))]),
		}).Error("Failed to parse Zen response")
		return nil, fmt.Errorf("failed to parse Zen response: %w", err)
	}

	// Check for empty choices
	if len(zenResp.Choices) == 0 {
		log.WithFields(logrus.Fields{
			"provider":   "zen",
			"request_id": requestID,
			"response":   string(body[:min(500, len(body))]),
		}).Error("Zen API returned no choices")
		return nil, fmt.Errorf("Zen API returned no choices")
	}

	duration := time.Since(startTime)
	log.WithFields(logrus.Fields{
		"provider":      "zen",
		"request_id":    requestID,
		"duration":      duration.String(),
		"tokens_used":   zenResp.Usage.TotalTokens,
		"content_len":   len(zenResp.Choices[0].Message.Content),
		"finish_reason": zenResp.Choices[0].FinishReason,
	}).Info("Zen API call completed successfully")

	// Convert back to internal format
	return p.convertResponse(req, &zenResp, startTime), nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// CompleteStream performs a streaming completion request
func (p *ZenProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	// Convert internal request to Zen format
	zenReq := p.convertRequest(req)
	zenReq.Stream = true

	// Make streaming API call
	resp, err := p.makeAPICall(ctx, zenReq)
	if err != nil {
		return nil, fmt.Errorf("Zen streaming API call failed: %w", err)
	}

	// Check for HTTP errors before starting stream
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()            //nolint:errcheck
		return nil, fmt.Errorf("Zen API error: HTTP %d - %s", resp.StatusCode, string(body))
	}

	// Create response channel
	ch := make(chan *models.LLMResponse)

	go func() {
		defer func() { _ = resp.Body.Close() }() //nolint:errcheck
		defer close(ch)

		reader := bufio.NewReader(resp.Body)
		var fullContent string

		for {
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				// Send error response and exit
				errorResp := &models.LLMResponse{
					ID:             "stream-error-" + req.ID,
					RequestID:      req.ID,
					ProviderID:     "zen",
					ProviderName:   "OpenCode Zen",
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

			// Skip empty lines and "data: " prefix
			line = bytes.TrimSpace(line)
			if !bytes.HasPrefix(line, []byte("data: ")) {
				continue
			}
			line = bytes.TrimPrefix(line, []byte("data: "))

			// Skip "[DONE]" marker
			if bytes.Equal(line, []byte("[DONE]")) {
				break
			}

			// Parse JSON
			var streamResp ZenStreamResponse
			if err := json.Unmarshal(line, &streamResp); err != nil {
				continue // Skip malformed JSON
			}

			// Extract content
			if len(streamResp.Choices) > 0 {
				delta := streamResp.Choices[0].Delta.Content
				if delta != "" {
					fullContent += delta

					// Send chunk response
					chunkResp := &models.LLMResponse{
						ID:             streamResp.ID,
						RequestID:      req.ID,
						ProviderID:     "zen",
						ProviderName:   "OpenCode Zen",
						Content:        delta,
						Confidence:     0.8,
						TokensUsed:     1,
						ResponseTime:   time.Since(startTime).Milliseconds(),
						FinishReason:   "",
						Selected:       false,
						SelectionScore: 0.0,
						CreatedAt:      time.Now(),
					}
					ch <- chunkResp
				}

				// Check if stream is finished
				if streamResp.Choices[0].FinishReason != nil {
					break
				}
			}
		}

		// Send final response
		finalResp := &models.LLMResponse{
			ID:             "stream-final-" + req.ID,
			RequestID:      req.ID,
			ProviderID:     "zen",
			ProviderName:   "OpenCode Zen",
			Content:        "",
			Confidence:     0.8,
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

// convertRequest converts internal request to Zen format
func (p *ZenProvider) convertRequest(req *models.LLMRequest) ZenRequest {
	// Convert messages
	messages := make([]ZenMessage, 0, len(req.Messages)+1)

	// Add system prompt if present
	if req.Prompt != "" {
		messages = append(messages, ZenMessage{
			Role:    "system",
			Content: req.Prompt,
		})
	}

	// Add conversation messages
	for _, msg := range req.Messages {
		messages = append(messages, ZenMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	// Cap max_tokens to reasonable limits for free models
	maxTokens := req.ModelParams.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096 // Default
	} else if maxTokens > 16384 {
		maxTokens = 16384 // Zen's typical limit
	}

	// Determine which model to use
	model := p.model
	if req.ModelParams.Model != "" {
		model = normalizeModelID(req.ModelParams.Model)
	}

	return ZenRequest{
		Model:       model,
		Messages:    messages,
		Temperature: req.ModelParams.Temperature,
		MaxTokens:   maxTokens,
		TopP:        req.ModelParams.TopP,
		Stream:      false,
	}
}

// convertResponse converts Zen response to internal format
func (p *ZenProvider) convertResponse(req *models.LLMRequest, zenResp *ZenResponse, startTime time.Time) *models.LLMResponse {
	var content string
	var finishReason string

	if len(zenResp.Choices) > 0 {
		content = zenResp.Choices[0].Message.Content
		finishReason = zenResp.Choices[0].FinishReason
	}

	// Calculate confidence based on finish reason and response quality
	confidence := p.calculateConfidence(content, finishReason)

	// Extract model name for display
	modelName := zenResp.Model
	if modelName == "" {
		modelName = p.model
	}

	return &models.LLMResponse{
		ID:           zenResp.ID,
		RequestID:    req.ID,
		ProviderID:   "zen",
		ProviderName: "OpenCode Zen",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   zenResp.Usage.TotalTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":             modelName,
			"prompt_tokens":     zenResp.Usage.PromptTokens,
			"completion_tokens": zenResp.Usage.CompletionTokens,
			"free_model":        isFreeModel(modelName),
		},
		Selected:       false,
		SelectionScore: 0.0,
		CreatedAt:      time.Now(),
	}
}

// isFreeModel checks if a model is in the free tier.
// It checks both the dynamically discovered list AND the static known list,
// so models aren't misclassified when the API returns a partial response.
func isFreeModel(model string) bool {
	normalizedModel := normalizeModelID(model)
	// Check static known free models first (no network dependency)
	for _, m := range knownFreeModels {
		if m == normalizedModel {
			return true
		}
	}
	// Also check dynamically discovered models (may include new free models)
	for _, m := range FreeModels() {
		if m == normalizedModel {
			return true
		}
	}
	return false
}

// normalizeModelID strips the "opencode/" prefix if present
// Zen API requires model names WITHOUT the prefix (e.g., "grok-code" not "opencode/grok-code")
func normalizeModelID(modelID string) string {
	// Strip "opencode/" prefix if present
	if strings.HasPrefix(modelID, "opencode/") {
		return strings.TrimPrefix(modelID, "opencode/")
	}
	// Strip "opencode-" prefix if present (alternate format)
	if strings.HasPrefix(modelID, "opencode-") {
		return strings.TrimPrefix(modelID, "opencode-")
	}
	return modelID
}

// calculateConfidence calculates confidence score based on response
func (p *ZenProvider) calculateConfidence(content, finishReason string) float64 {
	confidence := 0.8 // Base confidence for free models

	// Adjust based on finish reason
	switch finishReason {
	case "stop":
		confidence += 0.1
	case "length":
		confidence -= 0.1
	}

	// Adjust based on content length
	if len(content) > 100 {
		confidence += 0.05
	}
	if len(content) > 500 {
		confidence += 0.05
	}

	// Ensure confidence is within bounds
	if confidence > 1.0 {
		confidence = 1.0
	}
	if confidence < 0.0 {
		confidence = 0.0
	}

	return confidence
}

// makeAPICall performs the HTTP request to Zen API
func (p *ZenProvider) makeAPICall(ctx context.Context, req ZenRequest) (*http.Response, error) {
	return p.makeAPICallWithAuthRetry(ctx, req, true)
}

// makeAPICallWithAuthRetry performs the API call with optional 401 retry
func (p *ZenProvider) makeAPICallWithAuthRetry(ctx context.Context, req ZenRequest, allowAuthRetry bool) (*http.Response, error) {
	// Marshal request
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var lastErr error
	delay := p.retryConfig.InitialDelay

	for attempt := 0; attempt <= p.retryConfig.MaxRetries; attempt++ {
		// Check context before making request
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context cancelled: %w", ctx.Err())
		default:
		}

		// Create HTTP request (fresh for each attempt)
		httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL, bytes.NewBuffer(body))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		// Set headers - Zen uses Bearer token auth or device ID for anonymous access
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("User-Agent", "LLMProvider/1.0")
		if p.anonymousMode {
			// Anonymous mode: use device ID for free models
			httpReq.Header.Set(AnonymousDeviceHeader, p.deviceID)
		} else if p.apiKey != "" {
			// Authenticated mode: use Bearer token
			httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
		}

		// Make request
		resp, err := p.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("HTTP request failed: %w", err)
			// Retry on network errors
			if attempt < p.retryConfig.MaxRetries {
				p.waitWithJitter(ctx, delay)
				delay = p.nextDelay(delay)
				continue
			}
			return nil, lastErr
		}

		// Check for auth errors (401) - retry once with a short delay
		if isAuthRetryableStatus(resp.StatusCode) && allowAuthRetry {
			_ = resp.Body.Close()
			log.WithFields(logrus.Fields{
				"provider":    "zen",
				"status_code": resp.StatusCode,
				"attempt":     attempt + 1,
			}).Warn("Received 401 Unauthorized, retrying once after short delay")

			// Short delay before auth retry (500ms with jitter)
			authRetryDelay := 500 * time.Millisecond
			p.waitWithJitter(ctx, authRetryDelay)

			// Recursive call with auth retry disabled to prevent infinite loops
			return p.makeAPICallWithAuthRetry(ctx, req, false)
		}

		// Check for retryable status codes (429, 5xx)
		if isRetryableStatus(resp.StatusCode) && attempt < p.retryConfig.MaxRetries {
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("HTTP %d: retryable error", resp.StatusCode)
			log.WithFields(logrus.Fields{
				"provider":    "zen",
				"status_code": resp.StatusCode,
				"attempt":     attempt + 1,
				"max_retries": p.retryConfig.MaxRetries,
			}).Debug("Retrying after retryable error")
			p.waitWithJitter(ctx, delay)
			delay = p.nextDelay(delay)
			continue
		}

		return resp, nil
	}

	return nil, fmt.Errorf("all %d retry attempts failed: %w", p.retryConfig.MaxRetries+1, lastErr)
}

// isRetryableStatus returns true for HTTP status codes that warrant a retry
func isRetryableStatus(statusCode int) bool {
	switch statusCode {
	case http.StatusTooManyRequests, // 429 - Rate limited
		http.StatusInternalServerError, // 500
		http.StatusBadGateway,          // 502
		http.StatusServiceUnavailable,  // 503
		http.StatusGatewayTimeout:      // 504
		return true
	default:
		return false
	}
}

// isAuthRetryableStatus returns true for auth errors that may be transient
func isAuthRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusUnauthorized // 401
}

// waitWithJitter waits for the specified duration plus random jitter
func (p *ZenProvider) waitWithJitter(ctx context.Context, delay time.Duration) {
	select {
	case <-ctx.Done():
	case <-time.After(jitteredDelay(delay)):
	}
}

// jitteredDelay returns delay plus up to 10% random jitter -- the exact
// duration waitWithJitter arms its timer with.
//
// Extracted so that the "at most 10% over" bound can be asserted directly and
// deterministically. Asserting it through wall-clock elapsed time does not
// work: elapsed time also contains scheduler latency the host controls, so a
// wall-clock ceiling measures the machine at least as much as it measures this
// package, and fails on a busy host while the code is correct.
func jitteredDelay(delay time.Duration) time.Duration {
	// Add 10% jitter - using math/rand is acceptable for non-security jitter
	jitter := time.Duration(rand.Float64() * 0.1 * float64(delay)) // #nosec G404 - jitter doesn't require cryptographic randomness
	return delay + jitter
}

// nextDelay calculates the next delay using exponential backoff
func (p *ZenProvider) nextDelay(currentDelay time.Duration) time.Duration {
	nextDelay := time.Duration(float64(currentDelay) * p.retryConfig.Multiplier)
	if nextDelay > p.retryConfig.MaxDelay {
		nextDelay = p.retryConfig.MaxDelay
	}
	return nextDelay
}

// GetCapabilities returns the capabilities of the Zen provider
func (p *ZenProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: []string{
			ModelBigPickle,
			ModelGLM5Free,
			ModelMinimaxM25Free,
			ModelMinimaxM21Free,
			ModelTrinityLargePreviewFree,
		},
		SupportedFeatures: []string{
			"text_completion",
			"chat",
			"streaming",
			"code_completion",
		},
		SupportedRequestTypes: []string{
			"text_completion",
			"chat",
		},
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          false,
		SupportsTools:           true,
		SupportsSearch:          false,
		SupportsReasoning:       true,
		SupportsCodeCompletion:  true,
		SupportsCodeAnalysis:    true,
		SupportsRefactoring:     true,
		Limits: models.ModelLimits{
			MaxTokens:             16384,
			MaxInputLength:        200000,
			MaxOutputLength:       16384,
			MaxConcurrentRequests: 10,
		},
		Metadata: map[string]string{
			"provider":     "OpenCode Zen",
			"model_family": "Mixed",
			"api_version":  "v1",
			"note":         i18n.Tr(context.Background(), "provider.zen.description", nil),
			"free_tier":    "true",
			// The endpoint actually in force, not the compiled constant.
			// Reporting ZenAPIURL here made this metadata LIE the moment
			// LLMPROVIDER_ZEN_API_BASE_URL or an explicit baseURL argument was
			// used — a diagnostic that names the wrong host is worse than none.
			"base_url": p.baseURL,
		},
	}
}

// ValidateConfig validates the provider configuration
func (p *ZenProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string

	// API key is only required for non-free models
	// Free models can work in anonymous mode without API key
	if p.apiKey == "" && !p.anonymousMode {
		// Check if the model is a free model - if so, anonymous mode should be enabled
		if !isFreeModel(p.model) {
			errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_opencode_api_key_required", nil))
		}
	}

	if p.baseURL == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_base_url_required", nil))
	}

	if p.model == "" {
		errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_model_required", nil))
	}

	return len(errors) == 0, errors
}

// IsAnonymousMode returns whether the provider is in anonymous mode
func (p *ZenProvider) IsAnonymousMode() bool {
	return p.anonymousMode
}

// HealthCheck implements health checking for the Zen provider
func (p *ZenProvider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Health check - try to get models list
	req, err := http.NewRequestWithContext(ctx, "GET", ZenModelsURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if p.anonymousMode {
		req.Header.Set(AnonymousDeviceHeader, p.deviceID)
	} else if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}

	return nil
}

// GetModel returns the current model
func (p *ZenProvider) GetModel() string {
	return p.model
}

// SetModel sets the model to use
func (p *ZenProvider) SetModel(model string) {
	p.model = model
}

// GetAvailableModels fetches the list of available models from Zen API
func (p *ZenProvider) GetAvailableModels(ctx context.Context) ([]ZenModelInfo, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", ZenModelsURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if p.anonymousMode {
		req.Header.Set(AnonymousDeviceHeader, p.deviceID)
	} else if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var modelsResp ZenModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelsResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return modelsResp.Data, nil
}

// GetFreeModels returns only the free models from the available models list
func (p *ZenProvider) GetFreeModels(ctx context.Context) ([]ZenModelInfo, error) {
	allModels, err := p.GetAvailableModels(ctx)
	if err != nil {
		return nil, err
	}

	freeModelIDs := FreeModels()
	freeModels := make([]ZenModelInfo, 0)

	for _, model := range allModels {
		for _, freeID := range freeModelIDs {
			if model.ID == freeID || strings.HasSuffix(freeID, model.ID) {
				freeModels = append(freeModels, model)
				break
			}
		}
	}

	return freeModels, nil
}

// IsOpenCodeInstalled reports whether the `opencode` CLI is present and
// executable on PATH.
//
// It used to be a hardcoded `return false` with a comment claiming the CLI was
// "not available in standalone module". That was not true of the module: this
// package already answers exactly this question correctly, in zen_http.go, and
// USES the CLI (StartServer execs it). What the stub did have was ZERO
// production callers -- its only three call sites were test skip-guards, which
// it therefore held permanently dark, including on hosts where opencode IS
// installed.
//
// It delegates rather than repeating the probe, so the two entry points to the
// same question cannot drift apart.
func IsOpenCodeInstalled() bool { return IsZenHTTPAvailable() }
