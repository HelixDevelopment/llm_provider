package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"digital.vasic.llmprovider/pkg/i18n"
	"digital.vasic.llmprovider/pkg/models"
	"digital.vasic.llmprovider/pkg/settings"
)

// Settings keys for this adapter (one Go package, two files: gemini.go carries
// the unified provider, gemini_api.go the HTTP transport). They share
// LLMPROVIDER_GEMINI_{MODEL,BASE_URL,TIMEOUT}, because they are one backend
// reached one way. The models-listing endpoint is keyed separately: it is a
// different URL with a different shape, and folding it into BASE_URL would mean
// an operator pointing the adapter at a proxy silently broke health checks.
const (
	// SettingsProvider keys the endpoint, model and timeout:
	// LLMPROVIDER_GEMINI_BASE_URL, LLMPROVIDER_GEMINI_MODEL,
	// LLMPROVIDER_GEMINI_TIMEOUT.
	SettingsProvider = "gemini"
	// SettingsProviderModels keys the models-listing endpoint used by
	// HealthCheck: LLMPROVIDER_GEMINI_MODELS_BASE_URL.
	SettingsProviderModels = "gemini_models"
)

const (
	// GeminiAPIURL is the base URL for the Gemini API generateContent endpoint.
	GeminiAPIURL = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent"
	// GeminiStreamAPIURL is the base URL for the Gemini API streaming endpoint.
	GeminiStreamAPIURL = "https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent"
	// GeminiModelsURL is the models-listing endpoint HealthCheck probes.
	GeminiModelsURL = "https://generativelanguage.googleapis.com/v1beta/models"
	// GeminiModel is the legacy default model (kept for backward compatibility).
	GeminiModel = "gemini-2.0-flash"
)

// DefaultUnifiedTimeout is the compiled fallback for the unified provider,
// which may drive a CLI or an ACP sub-provider and so waits longer than the
// plain HTTP transport in gemini_api.go. Override with
// LLMPROVIDER_GEMINI_TIMEOUT.
const DefaultUnifiedTimeout = 180 * time.Second

// ---------------------------------------------------------------------------
// Shared types — used by all sub-providers (API, CLI, ACP)
// ---------------------------------------------------------------------------

// RetryConfig defines retry behavior for API calls.
type RetryConfig struct {
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

// DefaultRetryConfig returns sensible defaults for Gemini API retry behavior.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

// GeminiRequest represents a request to the Gemini API (legacy format).
type GeminiRequest struct {
	Contents         []GeminiContent        `json:"contents"`
	GenerationConfig GeminiGenerationConfig `json:"generationConfig,omitempty"`
	SafetySettings   []GeminiSafetySetting  `json:"safetySettings,omitempty"`
	Tools            []GeminiToolDef        `json:"tools,omitempty"`
	ToolConfig       *GeminiToolConfig      `json:"toolConfig,omitempty"`
}

// GeminiToolDef represents a tool definition for the Gemini API.
type GeminiToolDef struct {
	FunctionDeclarations []GeminiFunctionDeclaration `json:"functionDeclarations,omitempty"`
}

// GeminiFunctionDeclaration represents a function declaration.
type GeminiFunctionDeclaration struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// GeminiToolConfig configures tool behavior.
type GeminiToolConfig struct {
	FunctionCallingConfig *GeminiFunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

// GeminiFunctionCallingConfig configures function calling.
type GeminiFunctionCallingConfig struct {
	Mode string `json:"mode,omitempty"` // AUTO, NONE, ANY
}

// GeminiContent represents a content block in the Gemini API.
type GeminiContent struct {
	Parts []GeminiPart `json:"parts"`
	Role  string       `json:"role,omitempty"`
}

// GeminiPart represents a part of a content block.
type GeminiPart struct {
	Text         string            `json:"text,omitempty"`
	InlineData   *GeminiInlineData `json:"inlineData,omitempty"`
	FunctionCall map[string]any    `json:"functionCall,omitempty"`
	Thought      bool              `json:"thought,omitempty"`
}

// GeminiInlineData represents inline binary data.
type GeminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

// GeminiGenerationConfig holds generation parameters.
type GeminiGenerationConfig struct {
	Temperature     float64  `json:"temperature,omitempty"`
	TopP            float64  `json:"topP,omitempty"`
	TopK            int      `json:"topK,omitempty"`
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

// GeminiSafetySetting configures content safety thresholds.
type GeminiSafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// GeminiResponse represents the Gemini API response.
type GeminiResponse struct {
	Candidates     []GeminiCandidate     `json:"candidates"`
	PromptFeedback *GeminiPromptFeedback `json:"promptFeedback,omitempty"`
	UsageMetadata  *GeminiUsageMetadata  `json:"usageMetadata,omitempty"`
}

// GeminiCandidate represents a response candidate.
type GeminiCandidate struct {
	Content       GeminiContent        `json:"content"`
	FinishReason  string               `json:"finishReason"`
	Index         int                  `json:"index"`
	SafetyRatings []GeminiSafetyRating `json:"safetyRatings,omitempty"`
}

// GeminiPromptFeedback holds prompt-level feedback.
type GeminiPromptFeedback struct {
	BlockReason   string               `json:"blockReason"`
	SafetyRatings []GeminiSafetyRating `json:"safetyRatings,omitempty"`
}

// GeminiSafetyRating represents a safety rating.
type GeminiSafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
	Blocked     bool   `json:"blocked"`
}

// GeminiUsageMetadata holds token usage information.
type GeminiUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// GeminiStreamResponse represents a streaming response chunk.
type GeminiStreamResponse struct {
	Candidates    []GeminiCandidate    `json:"candidates,omitempty"`
	UsageMetadata *GeminiUsageMetadata `json:"usageMetadata,omitempty"`
}

// isRetryableStatus returns true for HTTP status codes that warrant a retry.
func isRetryableStatus(statusCode int) bool {
	switch statusCode {
	case 429, 500, 502, 503, 504:
		return true
	default:
		return false
	}
}

// isAuthRetryableStatus returns true for auth errors that may be transient.
func isAuthRetryableStatus(statusCode int) bool {
	return statusCode == 401
}

// ---------------------------------------------------------------------------
// Unified Gemini Provider — orchestrates API, CLI, and ACP sub-providers
// ---------------------------------------------------------------------------

// GeminiUnifiedProvider is the unified Gemini provider with multiple access
// methods:
//  1. Primary: Direct Gemini API with API key
//  2. Fallback 1: Gemini CLI headless mode
//  3. Fallback 2: Gemini ACP (Agent Communication Protocol)
//
// Access method selection is automatic based on available credentials and
// installed tooling.
type GeminiUnifiedProvider struct {
	model     string
	timeout   time.Duration
	maxTokens int
	apiKey    string

	apiProvider *GeminiAPIProvider
	cliProvider *GeminiCLIProvider
	acpProvider *GeminiACPProvider

	initialized     bool
	initOnce        sync.Once
	initErr         error
	preferredMethod string
}

// GeminiUnifiedConfig holds configuration for the unified Gemini provider.
type GeminiUnifiedConfig struct {
	Model           string
	Timeout         time.Duration
	MaxTokens       int
	APIKey          string
	BaseURL         string
	PreferredMethod string // "auto", "api", "cli", "acp"
}

// DefaultGeminiUnifiedConfig returns default configuration reading from env.
func DefaultGeminiUnifiedConfig() GeminiUnifiedConfig {
	return GeminiUnifiedConfig{
		// LLMPROVIDER_GEMINI_MODEL / LLMPROVIDER_GEMINI_TIMEOUT.
		Model:           settings.Model(SettingsProvider, GeminiDefaultModel),
		Timeout:         settings.Timeout(SettingsProvider, DefaultUnifiedTimeout),
		MaxTokens:       8192,
		APIKey:          os.Getenv("GEMINI_API_KEY"),
		PreferredMethod: "auto",
	}
}

// NewGeminiUnifiedProvider creates a new unified Gemini provider.
func NewGeminiUnifiedProvider(config GeminiUnifiedConfig) *GeminiUnifiedProvider {
	if config.Timeout == 0 {
		// LLMPROVIDER_GEMINI_TIMEOUT.
		config.Timeout = settings.Timeout(SettingsProvider, DefaultUnifiedTimeout)
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = 8192
	}
	if config.APIKey == "" {
		config.APIKey = os.Getenv("GEMINI_API_KEY")
	}
	if config.PreferredMethod == "" {
		config.PreferredMethod = "auto"
	}
	if config.Model == "" {
		// LLMPROVIDER_GEMINI_MODEL.
		config.Model = settings.Model(SettingsProvider, GeminiDefaultModel)
	}

	p := &GeminiUnifiedProvider{
		model:           config.Model,
		timeout:         config.Timeout,
		maxTokens:       config.MaxTokens,
		apiKey:          config.APIKey,
		preferredMethod: config.PreferredMethod,
	}

	// Eagerly create API provider if we have an API key
	if config.APIKey != "" {
		p.apiProvider = NewGeminiAPIProvider(config.APIKey, config.BaseURL, config.Model)
	}

	return p
}

// NewGeminiProvider creates a backward-compatible Gemini provider.
// It returns a GeminiUnifiedProvider that delegates to the API sub-provider
// when an API key is provided, with CLI and ACP as fallbacks.
func NewGeminiProvider(apiKey, baseURL, model string) *GeminiUnifiedProvider {
	config := GeminiUnifiedConfig{
		APIKey: apiKey,
		// Timeout is left at its ZERO value on purpose. Writing the compiled
		// 180s in here handed NewGeminiUnifiedProvider a non-zero timeout, so
		// its own `config.Timeout == 0` check never fired and
		// LLMPROVIDER_GEMINI_TIMEOUT was honoured on every construction path
		// except the two backward-compatible ones — the frozen value winning
		// silently. Leaving it zero routes this path through the same
		// resolution as every other.
		BaseURL:         baseURL,
		Model:           model,
		MaxTokens:       8192,
		PreferredMethod: "auto",
	}
	return NewGeminiUnifiedProvider(config)
}

// NewGeminiProviderWithRetry creates a backward-compatible provider with
// custom retry configuration.
func NewGeminiProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *GeminiUnifiedProvider {
	config := GeminiUnifiedConfig{
		APIKey: apiKey,
		// Timeout deliberately left zero — see NewGeminiProvider.
		BaseURL:         baseURL,
		Model:           model,
		MaxTokens:       8192,
		PreferredMethod: "auto",
	}
	p := NewGeminiUnifiedProvider(config)
	if p.apiProvider != nil {
		p.apiProvider.retryConfig = retryConfig
	}
	return p
}

// Initialize lazily initializes all sub-providers.
func (p *GeminiUnifiedProvider) Initialize() error {
	p.initOnce.Do(func() {
		p.initErr = p.initialize()
	})
	return p.initErr
}

func (p *GeminiUnifiedProvider) initialize() error {
	// API provider already created in constructor if key was present

	// Create CLI provider
	cliConfig := GeminiCLIConfig{
		Model:           p.model,
		Timeout:         p.timeout,
		MaxOutputTokens: p.maxTokens,
		APIKey:          p.apiKey,
	}
	p.cliProvider = NewGeminiCLIProvider(cliConfig)

	// Create ACP provider
	acpConfig := GeminiACPConfig{
		Model:     p.model,
		Timeout:   p.timeout,
		MaxTokens: p.maxTokens,
		APIKey:    p.apiKey,
	}
	p.acpProvider = NewGeminiACPProvider(acpConfig)

	p.initialized = true
	return nil
}

// Complete implements the LLMProvider interface with auto-detect fallback.
func (p *GeminiUnifiedProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	if err := p.Initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize Gemini: %w", err)
	}

	var lastErr error

	switch p.preferredMethod {
	case "api":
		return p.completeWithAPI(ctx, req)
	case "cli":
		return p.completeWithCLI(ctx, req)
	case "acp":
		return p.completeWithACP(ctx, req)
	default: // "auto"
		// Try API first (fastest, most reliable)
		if p.apiProvider != nil {
			if resp, err := p.completeWithAPI(ctx, req); err == nil {
				return resp, nil
			} else {
				lastErr = err
			}
		}

		// Try CLI fallback
		if p.cliProvider.IsCLIAvailable() {
			if resp, err := p.completeWithCLI(ctx, req); err == nil {
				return resp, nil
			} else {
				lastErr = err
			}
		}

		// Try ACP fallback
		if p.acpProvider.IsAvailable() {
			if resp, err := p.completeWithACP(ctx, req); err == nil {
				return resp, nil
			} else {
				lastErr = err
			}
		}

		if lastErr != nil {
			return nil, fmt.Errorf("all Gemini access methods failed: %w", lastErr)
		}
		return nil, fmt.Errorf("no Gemini access method available (no API key, CLI not installed, ACP unavailable)")
	}
}

func (p *GeminiUnifiedProvider) completeWithAPI(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	if p.apiProvider == nil {
		return nil, fmt.Errorf("Gemini API provider not configured (no API key)")
	}
	return p.apiProvider.Complete(ctx, req)
}

func (p *GeminiUnifiedProvider) completeWithCLI(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	if !p.cliProvider.IsCLIAvailable() {
		return nil, fmt.Errorf("Gemini CLI not available: %v", p.cliProvider.GetCLIError())
	}
	return p.cliProvider.Complete(ctx, req)
}

func (p *GeminiUnifiedProvider) completeWithACP(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	if !p.acpProvider.IsAvailable() {
		return nil, fmt.Errorf("Gemini ACP not available")
	}
	return p.acpProvider.Complete(ctx, req)
}

// CompleteStream implements streaming completion with auto-detect fallback.
func (p *GeminiUnifiedProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	if err := p.Initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize Gemini: %w", err)
	}

	switch p.preferredMethod {
	case "api":
		if p.apiProvider != nil {
			return p.apiProvider.CompleteStream(ctx, req)
		}
	case "cli":
		return p.cliProvider.CompleteStream(ctx, req)
	case "acp":
		return p.acpProvider.CompleteStream(ctx, req)
	}

	// Auto mode: try in order
	if p.apiProvider != nil {
		return p.apiProvider.CompleteStream(ctx, req)
	}
	if p.cliProvider.IsCLIAvailable() {
		return p.cliProvider.CompleteStream(ctx, req)
	}
	return p.acpProvider.CompleteStream(ctx, req)
}

// HealthCheck checks provider health using the best available method.
func (p *GeminiUnifiedProvider) HealthCheck() error {
	if err := p.Initialize(); err != nil {
		return err
	}

	// Prefer API health check (lightweight — just lists models)
	if p.apiProvider != nil {
		if err := p.apiProvider.HealthCheck(); err == nil {
			return nil
		}
	}

	// Try CLI
	if p.cliProvider.IsCLIAvailable() {
		return nil // CLI availability itself is the health check
	}

	// Try ACP
	if p.acpProvider.IsAvailable() {
		return nil
	}

	return fmt.Errorf("no Gemini access method available for health check")
}

// GetCapabilities returns merged capabilities from all available sub-providers.
func (p *GeminiUnifiedProvider) GetCapabilities() *models.ProviderCapabilities {
	caps := &models.ProviderCapabilities{
		SupportedModels: getAllGeminiModels(),
		SupportedFeatures: []string{
			"text_completion",
			"chat",
			"function_calling",
			"streaming",
			"vision",
			"extended_thinking",
			"google_search_grounding",
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
			MaxTokens:             65536,
			MaxInputLength:        1048576,
			MaxOutputLength:       65536,
			MaxConcurrentRequests: 10,
		},
		Metadata: map[string]string{
			"provider":       "Google",
			"model_family":   "Gemini",
			"api_version":    "v1beta",
			"access_methods": "api,cli,acp",
		},
	}

	return caps
}

// ValidateConfig validates the provider configuration.
func (p *GeminiUnifiedProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var issues []string

	if p.apiKey == "" && !IsGeminiCLIInstalled() {
		issues = append(issues, "no GEMINI_API_KEY and Gemini CLI not installed")
	}

	if len(issues) > 0 {
		return false, issues
	}
	return true, nil
}

// GetName returns the provider name.
func (p *GeminiUnifiedProvider) GetName() string {
	return "gemini"
}

// GetProviderType returns the provider type.
func (p *GeminiUnifiedProvider) GetProviderType() string {
	return "gemini"
}

// GetCurrentModel returns the current model.
func (p *GeminiUnifiedProvider) GetCurrentModel() string {
	return p.model
}

// SetModel sets the model across all sub-providers.
func (p *GeminiUnifiedProvider) SetModel(model string) {
	p.model = model
	if p.cliProvider != nil {
		p.cliProvider.SetModel(model)
	}
	if p.acpProvider != nil {
		p.acpProvider.SetModel(model)
	}
}

// GetPreferredMethod returns the preferred access method.
func (p *GeminiUnifiedProvider) GetPreferredMethod() string {
	return p.preferredMethod
}

// SetPreferredMethod sets the preferred access method.
func (p *GeminiUnifiedProvider) SetPreferredMethod(method string) {
	p.preferredMethod = method
}

// GetAvailableAccessMethods returns all currently available access methods.
func (p *GeminiUnifiedProvider) GetAvailableAccessMethods() []string {
	_ = p.Initialize()

	var methods []string

	if p.apiProvider != nil {
		methods = append(methods, "api")
	}

	if p.cliProvider != nil && p.cliProvider.IsCLIAvailable() {
		methods = append(methods, "cli")
	}

	if p.acpProvider != nil && p.acpProvider.IsAvailable() {
		methods = append(methods, "acp")
	}

	return methods
}

// getAllGeminiModels returns all known Gemini models.
func getAllGeminiModels() []string {
	return []string{
		"gemini-3.1-pro-preview",
		"gemini-3-pro-preview",
		"gemini-3-flash-preview",
		"gemini-2.5-pro",
		"gemini-2.5-flash",
		"gemini-2.5-flash-lite",
		"gemini-2.0-flash",
		"gemini-embedding-001",
	}
}

// IsGeminiAvailable checks if any Gemini access method is available.
func IsGeminiAvailable() bool {
	if os.Getenv("GEMINI_API_KEY") != "" {
		return true
	}
	return IsGeminiCLIInstalled() && IsGeminiCLIAuthenticated()
}

// CanUseGemini returns true if Gemini can be used via any method.
func CanUseGemini() bool {
	return IsGeminiAvailable() || CanUseGeminiCLI() || CanUseGeminiACP()
}

// GetGeminiProviderInfo returns provider information for the registry.
func GetGeminiProviderInfo() map[string]interface{} {
	return map[string]interface{}{
		"id":           "gemini",
		"name":         "Gemini (Google)",
		"type":         "gemini",
		"display_name": "Google Gemini",
		"description":  i18n.Tr(context.Background(), "provider.gemini.description", nil),
		"auth_type":    "api_key",
		"access_methods": []string{
			"api",
			"cli",
			"acp",
		},
		"models":             getAllGeminiModels(),
		"supports_streaming": true,
		"supports_tools":     true,
		"supports_vision":    true,
		"supports_search":    true,
		"supports_reasoning": true,
		"tier":               1,
		"priority":           1,
	}
}

// buildPromptFromRequest extracts a text prompt from an LLMRequest.
// Shared helper used by CLI and ACP sub-providers.
func buildPromptFromRequest(req *models.LLMRequest) string {
	var b []byte

	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			b = append(b, "System: "...)
			b = append(b, msg.Content...)
			b = append(b, "\n\n"...)
		case "user":
			b = append(b, msg.Content...)
			b = append(b, '\n')
		case "assistant":
			b = append(b, "Assistant: "...)
			b = append(b, msg.Content...)
			b = append(b, "\n\n"...)
		}
	}

	prompt := string(b)
	if prompt == "" && req.Prompt != "" {
		prompt = req.Prompt
	}

	return prompt
}

// marshalJSON is a convenience wrapper that ignores marshal errors for metadata.
func marshalJSON(v interface{}) string {
	data, _ := json.Marshal(v) //nolint:errcheck
	return string(data)
}
