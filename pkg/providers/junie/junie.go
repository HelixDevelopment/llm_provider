package junie

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"digital.vasic.llmprovider/pkg/i18n"
	"digital.vasic.llmprovider/pkg/models"
	"digital.vasic.llmprovider/pkg/settings"
)

// DefaultJunieModel is the compiled fallback model. Override with
// LLMPROVIDER_JUNIE_MODEL (see pkg/settings).
const DefaultJunieModel = "sonnet"

// DefaultJunieTimeout is the compiled fallback wait. Junie drives a headless
// CLI or an ACP session rather than a single HTTPS round trip, which is why it
// is longer than a plain adapter's. Override with LLMPROVIDER_JUNIE_TIMEOUT.
const DefaultJunieTimeout = 180 * time.Second

// JunieProvider is the unified Junie provider with multiple access methods:
// 1. Primary: Headless CLI mode with Junie API key
// 2. Fallback 1: ACP mode for IDE-like integration
// 3. BYOK: Direct provider API keys (Anthropic, OpenAI, Google, xAI)
//
// Junie supports BYOK (Bring Your Own Key) for multiple LLM providers,
// making it a unified gateway to various models.
type JunieProvider struct {
	model     string
	timeout   time.Duration
	maxTokens int
	apiKey    string

	cliProvider     *JunieCLIProvider
	acpProvider     *JunieACPProvider
	byokProviders   map[string]bool
	byokCredentials map[string]string

	initialized     bool
	initOnce        sync.Once
	initErr         error
	preferredMethod string
}

// JunieConfig holds configuration for the Junie provider
type JunieConfig struct {
	Model           string
	Timeout         time.Duration
	MaxTokens       int
	APIKey          string
	PreferredMethod string
}

// DefaultJunieConfig returns default configuration
func DefaultJunieConfig() JunieConfig {
	return JunieConfig{
		// LLMPROVIDER_JUNIE_MODEL / LLMPROVIDER_JUNIE_TIMEOUT.
		Model:           settings.Model("junie", DefaultJunieModel),
		Timeout:         settings.Timeout("junie", DefaultJunieTimeout),
		MaxTokens:       8192,
		APIKey:          os.Getenv("JUNIE_API_KEY"),
		PreferredMethod: "auto",
	}
}

// NewJunieProvider creates a new unified Junie provider
func NewJunieProvider(config JunieConfig) *JunieProvider {
	if config.Timeout == 0 {
		// LLMPROVIDER_JUNIE_TIMEOUT. This check is reached by any caller that
		// builds a JunieConfig itself rather than starting from
		// DefaultJunieConfig, so it must resolve too — resolving in only one
		// of the two places is how an override comes to work on some
		// construction paths and not others.
		config.Timeout = settings.Timeout("junie", DefaultJunieTimeout)
	}
	if config.Model == "" {
		// LLMPROVIDER_JUNIE_MODEL. There was no empty-check here at all: a
		// caller-built config with no Model produced a provider with an EMPTY
		// model string, which is a worse failure than a frozen one.
		config.Model = settings.Model("junie", DefaultJunieModel)
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = 8192
	}
	if config.APIKey == "" {
		config.APIKey = os.Getenv("JUNIE_API_KEY")
	}
	if config.PreferredMethod == "" {
		config.PreferredMethod = "auto"
	}

	p := &JunieProvider{
		model:           config.Model,
		timeout:         config.Timeout,
		maxTokens:       config.MaxTokens,
		apiKey:          config.APIKey,
		preferredMethod: config.PreferredMethod,
		byokProviders:   make(map[string]bool),
		byokCredentials: make(map[string]string),
	}

	p.discoverBYOKCredentials()

	return p
}

// NewJunieProviderWithModel creates a provider with a specific model
func NewJunieProviderWithModel(model string) *JunieProvider {
	config := DefaultJunieConfig()
	config.Model = model
	return NewJunieProvider(config)
}

// discoverBYOKCredentials discovers BYOK credentials from environment
func (p *JunieProvider) discoverBYOKCredentials() {
	byokEnvVars := map[string][]string{
		"anthropic":  {"JUNIE_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"},
		"openai":     {"JUNIE_OPENAI_API_KEY", "OPENAI_API_KEY"},
		"google":     {"JUNIE_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"},
		"grok":       {"JUNIE_GROK_API_KEY", "XAI_API_KEY", "GROK_API_KEY"},
		"openrouter": {"JUNIE_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"},
	}

	for provider, envVars := range byokEnvVars {
		for _, envVar := range envVars {
			if key := os.Getenv(envVar); key != "" {
				p.byokProviders[provider] = true
				p.byokCredentials[provider] = key
				break
			}
		}
	}
}

// Initialize initializes the provider
func (p *JunieProvider) Initialize() error {
	p.initOnce.Do(func() {
		p.initErr = p.initialize()
	})
	return p.initErr
}

func (p *JunieProvider) initialize() error {
	cliConfig := JunieCLIConfig{
		Model:           p.model,
		Timeout:         p.timeout,
		MaxOutputTokens: p.maxTokens,
		APIKey:          p.apiKey,
	}
	p.cliProvider = NewJunieCLIProvider(cliConfig)

	acpConfig := JunieACPConfig{
		Model:     p.model,
		Timeout:   p.timeout,
		MaxTokens: p.maxTokens,
		APIKey:    p.apiKey,
	}
	p.acpProvider = NewJunieACPProvider(acpConfig)

	p.initialized = true
	return nil
}

// Complete implements the LLMProvider interface
func (p *JunieProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	if err := p.Initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize Junie: %w", err)
	}

	var lastErr error

	switch p.preferredMethod {
	case "cli":
		return p.completeWithCLI(ctx, req)
	case "acp":
		return p.completeWithACP(ctx, req)
	default:
		if resp, err := p.completeWithCLI(ctx, req); err == nil {
			return resp, nil
		} else {
			lastErr = err
		}

		if resp, err := p.completeWithACP(ctx, req); err == nil {
			return resp, nil
		} else {
			lastErr = err
		}

		return nil, fmt.Errorf("all Junie access methods failed: %w", lastErr)
	}
}

// completeWithCLI completes using CLI mode
func (p *JunieProvider) completeWithCLI(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	if !p.cliProvider.IsCLIAvailable() {
		return nil, fmt.Errorf("Junie CLI not available: %v", p.cliProvider.GetCLIError())
	}

	return p.cliProvider.Complete(ctx, req)
}

// completeWithACP completes using ACP mode
func (p *JunieProvider) completeWithACP(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	if !p.acpProvider.IsAvailable() {
		return nil, fmt.Errorf("Junie ACP not available")
	}

	return p.acpProvider.Complete(ctx, req)
}

// CompleteStream implements streaming completion
func (p *JunieProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	if err := p.Initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize Junie: %w", err)
	}

	switch p.preferredMethod {
	case "cli":
		return p.cliProvider.CompleteStream(ctx, req)
	case "acp":
		return p.acpProvider.CompleteStream(ctx, req)
	default:
		if p.cliProvider.IsCLIAvailable() {
			return p.cliProvider.CompleteStream(ctx, req)
		}
		return p.acpProvider.CompleteStream(ctx, req)
	}
}

// HealthCheck checks provider health
func (p *JunieProvider) HealthCheck() error {
	if err := p.Initialize(); err != nil {
		return err
	}

	if p.cliProvider.IsCLIAvailable() {
		return p.cliProvider.HealthCheck()
	}

	if p.acpProvider.IsAvailable() {
		return p.acpProvider.HealthCheck()
	}

	return fmt.Errorf("no Junie access method available")
}

// GetCapabilities returns provider capabilities
func (p *JunieProvider) GetCapabilities() *models.ProviderCapabilities {
	capabilities := &models.ProviderCapabilities{
		SupportedModels:   getAllJunieModels(),
		SupportsStreaming: true,
		SupportsTools:     true,
		Limits: models.ModelLimits{
			MaxTokens:             p.maxTokens,
			MaxConcurrentRequests: 1,
		},
	}

	return capabilities
}

// ValidateConfig validates the configuration
func (p *JunieProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var issues []string

	if p.apiKey == "" && !IsJunieAuthenticated() {
		if len(p.byokProviders) == 0 {
			issues = append(issues, "no JUNIE_API_KEY or BYOK credentials found")
		}
	}

	if !IsJunieInstalled() {
		issues = append(issues, "junie command not installed")
	}

	if len(issues) > 0 {
		return false, issues
	}
	return true, nil
}

// GetName returns the provider name
func (p *JunieProvider) GetName() string {
	return "junie"
}

// GetProviderType returns the provider type
func (p *JunieProvider) GetProviderType() string {
	return "junie"
}

// GetCurrentModel returns the current model
func (p *JunieProvider) GetCurrentModel() string {
	return p.model
}

// SetModel sets the model to use
func (p *JunieProvider) SetModel(model string) {
	p.model = model
	if p.cliProvider != nil {
		p.cliProvider.SetModel(model)
	}
	if p.acpProvider != nil {
		p.acpProvider.SetModel(model)
	}
}

// GetBYOKProviders returns available BYOK providers
func (p *JunieProvider) GetBYOKProviders() []string {
	var providers []string
	for provider := range p.byokProviders {
		providers = append(providers, provider)
	}
	return providers
}

// HasBYOKProvider checks if a BYOK provider is configured
func (p *JunieProvider) HasBYOKProvider(provider string) bool {
	return p.byokProviders[provider]
}

// GetPreferredMethod returns the preferred access method
func (p *JunieProvider) GetPreferredMethod() string {
	return p.preferredMethod
}

// SetPreferredMethod sets the preferred access method
func (p *JunieProvider) SetPreferredMethod(method string) {
	p.preferredMethod = method
}

// getAllJunieModels returns all supported Junie models
func getAllJunieModels() []string {
	models := make([]string, len(knownJunieModels))
	copy(models, knownJunieModels)

	models = append(models, byokModels...)

	return models
}

// GetAvailableAccessMethods returns available access methods
func (p *JunieProvider) GetAvailableAccessMethods() []string {
	var methods []string

	if p.cliProvider != nil && p.cliProvider.IsCLIAvailable() {
		methods = append(methods, "cli")
	}

	if p.acpProvider != nil && p.acpProvider.IsAvailable() {
		methods = append(methods, "acp")
	}

	if len(p.byokProviders) > 0 {
		methods = append(methods, "byok")
	}

	return methods
}

// IsJunieAvailable checks if any Junie access method is available
func IsJunieAvailable() bool {
	if os.Getenv("JUNIE_API_KEY") != "" {
		return IsJunieInstalled()
	}
	return IsJunieInstalled() && IsJunieAuthenticated()
}

// CanUseJunie returns true if Junie can be used
func CanUseJunie() bool {
	return IsJunieAvailable() || CanUseJunieCLI() || CanUseJunieACP()
}

// GetJunieProviderInfo returns provider information for registry
func GetJunieProviderInfo() map[string]interface{} {
	return map[string]interface{}{
		"id":           "junie",
		"name":         "Junie (JetBrains)",
		"type":         "junie",
		"display_name": i18n.Tr(context.Background(), "provider.junie.display_name", nil),
		"description":  i18n.Tr(context.Background(), "provider.junie.description", nil),
		"auth_type":    "api_key",
		"access_methods": []string{
			"cli",
			"acp",
			"byok",
		},
		"byok_providers": []string{
			"anthropic",
			"openai",
			"google",
			"grok",
			"openrouter",
		},
		"models":             getAllJunieModels(),
		"supports_streaming": true,
		"supports_tools":     true,
		"tier":               2,
		"priority":           2,
	}
}
