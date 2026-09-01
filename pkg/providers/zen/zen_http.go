package zen

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"digital.vasic.llmprovider/pkg/models"
	"digital.vasic.llmprovider/pkg/settings"
)

// ZenHTTPProvider implements the LLMProvider interface using OpenCode's HTTP server
//
// This provides a full REST API integration:
// - Session management (conversation history)
// - Streaming responses via SSE
// - File operations
// - Multiple concurrent requests
//
// Start with: opencode serve --port 4096
type ZenHTTPProvider struct {
	baseURL    string
	username   string
	password   string
	model      string
	timeout    time.Duration
	maxTokens  int
	httpClient *http.Client
	sessionID  string

	// Server management
	serverCmd     *exec.Cmd
	serverStarted bool
	serverMu      sync.Mutex
	sessionMu     sync.Mutex
	autoStart     bool
}

// Defaults for the Zen HTTP provider. Each is a starting point that the
// environment can override through pkg/settings — LLMPROVIDER_ZEN_BASE_URL,
// LLMPROVIDER_ZEN_MODEL, LLMPROVIDER_ZEN_TIMEOUT — because an opencode server
// on another host or port is an operator's routine change, not a fork of this
// library. OPENCODE_BASE_URL is accepted as an alias for the endpoint, matching
// the OPENCODE_SERVER_PASSWORD variable this file already reads two fields away.
// DefaultZenModel is declared in zen.go (= ModelBigPickle) and is used here
// as this provider's model fallback; it is not redeclared.
const (
	DefaultZenBaseURL   = "http://localhost:4096"
	DefaultZenUsername  = "opencode"
	DefaultZenTimeout   = 180 * time.Second
	DefaultZenMaxTokens = 8192

	// EnvZenBaseURLAlias is opencode's own endpoint variable.
	EnvZenBaseURLAlias = "OPENCODE_BASE_URL"
)

// ZenHTTPConfig holds configuration for the HTTP provider
type ZenHTTPConfig struct {
	BaseURL   string // e.g., DefaultZenBaseURL; see LLMPROVIDER_ZEN_BASE_URL
	Username  string // Default: DefaultZenUsername
	Password  string // From OPENCODE_SERVER_PASSWORD
	Model     string
	Timeout   time.Duration
	MaxTokens int
	AutoStart bool // Auto-start server if not running
}

// DefaultZenHTTPConfig returns default configuration
func DefaultZenHTTPConfig() ZenHTTPConfig {
	password := os.Getenv("OPENCODE_SERVER_PASSWORD")
	return ZenHTTPConfig{
		BaseURL:   settings.BaseURL("zen", DefaultZenBaseURL, EnvZenBaseURLAlias),
		Username:  DefaultZenUsername,
		Password:  password,
		Model:     settings.Model("zen", DefaultZenModel),
		Timeout:   settings.Timeout("zen", DefaultZenTimeout),
		MaxTokens: DefaultZenMaxTokens,
		AutoStart: true,
	}
}

// NewZenHTTPProvider creates a new Zen HTTP provider
func NewZenHTTPProvider(config ZenHTTPConfig) *ZenHTTPProvider {
	if config.BaseURL == "" {
		config.BaseURL = settings.BaseURL("zen", DefaultZenBaseURL, EnvZenBaseURLAlias)
	}
	if config.Username == "" {
		config.Username = DefaultZenUsername
	}
	if config.Model == "" {
		config.Model = settings.Model("zen", DefaultZenModel)
	}
	if config.Timeout == 0 {
		config.Timeout = settings.Timeout("zen", DefaultZenTimeout)
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = DefaultZenMaxTokens
	}

	return &ZenHTTPProvider{
		baseURL:   strings.TrimSuffix(config.BaseURL, "/"),
		username:  config.Username,
		password:  config.Password,
		model:     config.Model,
		timeout:   config.Timeout,
		maxTokens: config.MaxTokens,
		autoStart: config.AutoStart,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

// NewZenHTTPProviderWithModel creates an HTTP provider with a specific model
func NewZenHTTPProviderWithModel(model string) *ZenHTTPProvider {
	config := DefaultZenHTTPConfig()
	config.Model = model
	return NewZenHTTPProvider(config)
}

// API Response Types
type sessionResponse struct {
	ID        string    `json:"id"`
	Title     string    `json:"title"`
	CreatedAt time.Time `json:"createdAt"`
	UpdatedAt time.Time `json:"updatedAt"`
}

type messageRequest struct {
	Content string `json:"content"`
	Model   string `json:"model,omitempty"`
}

type messageResponse struct {
	ID        string `json:"id"`
	Role      string `json:"role"`
	Content   string `json:"content"`
	Model     string `json:"model"`
	CreatedAt string `json:"createdAt"`
}

type errorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

// IsServerRunning checks if the OpenCode server is running
func (p *ZenHTTPProvider) IsServerRunning() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", p.baseURL+"/health", nil)
	if err != nil {
		return false
	}

	if p.password != "" {
		req.SetBasicAuth(p.username, p.password)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()

	return resp.StatusCode == http.StatusOK
}

// StartServer starts the OpenCode server if not running
func (p *ZenHTTPProvider) StartServer() error {
	p.serverMu.Lock()
	defer p.serverMu.Unlock()

	if p.serverStarted {
		return nil
	}

	if p.IsServerRunning() {
		p.serverStarted = true
		return nil
	}

	// Check if opencode command exists
	opencodePath, err := exec.LookPath("opencode")
	if err != nil {
		return fmt.Errorf("opencode command not found: %w", err)
	}

	// Start the server
	args := []string{"serve", "--port", "4096"}
	p.serverCmd = exec.Command(opencodePath, args...)

	// Set password if configured
	if p.password != "" {
		p.serverCmd.Env = append(os.Environ(),
			"OPENCODE_SERVER_PASSWORD="+p.password,
			"OPENCODE_SERVER_USERNAME="+p.username,
		)
	}

	// Start in background
	if err := p.serverCmd.Start(); err != nil {
		return fmt.Errorf("failed to start opencode serve: %w", err)
	}

	// Wait for server to be ready
	for i := 0; i < 30; i++ {
		time.Sleep(500 * time.Millisecond)
		if p.IsServerRunning() {
			p.serverStarted = true
			return nil
		}
	}

	return fmt.Errorf("server failed to start within timeout")
}

// StopServer stops the OpenCode server if we started it
func (p *ZenHTTPProvider) StopServer() {
	p.serverMu.Lock()
	defer p.serverMu.Unlock()

	if p.serverCmd != nil && p.serverCmd.Process != nil {
		_ = p.serverCmd.Process.Kill() //nolint:errcheck
		_ = p.serverCmd.Wait()         //nolint:errcheck
	}
	p.serverStarted = false
}

// doRequest performs an authenticated HTTP request
func (p *ZenHTTPProvider) doRequest(ctx context.Context, method, path string, body interface{}) (*http.Response, error) {
	var bodyReader io.Reader
	if body != nil {
		bodyBytes, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %w", err)
		}
		bodyReader = bytes.NewReader(bodyBytes)
	}

	req, err := http.NewRequestWithContext(ctx, method, p.baseURL+path, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	if p.password != "" {
		req.SetBasicAuth(p.username, p.password)
	}

	return p.httpClient.Do(req)
}

// createSession creates a new chat session
func (p *ZenHTTPProvider) createSession(ctx context.Context) (string, error) {
	resp, err := p.doRequest(ctx, "POST", "/sessions", nil)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", fmt.Errorf("failed to create session: %s, failed to read body: %v", resp.Status, err)
		}
		return "", fmt.Errorf("failed to create session: %s - %s", resp.Status, string(body))
	}

	// Validate Content-Type before JSON decoding to catch HTML error pages
	contentType := resp.Header.Get("Content-Type")
	if contentType != "" && !strings.Contains(contentType, "application/json") {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("unexpected response Content-Type: %s (expected JSON, body prefix: %.200s)", contentType, string(body))
	}

	var session sessionResponse
	if err := json.NewDecoder(resp.Body).Decode(&session); err != nil {
		return "", fmt.Errorf("failed to decode session response: %w", err)
	}

	return session.ID, nil
}

// sendMessage sends a message to a session and returns the response
func (p *ZenHTTPProvider) sendMessage(ctx context.Context, sessionID, content string) (*messageResponse, error) {
	reqBody := messageRequest{
		Content: content,
		Model:   p.model,
	}

	resp, err := p.doRequest(ctx, "POST", "/sessions/"+sessionID+"/messages", reqBody)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to send message: %s, failed to read body: %v", resp.Status, err)
		}
		return nil, fmt.Errorf("failed to send message: %s - %s", resp.Status, string(body))
	}

	// Validate Content-Type before JSON decoding to catch HTML error pages
	contentType := resp.Header.Get("Content-Type")
	if contentType != "" && !strings.Contains(contentType, "application/json") {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected response Content-Type: %s (expected JSON, body prefix: %.200s)", contentType, string(body))
	}

	var msg messageResponse
	if err := json.NewDecoder(resp.Body).Decode(&msg); err != nil {
		return nil, fmt.Errorf("failed to decode message response: %w", err)
	}

	return &msg, nil
}

// Complete implements the LLMProvider interface
func (p *ZenHTTPProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	// Auto-start server if needed
	if p.autoStart && !p.IsServerRunning() {
		if err := p.StartServer(); err != nil {
			return nil, fmt.Errorf("failed to start server: %w", err)
		}
	}

	// Create session if needed. Guard p.sessionID against concurrent Complete
	// calls (the provider advertises MaxConcurrentRequests > 1).
	p.sessionMu.Lock()
	if p.sessionID == "" {
		sessionID, err := p.createSession(ctx)
		if err != nil {
			p.sessionMu.Unlock()
			return nil, fmt.Errorf("failed to create session: %w", err)
		}
		p.sessionID = sessionID
	}
	sessionID := p.sessionID
	p.sessionMu.Unlock()

	// Build prompt from messages
	var promptBuilder strings.Builder
	// Forward the system instruction (req.Prompt) ahead of the conversation so
	// it is never silently dropped when req.Messages is also populated.
	if req.Prompt != "" {
		promptBuilder.WriteString("System: ")
		promptBuilder.WriteString(req.Prompt)
		promptBuilder.WriteString("\n\n")
	}
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			promptBuilder.WriteString("System: ")
			promptBuilder.WriteString(msg.Content)
			promptBuilder.WriteString("\n\n")
		case "user":
			promptBuilder.WriteString(msg.Content)
			promptBuilder.WriteString("\n")
		case "assistant":
			promptBuilder.WriteString("Assistant: ")
			promptBuilder.WriteString(msg.Content)
			promptBuilder.WriteString("\n\n")
		}
	}

	prompt := promptBuilder.String()
	if prompt == "" && req.Prompt != "" {
		prompt = req.Prompt
	}

	if prompt == "" {
		return nil, fmt.Errorf("no prompt provided")
	}

	// Send message
	startTime := time.Now()
	msgResp, err := p.sendMessage(ctx, sessionID, prompt)
	duration := time.Since(startTime)

	if err != nil {
		return nil, fmt.Errorf("failed to send message: %w", err)
	}

	// Estimate token count
	promptTokens := len(prompt) / 4
	completionTokens := len(msgResp.Content) / 4

	return &models.LLMResponse{
		ID:           fmt.Sprintf("zen-http-%s", msgResp.ID),
		ProviderID:   "zen-http",
		ProviderName: "zen-http",
		Content:      msgResp.Content,
		FinishReason: "stop",
		TokensUsed:   promptTokens + completionTokens,
		ResponseTime: duration.Milliseconds(),
		CreatedAt:    time.Now(),
		Metadata: map[string]interface{}{
			"source":            "opencode-http",
			"session_id":        sessionID,
			"message_id":        msgResp.ID,
			"model":             msgResp.Model,
			"base_url":          p.baseURL,
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"latency":           duration.String(),
		},
	}, nil
}

// CompleteStream implements streaming completion using HTTP SSE
func (p *ZenHTTPProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	// For now, use non-streaming and send as single chunk
	// Full SSE streaming could be implemented with /sessions/{id}/stream endpoint
	ch := make(chan *models.LLMResponse, 1)

	go func() {
		defer close(ch)

		resp, err := p.Complete(ctx, req)
		if err != nil {
			ch <- &models.LLMResponse{
				ProviderID:   "zen-http",
				ProviderName: "zen-http",
				Metadata: map[string]interface{}{
					"error": err.Error(),
				},
			}
			return
		}
		ch <- resp
	}()

	return ch, nil
}

// HealthCheck checks if the HTTP server is healthy
func (p *ZenHTTPProvider) HealthCheck() error {
	if !p.IsServerRunning() {
		if p.autoStart {
			return p.StartServer()
		}
		return fmt.Errorf("OpenCode server not running at %s", p.baseURL)
	}
	return nil
}

// GetCapabilities returns the capabilities of the HTTP provider
func (p *ZenHTTPProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: []string{
			"big-pickle",
			"gpt-5-nano",
			"glm-4.7",
			"kimi-k2",
			"gemini-3-flash",
			// Note: qwen3-coder removed from free tier 2026-02
		},
		SupportsStreaming: true,
		SupportsTools:     false, // Could be added via /commands endpoint
		Limits: models.ModelLimits{
			MaxTokens:             p.maxTokens,
			MaxConcurrentRequests: 10, // HTTP supports concurrent requests
		},
	}
}

// ValidateConfig validates the configuration
func (p *ZenHTTPProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var issues []string

	// Check if opencode is installed
	if _, err := exec.LookPath("opencode"); err != nil {
		issues = append(issues, "opencode command not available")
	}

	if len(issues) > 0 {
		return false, issues
	}
	return true, nil
}

// GetName returns the provider name
func (p *ZenHTTPProvider) GetName() string {
	return "zen-http"
}

// GetProviderType returns the provider type
func (p *ZenHTTPProvider) GetProviderType() string {
	return "zen"
}

// GetCurrentModel returns the current model
func (p *ZenHTTPProvider) GetCurrentModel() string {
	return p.model
}

// SetModel sets the model to use
func (p *ZenHTTPProvider) SetModel(model string) {
	p.model = model
}

// IsZenHTTPAvailable checks if OpenCode HTTP server can be used
func IsZenHTTPAvailable() bool {
	_, err := exec.LookPath("opencode")
	return err == nil
}

// CanUseZenHTTP returns true if OpenCode HTTP server can be used
func CanUseZenHTTP() bool {
	return IsZenHTTPAvailable()
}
