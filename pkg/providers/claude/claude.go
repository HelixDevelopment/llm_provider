package claude

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

// Settings keys for this adapter. The two AUTH paths get two MODEL keys and
// share one BASE_URL key, because that is what the code actually does: both
// transports post to ClaudeAPIURL, but an OAuth token from the Claude Code CLI
// is product-restricted and an operator may well want a different model on that
// path than on the API-key path. One shared MODEL key would make choosing a
// model for one transport silently change the other.
const (
	// SettingsProvider keys the shared endpoint and timeout:
	// LLMPROVIDER_CLAUDE_BASE_URL and LLMPROVIDER_CLAUDE_TIMEOUT, and the
	// API-key path's model: LLMPROVIDER_CLAUDE_MODEL.
	SettingsProvider = "claude"
	// SettingsProviderOAuth keys the OAuth path's model:
	// LLMPROVIDER_CLAUDE_OAUTH_MODEL.
	SettingsProviderOAuth = "claude_oauth"
)

// DefaultHTTPTimeout is the compiled fallback for this adapter's HTTP client.
// It is a starting point, not a decision the library keeps making:
// LLMPROVIDER_CLAUDE_TIMEOUT overrides it (see pkg/settings).
const DefaultHTTPTimeout = 60 * time.Second

const (
	ClaudeAPIURL     = "https://api.anthropic.com/v1/messages"
	ClaudeModel      = "claude-sonnet-4-6" // Current default model for API key auth (latest)
	ClaudeOAuthModel = "claude-sonnet-4-6" // Default model for OAuth auth (Claude Code compatible, latest)

	// Claude 4.6 (Latest generation - February 2026)
	ClaudeOpus46Model   = "claude-opus-4-6"
	ClaudeSonnet46Model = "claude-sonnet-4-6"

	// Alternative models
	ClaudeOpusModel  = "claude-opus-4-20250514"
	ClaudeHaikuModel = "claude-haiku-4-20250514"
	ClaudeSonnet35   = "claude-3-5-sonnet-20241022"

	// OAuth restriction error message from Anthropic
	// IMPORTANT: OAuth tokens from Claude Code CLI are PRODUCT-RESTRICTED and can ONLY
	// be used with Claude Code itself. They CANNOT be used for general API access.
	// See: https://platform.claude.com/docs/en/api/overview
	OAuthRestrictionError = "This credential is only authorized for use with Claude Code"
)

// AuthType represents the type of authentication used
type AuthType string

const (
	AuthTypeAPIKey AuthType = "api_key"
	AuthTypeOAuth  AuthType = "oauth"
)

type ClaudeProvider struct {
	apiKey          string
	baseURL         string
	model           string
	httpClient      *http.Client
	retryConfig     RetryConfig
	authType        AuthType
	oauthCredReader OAuthCredentialReader
	discoverer      *discovery.Discoverer
}

// RetryConfig defines retry behavior for API calls
type RetryConfig struct {
	MaxRetries   int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Multiplier   float64
}

type ClaudeRequest struct {
	Model         string          `json:"model"`
	MaxTokens     int             `json:"max_tokens,omitempty"`
	Temperature   float64         `json:"temperature,omitempty"`
	TopP          float64         `json:"top_p,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Stream        bool            `json:"stream,omitempty"`
	Messages      []ClaudeMessage `json:"messages"`
	System        string          `json:"system,omitempty"`
	// Tools for function calling
	Tools      []ClaudeTool `json:"tools,omitempty"`
	ToolChoice interface{}  `json:"tool_choice,omitempty"` // "auto", "any", {"type": "tool", "name": "tool_name"}
}

// ClaudeTool represents a tool definition for Claude
type ClaudeTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

type ClaudeMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ClaudeResponse struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Role         string          `json:"role"`
	Content      []ClaudeContent `json:"content"`
	Model        string          `json:"model"`
	StopReason   *string         `json:"stop_reason"`
	StopSequence *string         `json:"stop_sequence"`
	Usage        ClaudeUsage     `json:"usage"`
}

type ClaudeContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
	// For tool_use content blocks
	ID    string                 `json:"id,omitempty"`
	Name  string                 `json:"name,omitempty"`
	Input map[string]interface{} `json:"input,omitempty"`
}

type ClaudeUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type ClaudeStreamResponse struct {
	Type    string          `json:"type"`
	Message *ClaudeResponse `json:"message,omitempty"`
	Delta   *ClaudeDelta    `json:"delta,omitempty"`
	Usage   *ClaudeUsage    `json:"usage,omitempty"`
}

type ClaudeDelta struct {
	Type         string `json:"type"`
	Text         string `json:"text,omitempty"`
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`
}

// DefaultRetryConfig returns sensible defaults for Claude API retry behavior
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:   3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
	}
}

func NewClaudeProvider(apiKey, baseURL, model string) *ClaudeProvider {
	return NewClaudeProviderWithRetry(apiKey, baseURL, model, DefaultRetryConfig())
}

func NewClaudeProviderWithRetry(apiKey, baseURL, model string, retryConfig RetryConfig) *ClaudeProvider {
	if baseURL == "" {
		// LLMPROVIDER_CLAUDE_BASE_URL.
		baseURL = settings.BaseURL(SettingsProvider, ClaudeAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_CLAUDE_MODEL.
		model = settings.Model(SettingsProvider, ClaudeModel)
	}

	p := &ClaudeProvider{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			// LLMPROVIDER_CLAUDE_TIMEOUT.
			Timeout: settings.Timeout(SettingsProvider, DefaultHTTPTimeout),
		},
		retryConfig: retryConfig,
		authType:    AuthTypeAPIKey,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "claude",
		ModelsEndpoint: "https://api.anthropic.com/v1/models",
		ModelsDevID:    "anthropic",
		APIKey:         apiKey,
		AuthHeader:     "x-api-key",
		AuthPrefix:     "",
		ExtraHeaders: map[string]string{
			"anthropic-version": "2023-06-01",
		},
		FallbackModels: []string{
			// Claude 4.6 (Latest - February 2026)
			"claude-opus-4-6",
			"claude-sonnet-4-6",
			// Claude 4.5 (November 2025)
			"claude-opus-4-5-20251101",
			"claude-sonnet-4-5-20250929",
			"claude-haiku-4-5-20251001",
			// Claude 4 models (May 2025)
			"claude-opus-4-20250514",
			"claude-sonnet-4-20250514",
			"claude-haiku-4-20250514",
			// Claude 3.5 models
			"claude-3-5-sonnet-20241022",
			"claude-3-5-haiku-20241022",
			// Claude 3 models (legacy)
			"claude-3-opus-20240229",
			"claude-3-sonnet-20240229",
			"claude-3-haiku-20240307",
		},
	})

	return p
}

// OAuthCredentialReader is an interface for reading OAuth credentials.
// Implement this interface to provide OAuth support for the Claude provider.
type OAuthCredentialReader interface {
	HasValidClaudeCredentials() bool
	GetClaudeAccessToken() (string, error)
}

// NewClaudeProviderWithOAuth creates a new Claude provider using OAuth credentials
func NewClaudeProviderWithOAuth(baseURL, model string, credReader OAuthCredentialReader) (*ClaudeProvider, error) {
	return NewClaudeProviderWithOAuthAndRetry(baseURL, model, DefaultRetryConfig(), credReader)
}

// NewClaudeProviderWithOAuthAndRetry creates a new Claude provider using OAuth credentials with custom retry config
func NewClaudeProviderWithOAuthAndRetry(baseURL, model string, retryConfig RetryConfig, credReader OAuthCredentialReader) (*ClaudeProvider, error) {
	if baseURL == "" {
		// LLMPROVIDER_CLAUDE_BASE_URL — shared with the API-key path by design.
		baseURL = settings.BaseURL(SettingsProvider, ClaudeAPIURL)
	}
	if model == "" {
		// LLMPROVIDER_CLAUDE_OAUTH_MODEL — deliberately NOT the API-key path's
		// key; see SettingsProviderOAuth.
		model = settings.Model(SettingsProviderOAuth, ClaudeOAuthModel)
	}

	if credReader == nil || !credReader.HasValidClaudeCredentials() {
		return nil, fmt.Errorf("no valid Claude OAuth credentials available")
	}

	p := &ClaudeProvider{
		apiKey:  "",
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			// LLMPROVIDER_CLAUDE_TIMEOUT.
			Timeout: settings.Timeout(SettingsProvider, DefaultHTTPTimeout),
		},
		retryConfig:     retryConfig,
		authType:        AuthTypeOAuth,
		oauthCredReader: credReader,
	}

	p.discoverer = discovery.NewDiscoverer(discovery.ProviderConfig{
		ProviderName:   "claude",
		ModelsEndpoint: "https://api.anthropic.com/v1/models",
		ModelsDevID:    "anthropic",
		APIKey:         "",
		AuthHeader:     "x-api-key",
		AuthPrefix:     "",
		ExtraHeaders: map[string]string{
			"anthropic-version": "2023-06-01",
		},
		FallbackModels: []string{
			"claude-opus-4-6",
			"claude-sonnet-4-6",
			"claude-opus-4-5-20251101",
			"claude-sonnet-4-5-20250929",
			"claude-haiku-4-5-20251001",
			"claude-opus-4-20250514",
			"claude-sonnet-4-20250514",
			"claude-haiku-4-20250514",
			"claude-3-5-sonnet-20241022",
			"claude-3-5-haiku-20241022",
			"claude-3-opus-20240229",
			"claude-3-sonnet-20240229",
			"claude-3-haiku-20240307",
		},
	})

	return p, nil
}

// GetAuthType returns the authentication type being used
func (p *ClaudeProvider) GetAuthType() AuthType {
	return p.authType
}

// getAuthHeader returns the appropriate authorization header based on auth type
func (p *ClaudeProvider) getAuthHeader() (string, string, error) {
	switch p.authType {
	case AuthTypeOAuth:
		if p.oauthCredReader == nil {
			return "", "", fmt.Errorf("OAuth credential reader not initialized")
		}
		token, err := p.oauthCredReader.GetClaudeAccessToken()
		if err != nil {
			return "", "", fmt.Errorf("failed to get OAuth token: %w", err)
		}
		// Claude OAuth tokens (sk-ant-oat01-*) use Bearer authentication
		// Required headers: Authorization: Bearer <token>, anthropic-beta: oauth-2025-04-20
		return "Authorization", "Bearer " + token, nil
	default:
		// Regular API keys use x-api-key header
		return "x-api-key", p.apiKey, nil
	}
}

func (p *ClaudeProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	startTime := time.Now()

	// Convert internal request to Claude format
	claudeReq := p.convertRequest(req)

	// Make API call
	resp, err := p.makeAPICall(ctx, claudeReq)
	if err != nil {
		return nil, fmt.Errorf("Claude API call failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Parse response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		bodyStr := string(body)
		// Check for OAuth restriction error
		if strings.Contains(bodyStr, OAuthRestrictionError) {
			return nil, &OAuthRestrictionErr{
				// CONST-046 round-441: OAuth-restriction message routed through i18n.
				Message: i18n.Tr(context.Background(),
					"llmprovider_claude_oauth_product_restricted", nil),
			}
		}
		return nil, fmt.Errorf("Claude API error: %d - %s", resp.StatusCode, bodyStr)
	}

	var claudeResp ClaudeResponse
	if err := json.Unmarshal(body, &claudeResp); err != nil {
		return nil, fmt.Errorf("failed to parse Claude response: %w", err)
	}

	// Convert back to internal format
	return p.convertResponse(req, &claudeResp, startTime), nil
}

func (p *ClaudeProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	startTime := time.Now()

	// Convert internal request to Claude format
	claudeReq := p.convertRequest(req)
	claudeReq.Stream = true

	// Make streaming API call
	resp, err := p.makeAPICall(ctx, claudeReq)
	if err != nil {
		return nil, fmt.Errorf("Claude streaming API call failed: %w", err)
	}

	// Guard on status BEFORE consuming the body as SSE. makeAPICall returns
	// (resp, nil) for non-retryable errors (e.g. HTTP 400), and a JSON error
	// body has no "data: " lines, so the stream loop would otherwise hit EOF
	// and emit a FinishReason="stop" final response — a success-on-error
	// bluff. Surface the error instead.
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body) //nolint:errcheck
		_ = resp.Body.Close()
		bodyStr := string(body)
		if strings.Contains(bodyStr, OAuthRestrictionError) {
			return nil, &OAuthRestrictionErr{
				Message: i18n.Tr(context.Background(),
					"llmprovider_claude_oauth_product_restricted", nil),
			}
		}
		return nil, fmt.Errorf("Claude API error: %d - %s", resp.StatusCode, bodyStr)
	}

	// Create response channel
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
				// Send error response and exit
				errorResp := &models.LLMResponse{
					ID:             "stream-error-" + req.ID,
					RequestID:      req.ID,
					ProviderID:     "claude",
					ProviderName:   "Claude",
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
			var streamResp ClaudeStreamResponse
			if err := json.Unmarshal(line, &streamResp); err != nil {
				continue // Skip malformed JSON
			}

			// Handle different event types
			switch streamResp.Type {
			case "content_block_delta":
				if streamResp.Delta != nil && streamResp.Delta.Type == "text_delta" {
					delta := streamResp.Delta.Text
					if delta != "" {
						fullContent += delta

						// Send chunk response
						chunkResp := &models.LLMResponse{
							ID:             "claude-stream-" + req.ID,
							RequestID:      req.ID,
							ProviderID:     "claude",
							ProviderName:   "Claude",
							Content:        delta,
							Confidence:     0.9, // High confidence for Claude
							TokensUsed:     1,   // Estimated
							ResponseTime:   time.Since(startTime).Milliseconds(),
							FinishReason:   "",
							Selected:       false,
							SelectionScore: 0.0,
							CreatedAt:      time.Now(),
						}
						ch <- chunkResp
					}
				}
			case "message_stop":
				// Stream finished
				break
			}
		}

		// Send final response
		finalResp := &models.LLMResponse{
			ID:             "claude-final-" + req.ID,
			RequestID:      req.ID,
			ProviderID:     "claude",
			ProviderName:   "Claude",
			Content:        "",
			Confidence:     0.9,
			TokensUsed:     len(fullContent) / 4, // Rough estimate
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

func (p *ClaudeProvider) convertRequest(req *models.LLMRequest) ClaudeRequest {
	// Convert messages
	messages := make([]ClaudeMessage, 0, len(req.Messages))

	// Add conversation messages (Claude doesn't use system prompt in messages)
	for _, msg := range req.Messages {
		if msg.Role != "system" {
			messages = append(messages, ClaudeMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}

	// Extract system message if present
	var systemPrompt string
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			systemPrompt = msg.Content
			break
		}
	}

	claudeReq := ClaudeRequest{
		Model:         p.model,
		MaxTokens:     req.ModelParams.MaxTokens,
		Temperature:   req.ModelParams.Temperature,
		TopP:          req.ModelParams.TopP,
		StopSequences: req.ModelParams.StopSequences,
		Stream:        false,
		Messages:      messages,
		System:        systemPrompt,
	}

	// Convert tools if present
	if len(req.Tools) > 0 {
		claudeReq.Tools = make([]ClaudeTool, 0, len(req.Tools))
		for _, tool := range req.Tools {
			if tool.Type == "function" {
				claudeTool := ClaudeTool{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					InputSchema: tool.Function.Parameters,
				}
				// Ensure input_schema has a type field (required by Claude)
				if claudeTool.InputSchema == nil {
					claudeTool.InputSchema = map[string]interface{}{
						"type":       "object",
						"properties": map[string]interface{}{},
					}
				} else if _, hasType := claudeTool.InputSchema["type"]; !hasType {
					claudeTool.InputSchema["type"] = "object"
				}
				claudeReq.Tools = append(claudeReq.Tools, claudeTool)
			}
		}
		// Set tool_choice based on request
		// CRITICAL: Claude API requires tool_choice to be an object {"type": "auto"}, not a string
		if req.ToolChoice != nil {
			switch tc := req.ToolChoice.(type) {
			case string:
				// Convert string "auto" or "any" to object format
				if tc == "auto" || tc == "any" {
					claudeReq.ToolChoice = map[string]interface{}{"type": tc}
				} else {
					claudeReq.ToolChoice = req.ToolChoice
				}
			default:
				claudeReq.ToolChoice = req.ToolChoice
			}
		}
	}

	return claudeReq
}

func (p *ClaudeProvider) convertResponse(req *models.LLMRequest, claudeResp *ClaudeResponse, startTime time.Time) *models.LLMResponse {
	var content string
	var finishReason string
	var toolCalls []models.ToolCall

	// Process all content blocks
	for _, block := range claudeResp.Content {
		switch block.Type {
		case "text":
			content += block.Text
		case "tool_use":
			// Convert tool_use to ToolCall
			args, err := json.Marshal(block.Input)
			if err != nil {
				args = []byte("{}")
			}
			toolCalls = append(toolCalls, models.ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: models.ToolCallFunction{
					Name:      block.Name,
					Arguments: string(args),
				},
			})
		}
	}

	if claudeResp.StopReason != nil {
		finishReason = *claudeResp.StopReason
		// Map Claude's tool_use stop reason to OpenAI's tool_calls
		if finishReason == "tool_use" {
			finishReason = "tool_calls"
		}
	}

	// Calculate confidence based on finish reason and response quality
	confidence := p.calculateConfidence(content, finishReason)

	return &models.LLMResponse{
		ID:           claudeResp.ID,
		RequestID:    req.ID,
		ProviderID:   "claude",
		ProviderName: "Claude",
		Content:      content,
		Confidence:   confidence,
		TokensUsed:   claudeResp.Usage.OutputTokens,
		ResponseTime: time.Since(startTime).Milliseconds(),
		FinishReason: finishReason,
		Metadata: map[string]any{
			"model":        claudeResp.Model,
			"input_tokens": claudeResp.Usage.InputTokens,
			"type":         claudeResp.Type,
		},
		Selected:       false,
		SelectionScore: 0.0,
		CreatedAt:      time.Now(),
		ToolCalls:      toolCalls,
	}
}

func (p *ClaudeProvider) calculateConfidence(content, finishReason string) float64 {
	confidence := 0.9 // High base confidence for Claude

	// Adjust based on finish reason
	switch finishReason {
	case "end_turn":
		confidence += 0.05
	case "max_tokens":
		confidence -= 0.1
	case "stop_sequence":
		confidence += 0.02
	}

	// Adjust based on content length and quality
	if len(content) > 50 {
		confidence += 0.02
	}
	if len(content) > 200 {
		confidence += 0.02
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

func (p *ClaudeProvider) makeAPICall(ctx context.Context, req ClaudeRequest) (*http.Response, error) {
	return p.makeAPICallWithAuthRetry(ctx, req, true)
}

// makeAPICallWithAuthRetry performs the API call with optional 401 retry
func (p *ClaudeProvider) makeAPICallWithAuthRetry(ctx context.Context, req ClaudeRequest, allowAuthRetry bool) (*http.Response, error) {
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

		// Set headers
		httpReq.Header.Set("Content-Type", "application/json")

		// Set authentication header based on auth type
		authHeaderName, authHeaderValue, authErr := p.getAuthHeader()
		if authErr != nil {
			return nil, fmt.Errorf("failed to get auth header: %w", authErr)
		}
		httpReq.Header.Set(authHeaderName, authHeaderValue)

		httpReq.Header.Set("anthropic-version", "2023-06-01")
		httpReq.Header.Set("User-Agent", "LLMProvider/1.0")

		// Add OAuth-specific headers for Claude Code OAuth tokens
		if p.authType == AuthTypeOAuth {
			httpReq.Header.Set("anthropic-beta", "oauth-2025-04-20")
			httpReq.Header.Set("anthropic-product", "claude-code")
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
		// This handles transient auth issues (token validation delays, auth service hiccups)
		if isAuthRetryableStatus(resp.StatusCode) && allowAuthRetry {
			_ = resp.Body.Close()
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
// (e.g., token validation delays, temporary auth service issues)
func isAuthRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusUnauthorized // 401
}

// waitWithJitter waits for the specified duration plus random jitter
func (p *ClaudeProvider) waitWithJitter(ctx context.Context, delay time.Duration) {
	// Add 10% jitter - using math/rand is acceptable for non-security jitter
	jitter := time.Duration(rand.Float64() * 0.1 * float64(delay)) // #nosec G404 - jitter doesn't require cryptographic randomness
	select {
	case <-ctx.Done():
	case <-time.After(delay + jitter):
	}
}

// nextDelay calculates the next delay using exponential backoff
func (p *ClaudeProvider) nextDelay(currentDelay time.Duration) time.Duration {
	nextDelay := time.Duration(float64(currentDelay) * p.retryConfig.Multiplier)
	if nextDelay > p.retryConfig.MaxDelay {
		nextDelay = p.retryConfig.MaxDelay
	}
	return nextDelay
}

// GetCapabilities returns the capabilities of the Claude provider
func (p *ClaudeProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{
		SupportedModels: p.discoverer.DiscoverModels(),
		SupportedFeatures: []string{
			"text_completion",
			"chat",
			"function_calling",
			"streaming",
		},
		SupportedRequestTypes: []string{
			"text_completion",
			"chat",
		},
		SupportsStreaming:       true,
		SupportsFunctionCalling: true,
		SupportsVision:          true,
		SupportsTools:           true,
		SupportsSearch:          false,
		SupportsReasoning:       true,
		SupportsCodeCompletion:  true,
		SupportsCodeAnalysis:    true,
		SupportsRefactoring:     true,
		Limits: models.ModelLimits{
			MaxTokens:             200000,
			MaxInputLength:        100000,
			MaxOutputLength:       4096,
			MaxConcurrentRequests: 10,
		},
		Metadata: map[string]string{
			"provider":     "Anthropic",
			"model_family": "Claude",
			"api_version":  "2023-06-01",
		},
	}
}

// ValidateConfig validates the provider configuration
func (p *ClaudeProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	var errors []string

	// For OAuth auth, we don't need API key - check OAuth credentials instead
	if p.authType == AuthTypeOAuth {
		if p.oauthCredReader == nil {
			errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_oauth_reader_required", nil))
		} else if !p.oauthCredReader.HasValidClaudeCredentials() {
			errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_oauth_creds_invalid", nil))
		}
	} else {
		// API key auth
		if p.apiKey == "" {
			errors = append(errors, i18n.Tr(context.Background(), "llmprovider_validate_api_key_required", nil))
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

// modelsURL derives the health/models endpoint from the configured base URL.
// Claude's base IS the messages endpoint ("/v1/messages"), so the health check
// targets the configured baseURL directly (CONST-051(B) config-injection)
// instead of a hardcoded production literal.
func (p *ClaudeProvider) modelsURL() string {
	return p.baseURL
}

// HealthCheck implements health checking for the Claude provider
func (p *ClaudeProvider) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Simple health check against the configured messages endpoint. Derived from
	// the injected baseURL (CONST-051(B) config-injection) instead of a hardcoded
	// production literal, so a custom baseURL (e.g. a proxy) is honored.
	req, err := http.NewRequestWithContext(ctx, "GET", p.modelsURL(), nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	// Set authentication header based on auth type
	authHeaderName, authHeaderValue, authErr := p.getAuthHeader()
	if authErr != nil {
		return fmt.Errorf("failed to get auth header: %w", authErr)
	}
	req.Header.Set(authHeaderName, authHeaderValue)
	req.Header.Set("anthropic-version", "2023-06-01")

	// Add OAuth-specific headers for Claude Code OAuth tokens
	if p.authType == AuthTypeOAuth {
		req.Header.Set("anthropic-beta", "oauth-2025-04-20")
		req.Header.Set("anthropic-product", "claude-code")
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Read response body for error detection
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("health check failed to read response body: %w", err)
	}
	bodyStr := string(body)

	// Claude API returns 400 for GET requests to messages endpoint (expected)
	// We just check that the API is reachable and returns a response

	// Check for OAuth restriction error first
	// IMPORTANT: OAuth tokens from Claude Code CLI are PRODUCT-RESTRICTED
	// They can ONLY be used with Claude Code itself, not the standard API
	if strings.Contains(bodyStr, OAuthRestrictionError) {
		return &OAuthRestrictionErr{
			Message: "OAuth tokens from Claude Code CLI are product-restricted and cannot be used for general API access. Use an API key from console.anthropic.com instead.",
		}
	}

	// For OAuth tokens, 401 means the token is invalid/expired
	if resp.StatusCode == http.StatusUnauthorized {
		return fmt.Errorf("health check failed: unauthorized (token may be expired or invalid)")
	}
	if resp.StatusCode >= 500 {
		return fmt.Errorf("health check failed with server error: %d", resp.StatusCode)
	}

	return nil
}

// OAuthRestrictionErr indicates that an OAuth token cannot be used for general API access
// This is NOT an error condition per se - it's expected behavior for Claude Code OAuth tokens
type OAuthRestrictionErr struct {
	Message string
}

func (e *OAuthRestrictionErr) Error() string {
	return e.Message
}
