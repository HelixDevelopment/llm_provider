package zen

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"digital.vasic.llmprovider/pkg/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// withOpenCodeOnPath replaces PATH with a directory this test owns, optionally
// containing an executable named `opencode`, and returns that directory.
//
// The point is that the EXPECTED answer is then known independently of the
// host. The tests below used to derive their expectation from the very
// exec.LookPath call the functions under test wrap, which made them agree with
// the implementation by construction: they could not fail in either
// environment, on a machine with opencode or without it.
func withOpenCodeOnPath(t *testing.T, present bool, mode os.FileMode) string {
	t.Helper()
	dir := t.TempDir()
	if present {
		require.NoError(t, os.WriteFile(
			filepath.Join(dir, "opencode"),
			[]byte("#!/bin/sh\nexit 0\n"), mode))
	}
	t.Setenv("PATH", dir)
	return dir
}

// TestZenHTTPProvider_DefaultConfig tests default configuration
func TestZenHTTPProvider_DefaultConfig(t *testing.T) {
	config := DefaultZenHTTPConfig()

	assert.Equal(t, "http://localhost:4096", config.BaseURL)
	assert.Equal(t, "opencode", config.Username)
	assert.Equal(t, "big-pickle", config.Model)
	assert.Equal(t, 180*time.Second, config.Timeout)
	assert.Equal(t, 8192, config.MaxTokens)
	assert.True(t, config.AutoStart)
}

// TestZenHTTPProvider_NewProvider tests provider creation
func TestZenHTTPProvider_NewProvider(t *testing.T) {
	config := ZenHTTPConfig{
		BaseURL:   "http://localhost:5000",
		Username:  "testuser",
		Password:  "testpass",
		Model:     "grok-code",
		Timeout:   60 * time.Second,
		MaxTokens: 4096,
		AutoStart: false,
	}

	provider := NewZenHTTPProvider(config)

	assert.NotNil(t, provider)
	assert.Equal(t, "http://localhost:5000", provider.baseURL)
	assert.Equal(t, "testuser", provider.username)
	assert.Equal(t, "testpass", provider.password)
	assert.Equal(t, "grok-code", provider.model)
	assert.Equal(t, 60*time.Second, provider.timeout)
	assert.Equal(t, 4096, provider.maxTokens)
	assert.False(t, provider.autoStart)
	assert.NotNil(t, provider.httpClient)
}

// TestZenHTTPProvider_NewProviderWithModel tests model-specific creation
func TestZenHTTPProvider_NewProviderWithModel(t *testing.T) {
	provider := NewZenHTTPProviderWithModel("big-pickle")

	assert.NotNil(t, provider)
	assert.Equal(t, "big-pickle", provider.model)
}

// TestZenHTTPProvider_GetName tests provider name
func TestZenHTTPProvider_GetName(t *testing.T) {
	provider := NewZenHTTPProviderWithModel("grok-code")
	assert.Equal(t, "zen-http", provider.GetName())
}

// TestZenHTTPProvider_GetProviderType tests provider type
func TestZenHTTPProvider_GetProviderType(t *testing.T) {
	provider := NewZenHTTPProviderWithModel("grok-code")
	assert.Equal(t, "zen", provider.GetProviderType())
}

// TestZenHTTPProvider_GetCapabilities tests capabilities
func TestZenHTTPProvider_GetCapabilities(t *testing.T) {
	provider := NewZenHTTPProviderWithModel("grok-code")
	caps := provider.GetCapabilities()

	assert.NotNil(t, caps)
	assert.True(t, caps.SupportsStreaming)
	assert.False(t, caps.SupportsTools)
	assert.GreaterOrEqual(t, len(caps.SupportedModels), 5, "Should support multiple models")

	// Check for specific free models (as of 2026-02)
	assert.Contains(t, caps.SupportedModels, "big-pickle")
	assert.Contains(t, caps.SupportedModels, "gpt-5-nano")
	assert.Contains(t, caps.SupportedModels, "glm-4.7")
	// Note: qwen3-coder removed from free tier 2026-02
	assert.Contains(t, caps.SupportedModels, "kimi-k2")
	assert.Contains(t, caps.SupportedModels, "gemini-3-flash")
}

// TestZenHTTPProvider_SetModel tests model setting
func TestZenHTTPProvider_SetModel(t *testing.T) {
	provider := NewZenHTTPProviderWithModel("grok-code")
	assert.Equal(t, "grok-code", provider.GetCurrentModel())

	provider.SetModel("big-pickle")
	assert.Equal(t, "big-pickle", provider.GetCurrentModel())
}

// TestZenHTTPProvider_DefaultValues tests default value application
func TestZenHTTPProvider_DefaultValues(t *testing.T) {
	// Empty config should get defaults
	config := ZenHTTPConfig{}
	provider := NewZenHTTPProvider(config)

	assert.Equal(t, "http://localhost:4096", provider.baseURL)
	assert.Equal(t, "opencode", provider.username)
	assert.Equal(t, 180*time.Second, provider.timeout)
	assert.Equal(t, 8192, provider.maxTokens)
}

// TestZenHTTPProvider_URLTrailingSlash tests URL normalization
func TestZenHTTPProvider_URLTrailingSlash(t *testing.T) {
	config := ZenHTTPConfig{
		BaseURL: "http://localhost:4096/", // With trailing slash
	}
	provider := NewZenHTTPProvider(config)

	// Trailing slash should be removed
	assert.Equal(t, "http://localhost:4096", provider.baseURL)
}

// TestZenHTTPProvider_ValidateConfig tests config validation
func TestZenHTTPProvider_ValidateConfig(t *testing.T) {
	provider := NewZenHTTPProviderWithModel("big-pickle")

	valid, errs := provider.ValidateConfig(nil)

	if IsZenHTTPAvailable() {
		assert.True(t, valid, "Should be true when opencode is installed")
		assert.Empty(t, errs, "Should have no issues when opencode is installed")
	} else {
		assert.False(t, valid, "Should be false when opencode is not installed")
		assert.NotEmpty(t, errs, "Should have issues when opencode is not installed")
		assert.Contains(t, errs[0], "not available")
	}
}

// TestZenHTTPProvider_IsServerRunning tests server check
func TestZenHTTPProvider_IsServerRunning(t *testing.T) {
	provider := NewZenHTTPProviderWithModel("big-pickle")

	running := provider.IsServerRunning()
	t.Logf("OpenCode HTTP server running: %v", running)

	// Should not panic
	assert.NotPanics(t, func() {
		provider.IsServerRunning()
	})
}

// TestIsZenHTTPAvailable drives the probe from a CONTROLLED PATH, so each case
// has an expected answer that does not come from the implementation itself.
func TestIsZenHTTPAvailable(t *testing.T) {
	t.Run("absent from PATH", func(t *testing.T) {
		withOpenCodeOnPath(t, false, 0o700)
		assert.False(t, IsZenHTTPAvailable(),
			"no opencode on PATH must report unavailable")
		assert.False(t, CanUseZenHTTP(),
			"CanUseZenHTTP must follow IsZenHTTPAvailable")
		assert.False(t, IsOpenCodeInstalled(),
			"IsOpenCodeInstalled must answer the same question")
	})

	t.Run("present on PATH", func(t *testing.T) {
		withOpenCodeOnPath(t, true, 0o700)
		assert.True(t, IsZenHTTPAvailable(),
			"an executable opencode on PATH must report available")
		assert.True(t, CanUseZenHTTP(),
			"CanUseZenHTTP must follow IsZenHTTPAvailable")
		assert.True(t, IsOpenCodeInstalled(),
			"IsOpenCodeInstalled must answer the same question")
	})

	t.Run("present but not executable", func(t *testing.T) {
		withOpenCodeOnPath(t, true, 0o600)
		assert.False(t, IsZenHTTPAvailable(),
			"a non-executable file named opencode is not an installation")
	})
}

// TestZenHTTPProvider_APITypes tests API type structures
func TestZenHTTPProvider_APITypes(t *testing.T) {
	// Test sessionResponse serialization
	session := sessionResponse{
		ID:        "sess-123",
		Title:     "Test Session",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	data, err := json.Marshal(session)
	assert.NoError(t, err)
	assert.Contains(t, string(data), `"id":"sess-123"`)
	assert.Contains(t, string(data), `"title":"Test Session"`)

	// Test messageRequest serialization
	msgReq := messageRequest{
		Content: "Hello, world!",
		Model:   "big-pickle",
	}

	data, err = json.Marshal(msgReq)
	assert.NoError(t, err)
	assert.Contains(t, string(data), `"content":"Hello, world!"`)
	assert.Contains(t, string(data), `"model":"big-pickle"`)

	// Test messageResponse deserialization
	respJSON := `{
		"id": "msg-456",
		"role": "assistant",
		"content": "Hello! How can I help?",
		"model": "big-pickle",
		"createdAt": "2026-01-27T10:00:00Z"
	}`

	var msgResp messageResponse
	err = json.Unmarshal([]byte(respJSON), &msgResp)
	assert.NoError(t, err)
	assert.Equal(t, "msg-456", msgResp.ID)
	assert.Equal(t, "assistant", msgResp.Role)
	assert.Equal(t, "Hello! How can I help?", msgResp.Content)
	assert.Equal(t, "big-pickle", msgResp.Model)

	// Test errorResponse deserialization
	errJSON := `{"error": "not_found", "message": "Session not found"}`

	var errResp errorResponse
	err = json.Unmarshal([]byte(errJSON), &errResp)
	assert.NoError(t, err)
	assert.Equal(t, "not_found", errResp.Error)
	assert.Equal(t, "Session not found", errResp.Message)
}

// TestZenHTTPProvider_Complete_NoPrompt tests error on empty prompt
func TestZenHTTPProvider_Complete_NoPrompt(t *testing.T) {
	provider := NewZenHTTPProviderWithModel("big-pickle")
	provider.autoStart = false // Don't try to start server

	// Simulate running server
	provider.serverStarted = true
	provider.sessionID = "test-session"

	ctx := context.Background()
	resp, err := provider.Complete(ctx, &models.LLMRequest{
		Prompt:   "",
		Messages: nil,
	})

	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "no prompt")
}

// TestZenHTTPProvider_HealthCheck_NotRunning tests health check when server not running
func TestZenHTTPProvider_HealthCheck_NotRunning(t *testing.T) {
	provider := NewZenHTTPProviderWithModel("big-pickle")
	provider.autoStart = false // Don't auto-start

	// Mock a server that's not running
	if !provider.IsServerRunning() {
		err := provider.HealthCheck()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not running")
	}
}

// TestZenHTTPProvider_StopServer tests stop server functionality
func TestZenHTTPProvider_StopServer(t *testing.T) {
	provider := &ZenHTTPProvider{
		model:         "big-pickle",
		serverStarted: true,
	}

	// Stop should not panic even without a running process
	assert.NotPanics(t, func() {
		provider.StopServer()
	})

	assert.False(t, provider.serverStarted)
}

// TestZenHTTPProvider_SessionManagement tests session ID handling
func TestZenHTTPProvider_SessionManagement(t *testing.T) {
	provider := NewZenHTTPProviderWithModel("big-pickle")

	// Initially no session
	assert.Empty(t, provider.sessionID)

	// Set a session
	provider.sessionID = "test-session-123"
	assert.Equal(t, "test-session-123", provider.sessionID)
}

// TestZenHTTPProvider_MaxConcurrentRequests tests concurrent request limit
func TestZenHTTPProvider_MaxConcurrentRequests(t *testing.T) {
	provider := NewZenHTTPProviderWithModel("big-pickle")
	caps := provider.GetCapabilities()

	// HTTP supports concurrent requests
	assert.Equal(t, 10, caps.Limits.MaxConcurrentRequests)
}

// TestZenHTTPProvider_ModelSupportViaCapabilities tests model list via capabilities
func TestZenHTTPProvider_ModelSupportViaCapabilities(t *testing.T) {
	provider := NewZenHTTPProviderWithModel("big-pickle")
	caps := provider.GetCapabilities()

	// Should include known Zen free models (as of 2026-02)
	knownModels := []string{
		"big-pickle",
		"gpt-5-nano",
		"glm-4.7",
		"kimi-k2",
		"gemini-3-flash",
		// Note: qwen3-coder removed from free tier 2026-02
	}

	for _, model := range knownModels {
		assert.Contains(t, caps.SupportedModels, model, "Should support model: %s", model)
	}
}

// EnvZenHTTPIntegration opts a run in to the two tests below that send a real
// prompt to a live OpenCode server and therefore reach a real model.
//
// This gate exists because IsOpenCodeInstalled() was fixed. While it was a
// hardcoded false these tests were dead everywhere; making it truthful woke
// them up, and on this host -- which has both the CLI and a server on :4096 --
// `go test ./...` began issuing real model calls with no opt-in at all. The
// unit suite has no business doing that. Un-darkening a test must not turn it
// into an unannounced live probe.
const EnvZenHTTPIntegration = "LLMPROVIDER_ZEN_HTTP_INTEGRATION"

// requireZenHTTPIntegration skips unless the CLI is present AND the caller has
// explicitly asked for live-server tests.
func requireZenHTTPIntegration(t *testing.T) {
	t.Helper()
	if !IsOpenCodeInstalled() {
		t.Skip("opencode CLI is not on PATH") // SKIP-OK: #env
	}
	if os.Getenv(EnvZenHTTPIntegration) == "" {
		t.Skip("set " + EnvZenHTTPIntegration + "=1 to run live OpenCode server tests") // SKIP-OK: #integration-mode-only
	}
}

// Integration test - only runs if OpenCode is installed and server is running
func TestZenHTTPProvider_Integration_Complete(t *testing.T) {
	requireZenHTTPIntegration(t)

	provider := NewZenHTTPProviderWithModel("big-pickle")

	// Check if server is already running
	if !provider.IsServerRunning() {
		t.Skip("OpenCode HTTP server not running - skipping integration test") // SKIP-OK: #integration-mode-only
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	resp, err := provider.Complete(ctx, &models.LLMRequest{
		Prompt: "Reply with exactly one word: hello",
	})

	// This used to turn any error into t.Skip, so the test could report
	// "skipped" but never "failed" -- a live probe that cannot fail is not a
	// probe. The caller has explicitly opted in to reaching a live server, so
	// an error here is a result, not a reason to look away.
	require.NoError(t, err, "opted-in live completion must succeed")

	require.NotNil(t, resp)
	assert.NotEmpty(t, resp.Content)
	assert.Equal(t, "zen-http", resp.ProviderName)
	t.Logf("Response: %s", resp.Content)
	t.Logf("Session ID: %s", resp.Metadata["session_id"])
}

// TestZenHTTPProvider_HealthCheckNoServer asserts the other half of
// HealthCheck's contract on EVERY host, not just on one that happens to have
// no server running.
//
// It exists because a mutation proved the gap: replacing HealthCheck's
// "not running" error with `return nil` was NOT caught by
// TestZenHTTPProvider_Integration_HealthCheck, because this host has an
// OpenCode server on :4096 and that test therefore only ever took the
// server-is-running branch. A test whose reachable branch depends on the host
// asserts nothing about the other one. Pointing at a closed port makes the
// no-server case constructible anywhere, with no CLI and no skip.
func TestZenHTTPProvider_HealthCheckNoServer(t *testing.T) {
	config := DefaultZenHTTPConfig()
	config.BaseURL = "http://127.0.0.1:1"
	config.AutoStart = false
	provider := NewZenHTTPProvider(config)

	require.False(t, provider.IsServerRunning(),
		"a closed port must not look like a running server")

	err := provider.HealthCheck()
	require.Error(t, err,
		"HealthCheck must fail when no server is running and AutoStart is off")
	assert.Contains(t, err.Error(), "not running")
}

// TestZenHTTPProvider_Integration_HealthCheck asserts HealthCheck's contract
// against whatever server state the host is in.
//
// Two things were wrong here. It asserted NOTHING -- both branches were a
// t.Log, so it was a verdict with no content. And it built its provider from
// the default config, whose AutoStart is true: HealthCheck() spawns
// `opencode serve` when no server is running, so once IsOpenCodeInstalled()
// started answering truthfully this test would have LAUNCHED A DAEMON as a
// side effect of `go test ./...`. AutoStart is therefore disabled explicitly:
// this test observes the server, it never creates one.
func TestZenHTTPProvider_Integration_HealthCheck(t *testing.T) {
	if !IsOpenCodeInstalled() {
		t.Skip("opencode CLI is not on PATH") // SKIP-OK: #env
	}

	config := DefaultZenHTTPConfig()
	config.Model = "big-pickle"
	config.AutoStart = false
	provider := NewZenHTTPProvider(config)

	// With AutoStart off the contract is exact: HealthCheck succeeds if and
	// only if a server is already running. (Both branches probe the same
	// endpoint moments apart; a server that stops between the two calls would
	// be a genuine environment change, not a masked defect.)
	running := provider.IsServerRunning()
	err := provider.HealthCheck()

	if running {
		assert.NoError(t, err,
			"HealthCheck must succeed while the server is running")
		return
	}
	require.Error(t, err,
		"HealthCheck must report an error when no server is running and AutoStart is off")
	assert.Contains(t, err.Error(), "not running")
}

// TestZenHTTPProvider_CompleteStream tests streaming completion
func TestZenHTTPProvider_CompleteStream(t *testing.T) {
	requireZenHTTPIntegration(t)

	provider := NewZenHTTPProviderWithModel("big-pickle")

	if !provider.IsServerRunning() {
		t.Skip("no OpenCode server is running at the configured base URL") // SKIP-OK: #integration-mode-only
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	ch, err := provider.CompleteStream(ctx, &models.LLMRequest{
		Prompt: "Say hello",
	})

	// As above: an error used to become a skip, so this test could not fail.
	require.NoError(t, err, "opted-in live streaming must succeed")

	require.NotNil(t, ch)

	// Read from channel
	for resp := range ch {
		t.Logf("Streamed response: %s", resp.Content)
		if resp.FinishReason == "stop" {
			break
		}
	}
}

// TestZenHTTPProvider_AutoStartDisabled tests behavior with auto-start disabled
func TestZenHTTPProvider_AutoStartDisabled(t *testing.T) {
	config := ZenHTTPConfig{
		AutoStart: false,
		Model:     "big-pickle",
	}
	provider := NewZenHTTPProvider(config)

	assert.False(t, provider.autoStart)

	// Health check should fail if server not running and auto-start is disabled
	if !provider.IsServerRunning() {
		err := provider.HealthCheck()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not running")
	}
}

// TestZenHTTPProvider_BasicAuthCredentials tests authentication configuration
func TestZenHTTPProvider_BasicAuthCredentials(t *testing.T) {
	config := ZenHTTPConfig{
		Username: "myuser",
		Password: "mypassword",
	}
	provider := NewZenHTTPProvider(config)

	assert.Equal(t, "myuser", provider.username)
	assert.Equal(t, "mypassword", provider.password)
}

// TestZenHTTPProvider_StartServerWithoutCLI tests server start failure when CLI missing
// TestZenHTTPProvider_StartServerWithoutCLI no longer skips on a host that HAS
// opencode -- it constructs the missing-CLI scenario instead of waiting for a
// host that happens to be in it. PATH is emptied so LookPath must fail, and the
// base URL points at a closed port so StartServer's own "is it already
// running?" short-circuit cannot return nil ahead of the LookPath check.
func TestZenHTTPProvider_StartServerWithoutCLI(t *testing.T) {
	withOpenCodeOnPath(t, false, 0o700)

	config := DefaultZenHTTPConfig()
	config.Model = "big-pickle"
	config.BaseURL = "http://127.0.0.1:1"
	config.AutoStart = false
	provider := NewZenHTTPProvider(config)

	require.False(t, provider.IsServerRunning(),
		"the closed-port base URL must make the running-server short-circuit false")

	err := provider.StartServer()

	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

// TestZenHTTPProvider_MetadataInResponse tests metadata fields in response
func TestZenHTTPProvider_MetadataInResponse(t *testing.T) {
	// Test that response metadata has expected fields
	expectedMetadataKeys := []string{
		"source",
		"session_id",
		"message_id",
		"model",
		"base_url",
		"prompt_tokens",
		"completion_tokens",
		"latency",
	}

	// Create a mock response to verify structure
	resp := &models.LLMResponse{
		Metadata: map[string]interface{}{
			"source":            "opencode-http",
			"session_id":        "test-session",
			"message_id":        "msg-123",
			"model":             "big-pickle",
			"base_url":          "http://localhost:4096",
			"prompt_tokens":     100,
			"completion_tokens": 50,
			"latency":           "1.5s",
		},
	}

	for _, key := range expectedMetadataKeys {
		_, ok := resp.Metadata[key]
		assert.True(t, ok, "Response metadata should contain key: %s", key)
	}
}

// TestZenHTTPProvider_Timeout tests timeout configuration
func TestZenHTTPProvider_Timeout(t *testing.T) {
	config := ZenHTTPConfig{
		Timeout: 30 * time.Second,
	}
	provider := NewZenHTTPProvider(config)

	assert.Equal(t, 30*time.Second, provider.timeout)
	assert.NotNil(t, provider.httpClient)
	assert.Equal(t, 30*time.Second, provider.httpClient.Timeout)
}

// TestZenHTTPProvider_MaxTokens tests max tokens configuration
func TestZenHTTPProvider_MaxTokens(t *testing.T) {
	config := ZenHTTPConfig{
		MaxTokens: 2048,
	}
	provider := NewZenHTTPProvider(config)

	assert.Equal(t, 2048, provider.maxTokens)

	caps := provider.GetCapabilities()
	assert.Equal(t, 2048, caps.Limits.MaxTokens)
}
