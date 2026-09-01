package qwen

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// The variables an operator actually exports, written out in full rather than
// recomputed with settings.Key(): a test that builds its expectation with the
// same helper its subject uses agrees with that subject by construction and
// stays green even when the key is wrong. The convention is pinned
// independently by pkg/settings/keys_contract_test.go.
const (
	envModelKey        = "LLMPROVIDER_QWEN_MODEL"
	envBaseURLKey      = "LLMPROVIDER_QWEN_BASE_URL"
	envOAuthBaseURLKey = "LLMPROVIDER_QWEN_OAUTH_BASE_URL"
	envTimeoutKey      = "LLMPROVIDER_QWEN_TIMEOUT"
)

// SYNTHETIC fixtures — `.invalid` is reserved by RFC 6761 and can never
// resolve, and the model id is not any vendor's.
const (
	fixtureModel        = "synthetic-model-a"
	fixtureBaseURL      = "https://endpoint.invalid/api/v1"
	fixtureOAuthBaseURL = "https://oauth-endpoint.invalid/compatible/v1"
)

func clearEnv(t *testing.T) {
	t.Helper()
	t.Setenv(envModelKey, "")
	t.Setenv(envBaseURLKey, "")
	t.Setenv(envOAuthBaseURLKey, "")
	t.Setenv(envTimeoutKey, "")
}

func TestAPIKeyPathDefaultsAreOverridable(t *testing.T) {
	clearEnv(t)
	p := NewQwenProvider("k", "", "")
	assert.Equal(t, QwenAPIURL, p.baseURL, "with nothing set the compiled fallback must still apply")
	assert.Equal(t, QwenModel, p.model)
	assert.Equal(t, DefaultHTTPTimeout, p.httpClient.Timeout)

	t.Setenv(envBaseURLKey, fixtureBaseURL)
	t.Setenv(envModelKey, fixtureModel)
	t.Setenv(envTimeoutKey, "19s")
	p = NewQwenProvider("k", "", "")
	assert.Equal(t, fixtureBaseURL, p.baseURL, "%s did not reach the constructor", envBaseURLKey)
	assert.Equal(t, fixtureModel, p.model, "%s did not reach the constructor", envModelKey)
	assert.Equal(t, 19*time.Second, p.httpClient.Timeout, "%s did not reach the constructor", envTimeoutKey)
}

// The two auth paths do NOT talk to the same endpoint — the API-key path posts
// to DashScope's native API, the OAuth path to its OpenAI-compatible shim — so
// they carry two BASE_URL keys. One shared key would mean an operator
// redirecting OAuth traffic silently redirected API-key traffic to a URL that
// speaks a different wire format. Assert the split in BOTH directions; a
// one-directional check passes just as happily when the keys have been merged.
func TestTheTwoAuthPathsHaveSeparateEndpointKeys(t *testing.T) {
	clearEnv(t)
	t.Setenv(envBaseURLKey, fixtureBaseURL)

	assert.Equal(t, fixtureBaseURL, NewQwenProvider("k", "", "").baseURL)

	oauth, err := NewQwenProviderWithOAuth("", "")
	assert.NoError(t, err)
	assert.Equal(t, QwenOAuthAPIURL, oauth.baseURL,
		"%s must not move the OAuth path's endpoint", envBaseURLKey)

	t.Setenv(envOAuthBaseURLKey, fixtureOAuthBaseURL)
	oauth, err = NewQwenProviderWithOAuth("", "")
	assert.NoError(t, err)
	assert.Equal(t, fixtureOAuthBaseURL, oauth.baseURL, "%s did not reach the OAuth constructor", envOAuthBaseURLKey)
	assert.Equal(t, fixtureBaseURL, NewQwenProvider("k", "", "").baseURL,
		"%s must not move the API-key path's endpoint", envOAuthBaseURLKey)
}

// The MODEL genuinely IS shared between the two transports, so setting it must
// move both. Asserting only the endpoint split would let a change that gave the
// OAuth path its own frozen model pass unnoticed.
func TestTheTwoAuthPathsShareTheModelKey(t *testing.T) {
	clearEnv(t)
	t.Setenv(envModelKey, fixtureModel)

	assert.Equal(t, fixtureModel, NewQwenProvider("k", "", "").model)
	oauth, err := NewQwenProviderWithOAuth("", "")
	assert.NoError(t, err)
	assert.Equal(t, fixtureModel, oauth.model, "%s must reach the OAuth path too", envModelKey)
}

func TestExplicitArgumentOutranksTheEnvironment(t *testing.T) {
	clearEnv(t)
	t.Setenv(envModelKey, "from-env")
	t.Setenv(envBaseURLKey, "https://from-env.invalid/v1")

	p := NewQwenProvider("k", "https://caller.invalid/v1", "caller-model")
	assert.Equal(t, "caller-model", p.model)
	assert.Equal(t, "https://caller.invalid/v1", p.baseURL)
}

func TestUnparseableTimeoutFallsBackInsteadOfDisablingTheTimeout(t *testing.T) {
	clearEnv(t)
	for _, bad := range []string{"soon", "0s", "-5s", "60"} {
		t.Setenv(envTimeoutKey, bad)
		assert.Equal(t, DefaultHTTPTimeout, NewQwenProvider("k", "", "").httpClient.Timeout,
			"%s=%q must fall back, never disable the timeout", envTimeoutKey, bad)
	}
}
