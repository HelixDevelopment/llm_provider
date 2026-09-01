package claude

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// The variables an operator actually exports, written out in full rather than
// recomputed with settings.Key(). Deriving them from the same helper the
// subject uses would make this file agree with the subject by construction: a
// changed prefix or folding rule moves both sides together and the assertions
// stay green while every documented name changes underneath the operator. The
// convention is pinned independently by pkg/settings/keys_contract_test.go.
const (
	envModelKey      = "LLMPROVIDER_CLAUDE_MODEL"
	envOAuthModelKey = "LLMPROVIDER_CLAUDE_OAUTH_MODEL"
	envBaseURLKey    = "LLMPROVIDER_CLAUDE_BASE_URL"
	envTimeoutKey    = "LLMPROVIDER_CLAUDE_TIMEOUT"
)

// SYNTHETIC fixtures. `.invalid` is reserved by RFC 6761 and can never resolve;
// the model ids are not any vendor's. A fixture naming a real endpoint or model
// is the same frozen-default defect as the code it tests.
const (
	fixtureModel      = "synthetic-model-a"
	fixtureOAuthModel = "synthetic-model-b"
	fixtureBaseURL    = "https://endpoint.invalid/v1/messages"
)

type stubCreds struct{}

func (stubCreds) HasValidClaudeCredentials() bool { return true }
func (stubCreds) GetClaudeAccessToken() (string, error) {
	return "synthetic-token", nil
}

func clearEnv(t *testing.T) {
	t.Helper()
	t.Setenv(envModelKey, "")
	t.Setenv(envOAuthModelKey, "")
	t.Setenv(envBaseURLKey, "")
	t.Setenv(envTimeoutKey, "")
}

func TestAPIKeyPathDefaultsAreOverridable(t *testing.T) {
	clearEnv(t)
	p := NewClaudeProvider("k", "", "")
	assert.Equal(t, ClaudeAPIURL, p.baseURL, "with nothing set the compiled fallback must still apply")
	assert.Equal(t, ClaudeModel, p.model)
	assert.Equal(t, DefaultHTTPTimeout, p.httpClient.Timeout)

	t.Setenv(envBaseURLKey, fixtureBaseURL)
	t.Setenv(envModelKey, fixtureModel)
	t.Setenv(envTimeoutKey, "17s")
	p = NewClaudeProvider("k", "", "")
	assert.Equal(t, fixtureBaseURL, p.baseURL, "%s did not reach the constructor", envBaseURLKey)
	assert.Equal(t, fixtureModel, p.model, "%s did not reach the constructor", envModelKey)
	assert.Equal(t, 17*time.Second, p.httpClient.Timeout, "%s did not reach the constructor", envTimeoutKey)
}

// The two auth paths share a BASE_URL key and do NOT share a MODEL key. That
// split is a decision, not an accident: both transports post to the same
// endpoint, but an OAuth token from the Claude Code CLI is product-restricted,
// so an operator choosing a model for one path must not silently choose it for
// the other. Assert the split in BOTH directions — a one-directional check
// passes just as happily when the two keys have been quietly merged.
func TestTheTwoAuthPathsHaveSeparateModelKeys(t *testing.T) {
	clearEnv(t)
	t.Setenv(envModelKey, fixtureModel)

	assert.Equal(t, fixtureModel, NewClaudeProvider("k", "", "").model)

	oauth, err := NewClaudeProviderWithOAuth("", "", stubCreds{})
	assert.NoError(t, err)
	assert.Equal(t, ClaudeOAuthModel, oauth.model,
		"%s must not move the OAuth path's model", envModelKey)

	t.Setenv(envOAuthModelKey, fixtureOAuthModel)
	oauth, err = NewClaudeProviderWithOAuth("", "", stubCreds{})
	assert.NoError(t, err)
	assert.Equal(t, fixtureOAuthModel, oauth.model, "%s did not reach the OAuth constructor", envOAuthModelKey)
	assert.Equal(t, fixtureModel, NewClaudeProvider("k", "", "").model,
		"%s must not move the API-key path's model", envOAuthModelKey)
}

// The endpoint IS shared, so setting it must move both paths. Asserting only
// the split above would let a change that gave OAuth its own frozen endpoint
// pass unnoticed.
func TestTheTwoAuthPathsShareTheEndpointKey(t *testing.T) {
	clearEnv(t)
	t.Setenv(envBaseURLKey, fixtureBaseURL)

	assert.Equal(t, fixtureBaseURL, NewClaudeProvider("k", "", "").baseURL)
	oauth, err := NewClaudeProviderWithOAuth("", "", stubCreds{})
	assert.NoError(t, err)
	assert.Equal(t, fixtureBaseURL, oauth.baseURL, "%s must reach the OAuth path too", envBaseURLKey)
}

func TestExplicitArgumentOutranksTheEnvironment(t *testing.T) {
	clearEnv(t)
	t.Setenv(envModelKey, "from-env")
	t.Setenv(envBaseURLKey, "https://from-env.invalid/v1")

	p := NewClaudeProvider("k", "https://caller.invalid/v1", "caller-model")
	assert.Equal(t, "caller-model", p.model)
	assert.Equal(t, "https://caller.invalid/v1", p.baseURL)
}

func TestUnparseableTimeoutFallsBackInsteadOfDisablingTheTimeout(t *testing.T) {
	clearEnv(t)
	for _, bad := range []string{"soon", "0s", "-5s", "60"} {
		t.Setenv(envTimeoutKey, bad)
		assert.Equal(t, DefaultHTTPTimeout, NewClaudeProvider("k", "", "").httpClient.Timeout,
			"%s=%q must fall back, never disable the timeout", envTimeoutKey, bad)
	}
}
