package gemini

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// The variables an operator actually exports, written out in full rather than
// recomputed with settings.Key(). A test that derives its expectation from the
// same helper its subject uses agrees with that subject by construction and
// would stay green through a silent rename of every documented variable. The
// convention is pinned independently by pkg/settings/keys_contract_test.go.
const (
	envModelKey        = "LLMPROVIDER_GEMINI_MODEL"
	envBaseURLKey      = "LLMPROVIDER_GEMINI_BASE_URL"
	envTimeoutKey      = "LLMPROVIDER_GEMINI_TIMEOUT"
	envModelsURLKey    = "LLMPROVIDER_GEMINI_MODELS_BASE_URL"
	fixtureModel       = "synthetic-model-a"
	fixtureBaseURL     = "https://endpoint.invalid/v1/models/%s:generateContent"
	fixtureModelsURL   = "https://catalogue.invalid/v1/models"
	fixtureTimeoutText = "13s"
)

const fixtureTimeout = 13 * time.Second

func clearEnv(t *testing.T) {
	t.Helper()
	t.Setenv(envModelKey, "")
	t.Setenv(envBaseURLKey, "")
	t.Setenv(envTimeoutKey, "")
	t.Setenv(envModelsURLKey, "")
}

func TestUnifiedDefaultsAreOverridableFromTheEnvironment(t *testing.T) {
	clearEnv(t)
	c := DefaultGeminiUnifiedConfig()
	assert.Equal(t, GeminiDefaultModel, c.Model, "with nothing set the compiled fallback must still apply")
	assert.Equal(t, DefaultUnifiedTimeout, c.Timeout)

	t.Setenv(envModelKey, fixtureModel)
	t.Setenv(envTimeoutKey, fixtureTimeoutText)
	c = DefaultGeminiUnifiedConfig()
	assert.Equal(t, fixtureModel, c.Model, "%s did not reach DefaultGeminiUnifiedConfig", envModelKey)
	assert.Equal(t, fixtureTimeout, c.Timeout, "%s did not reach DefaultGeminiUnifiedConfig", envTimeoutKey)
}

// THE REGRESSION THIS FILE EXISTS FOR.
//
// NewGeminiProvider and NewGeminiProviderWithRetry each built a
// GeminiUnifiedConfig with `Timeout: 180 * time.Second` written into the struct
// literal. That is a POSITIONAL bypass: the value is non-zero, so
// NewGeminiUnifiedProvider's own `if config.Timeout == 0` check could never
// fire, and LLMPROVIDER_GEMINI_TIMEOUT was honoured on every construction path
// except the two backward-compatible ones. The override worked wherever anyone
// checked, and was frozen where nobody did.
func TestBackwardCompatibleConstructorsDoNotBypassTheOverride(t *testing.T) {
	clearEnv(t)
	assert.Equal(t, DefaultUnifiedTimeout, NewGeminiProvider("k", "", "").timeout,
		"with nothing set the compiled fallback must still apply")

	t.Setenv(envTimeoutKey, fixtureTimeoutText)
	assert.Equal(t, fixtureTimeout, NewGeminiProvider("k", "", "").timeout,
		"%s did not reach NewGeminiProvider - the struct literal is frozen again", envTimeoutKey)
	assert.Equal(t, fixtureTimeout, NewGeminiProviderWithRetry("k", "", "", DefaultRetryConfig()).timeout,
		"%s did not reach NewGeminiProviderWithRetry", envTimeoutKey)
}

// A caller-built config with no Model must still get one, and must get the
// operator's if one is set.
func TestUnifiedConstructorResolvesAModelForACallerBuiltConfig(t *testing.T) {
	clearEnv(t)
	assert.Equal(t, GeminiDefaultModel, NewGeminiUnifiedProvider(GeminiUnifiedConfig{}).model)

	t.Setenv(envModelKey, fixtureModel)
	assert.Equal(t, fixtureModel, NewGeminiUnifiedProvider(GeminiUnifiedConfig{}).model,
		"%s did not reach NewGeminiUnifiedProvider", envModelKey)
}

func TestAPITransportDefaultsAreOverridable(t *testing.T) {
	clearEnv(t)
	p := NewGeminiAPIProvider("k", "", "")
	assert.Equal(t, GeminiAPIURL, p.baseURL)
	assert.Equal(t, GeminiDefaultModel, p.model)
	assert.Equal(t, DefaultHTTPTimeout, p.httpClient.Timeout)

	t.Setenv(envBaseURLKey, fixtureBaseURL)
	t.Setenv(envModelKey, fixtureModel)
	t.Setenv(envTimeoutKey, fixtureTimeoutText)
	p = NewGeminiAPIProvider("k", "", "")
	assert.Equal(t, fixtureBaseURL, p.baseURL, "%s did not reach the transport", envBaseURLKey)
	assert.Equal(t, fixtureModel, p.model, "%s did not reach the transport", envModelKey)
	assert.Equal(t, fixtureTimeout, p.httpClient.Timeout, "%s did not reach the transport", envTimeoutKey)
}

// The models-listing endpoint is keyed APART from the generateContent endpoint.
// Folding them together would mean an operator pointing the adapter at a proxy
// silently redirected health checks to a URL that does not serve a catalogue —
// so setting one must not move the other.
func TestModelsEndpointIsKeyedApartFromTheCompletionEndpoint(t *testing.T) {
	clearEnv(t)
	t.Setenv(envBaseURLKey, fixtureBaseURL)
	assert.Equal(t, fixtureBaseURL, NewGeminiAPIProvider("k", "", "").baseURL)

	t.Setenv(envModelsURLKey, fixtureModelsURL)
	assert.Equal(t, fixtureBaseURL, NewGeminiAPIProvider("k", "", "").baseURL,
		"%s must not move the completion endpoint", envModelsURLKey)
}

func TestUnparseableTimeoutFallsBackInsteadOfDisablingTheTimeout(t *testing.T) {
	clearEnv(t)
	for _, bad := range []string{"soon", "0s", "-5s", "180"} {
		t.Setenv(envTimeoutKey, bad)
		assert.Equal(t, DefaultHTTPTimeout, NewGeminiAPIProvider("k", "", "").httpClient.Timeout,
			"%s=%q must fall back, never disable the timeout", envTimeoutKey, bad)
	}
}
