package zen

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The env-var names are written out as LITERALS here, on purpose.
//
// The obvious way to write this file is `settings.Key(SettingsProviderAPI,
// settings.SuffixBaseURL)` — and it is wrong. That expression is the very
// computation the constructor performs, so the test would agree with its
// subject BY CONSTRUCTION: rename the provider key, or break normalize(), and
// both sides move together and the assertion still passes while every operator
// who exported the documented variable is silently ignored. The whole point of
// this layer is the name an operator types into a shell, so the name an
// operator types is what gets asserted. If someone changes the key, these
// constants must be edited by hand — that edit is the review moment.
const (
	envZenAPIBaseURLKey = "LLMPROVIDER_ZEN_API_BASE_URL"
	envZenAPITimeoutKey = "LLMPROVIDER_ZEN_API_TIMEOUT"
	envZenModelKey      = "LLMPROVIDER_ZEN_MODEL"
)

// Fixture values are SYNTHETIC and deliberately unusable.
//
// A previous round of this work seeded fixtures with real model ids and real
// endpoints, and those fixtures became the next audit's findings — a test that
// freezes a vendor string is the same defect as the code it is testing, just
// harder to notice. `.invalid` is reserved by RFC 2606 and can never resolve,
// and the model ids below name nothing that exists.
const (
	synthZenBaseURL = "https://zen-endpoint.invalid/v1/chat"
	synthZenTimeout = "7s"
	synthZenModelID = "synthetic-model-alpha"
	synthZenPaidID  = "synthetic-model-not-free"
)

// TestZenAPI_EnvOverridesEndpointModelAndTimeout is the F23/F24 contract for
// the hosted-API transport: with nothing passed by the caller, the environment
// decides, and the compiled constants are only the last resort.
func TestZenAPI_EnvOverridesEndpointModelAndTimeout(t *testing.T) {
	t.Setenv(envZenAPIBaseURLKey, synthZenBaseURL)
	t.Setenv(envZenModelKey, synthZenModelID)
	t.Setenv(envZenAPITimeoutKey, synthZenTimeout)

	p := NewZenProviderWithRetry("", "", "", DefaultRetryConfig())
	require.NotNil(t, p)

	assert.Equal(t, synthZenBaseURL, p.baseURL,
		"%s must decide the endpoint when the caller passes none", envZenAPIBaseURLKey)
	assert.Equal(t, synthZenModelID, p.model,
		"%s must decide the model when the caller passes none", envZenModelKey)
	assert.Equal(t, 7*time.Second, p.httpClient.Timeout,
		"%s must decide the HTTP timeout", envZenAPITimeoutKey)
}

// TestZenAPI_CompiledFallbacksWhenEnvUnset is the other half of the contract.
// Without it, a resolver that returned the env value unconditionally — or one
// that returned the empty string — would still pass the test above.
func TestZenAPI_CompiledFallbacksWhenEnvUnset(t *testing.T) {
	t.Setenv(envZenAPIBaseURLKey, "")
	t.Setenv(envZenModelKey, "")
	t.Setenv(envZenAPITimeoutKey, "")

	p := NewZenProviderWithRetry("", "", "", DefaultRetryConfig())
	require.NotNil(t, p)

	assert.Equal(t, ZenAPIURL, p.baseURL)
	assert.Equal(t, DefaultZenModel, p.model)
	assert.Equal(t, DefaultZenAPITimeout, p.httpClient.Timeout)
}

// TestZenAPI_ExplicitArgumentBeatsEnv pins the documented precedence. An
// override layer that silently outranked an explicit argument would be a new
// bug wearing the fix's clothes.
func TestZenAPI_ExplicitArgumentBeatsEnv(t *testing.T) {
	t.Setenv(envZenAPIBaseURLKey, synthZenBaseURL)
	t.Setenv(envZenModelKey, synthZenModelID)

	const callerURL = "https://caller-supplied.invalid/v1"
	p := NewZenProviderWithRetry("", callerURL, ModelGLM5Free, DefaultRetryConfig())
	require.NotNil(t, p)

	assert.Equal(t, callerURL, p.baseURL, "an explicit argument outranks %s", envZenAPIBaseURLKey)
	assert.Equal(t, ModelGLM5Free, p.model, "an explicit argument outranks %s", envZenModelKey)
}

// TestZenAnonymous_HonoursEndpointOverride is the regression test for the worst
// shape this defect took. NewZenProviderAnonymous used to hand ZenAPIURL to
// NewZenProviderWithRetry POSITIONALLY, which skipped that constructor's own
// empty-check entirely — so the endpoint override worked on every construction
// path except this one, silently. Fixing only the empty-check would have left
// this path frozen with the other tests still green.
func TestZenAnonymous_HonoursEndpointOverride(t *testing.T) {
	t.Setenv(envZenAPIBaseURLKey, synthZenBaseURL)
	// A KNOWN-free model, so the free-only clamp below is not what is being
	// measured here. Referenced as a constant rather than spelled out, so this
	// fixture introduces no frozen model literal of its own.
	t.Setenv(envZenModelKey, ModelGLM5Free)

	p := NewZenProviderAnonymous("")
	require.NotNil(t, p)

	assert.Equal(t, synthZenBaseURL, p.baseURL,
		"the anonymous constructor must honour %s, not the compiled ZenAPIURL", envZenAPIBaseURLKey)
	assert.Equal(t, ModelGLM5Free, p.model)
	assert.True(t, p.anonymousMode, "no API key and a free model means anonymous mode")
}

// TestZenAnonymous_FreeOnlyClampIgnoresEnv guards the one place in this file
// that deliberately does NOT consult the environment. The clamp exists to stop
// an anonymous caller being pointed at a billable model; resolving it through
// settings would let the variable that caused the violation also pick the
// remedy, so a non-free LLMPROVIDER_ZEN_MODEL would simply land back on itself.
func TestZenAnonymous_FreeOnlyClampIgnoresEnv(t *testing.T) {
	t.Setenv(envZenModelKey, synthZenPaidID)

	p := NewZenProviderAnonymous("")
	require.NotNil(t, p)

	assert.NotEqual(t, synthZenPaidID, p.model,
		"a non-free %s must never survive into an anonymous provider", envZenModelKey)
	assert.Equal(t, DefaultZenModel, p.model,
		"the clamp must fall back to the compiled free model, not to the environment")
	assert.True(t, isFreeModel(p.model), "the free-only invariant must hold")
}

// TestZenCapabilities_ReportTheEndpointInForce covers the metadata that used to
// report the compiled constant. A diagnostic that names the wrong host sends
// the reader to the wrong machine, which is worse than saying nothing.
func TestZenCapabilities_ReportTheEndpointInForce(t *testing.T) {
	t.Setenv(envZenAPIBaseURLKey, synthZenBaseURL)

	p := NewZenProviderWithRetry("", "", ModelGLM5Free, DefaultRetryConfig())
	require.NotNil(t, p)

	caps := p.GetCapabilities()
	require.NotNil(t, caps)
	assert.Equal(t, synthZenBaseURL, caps.Metadata["base_url"],
		"GetCapabilities must report the endpoint actually in force")
}
