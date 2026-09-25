package xai

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// The variables an operator actually exports, written out in full rather than
// recomputed with settings.Key(). A test that builds its expectation with the
// same helper its subject uses agrees with that subject by construction: change
// the prefix or the folding rule and BOTH sides move together, so the test
// stays green while every documented variable name silently changes. Spelling
// them out is what lets this file come back RED for that defect. The convention
// itself is pinned independently by pkg/settings/keys_contract_test.go.
const (
	envModelKey   = "LLMPROVIDER_XAI_MODEL"
	envBaseURLKey = "LLMPROVIDER_XAI_BASE_URL"
	envTimeoutKey = "LLMPROVIDER_XAI_TIMEOUT"
)

// Fixture values are SYNTHETIC. `.invalid` is reserved by RFC 6761 and can
// never resolve, and the model ids are not any vendor's. A fixture that names a
// real endpoint or a real model id is the same frozen-default defect as the
// code it tests, one layer out — an earlier round of this work created 13 new
// audit findings exactly that way.
const (
	fixtureModel   = "synthetic-model-a"
	fixtureBaseURL = "https://endpoint.invalid/v1"
)

func TestDefaultsAreOverridableFromTheEnvironment(t *testing.T) {
	t.Setenv(envModelKey, "")
	t.Setenv(envBaseURLKey, "")
	t.Setenv(envTimeoutKey, "")

	p := NewProvider("k", "", "")
	assert.Equal(t, XAIAPIBaseURL, p.baseURL, "with nothing set the compiled fallback must still apply")
	assert.Equal(t, DefaultModel, p.model)
	assert.Equal(t, DefaultHTTPTimeout, p.httpClient.Timeout)

	t.Setenv(envModelKey, fixtureModel)
	t.Setenv(envBaseURLKey, fixtureBaseURL)
	t.Setenv(envTimeoutKey, "11s")

	p = NewProvider("k", "", "")
	assert.Equal(t, fixtureBaseURL, p.baseURL, "%s did not reach the constructor", envBaseURLKey)
	assert.Equal(t, fixtureModel, p.model, "%s did not reach the constructor", envModelKey)
	assert.Equal(t, 11*time.Second, p.httpClient.Timeout, "%s did not reach the constructor", envTimeoutKey)
}

// THE REGRESSION THIS FILE EXISTS FOR.
//
// NewProviderWithRegion picks a compiled endpoint from the region and then
// hands it to NewProviderWithRetry POSITIONALLY. That argument is never empty,
// so the `if baseURL == ""` check in the callee cannot fire, and before the fix
// LLMPROVIDER_XAI_BASE_URL was honoured on every construction path EXCEPT this
// one. That is the worst shape of this defect: the override demonstrably works,
// so nobody looks again, and one path stays frozen in silence.
//
// Both regions are asserted, because fixing only the branch an operator happens
// to test leaves the other frozen.
func TestRegionSelectedEndpointIsStillOverridable(t *testing.T) {
	t.Setenv(envBaseURLKey, "")
	assert.Equal(t, XAIAPIBaseURL, NewProviderWithRegion("k", "", "us-east-1").baseURL)
	assert.Equal(t, XAIAPIEUBaseURL, NewProviderWithRegion("k", "", "eu-west-1").baseURL,
		"the region dispatch itself must still choose the EU endpoint")

	t.Setenv(envBaseURLKey, fixtureBaseURL)
	assert.Equal(t, fixtureBaseURL, NewProviderWithRegion("k", "", "us-east-1").baseURL,
		"%s must outrank the region-selected compiled default", envBaseURLKey)
	assert.Equal(t, fixtureBaseURL, NewProviderWithRegion("k", "", "eu-west-1").baseURL,
		"%s must outrank the region-selected compiled default on the EU path too", envBaseURLKey)
}

func TestExplicitArgumentOutranksTheEnvironment(t *testing.T) {
	t.Setenv(envModelKey, "from-env")
	t.Setenv(envBaseURLKey, "https://from-env.invalid/v1")

	p := NewProvider("k", "https://caller.invalid/v1", "caller-model")
	assert.Equal(t, "caller-model", p.model, "an explicit argument must outrank the environment")
	assert.Equal(t, "https://caller.invalid/v1", p.baseURL, "an explicit argument must outrank the environment")
}

// A timeout that does not parse, or is not positive, must fall back rather than
// be honoured literally: http.Client reads a zero Timeout as "wait forever", so
// taking a mistyped value at face value would turn an operator's attempt to
// SHORTEN a wait into an unbounded request.
func TestUnparseableTimeoutFallsBackInsteadOfDisablingTheTimeout(t *testing.T) {
	for _, bad := range []string{"soon", "0s", "-5s", "120"} {
		t.Setenv(envTimeoutKey, bad)
		assert.Equal(t, DefaultHTTPTimeout, NewProvider("k", "", "").httpClient.Timeout,
			"%s=%q must fall back, never disable the timeout", envTimeoutKey, bad)
	}
}
