package generic

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// THE POINT OF THIS ADAPTER'S SETTINGS KEY.
//
// This package is a shim in front of many OpenAI-compatible backends, and the
// backend's identity arrives as the `name` argument. So the timeout is keyed by
// THAT name, not by the literal "generic": a single LLMPROVIDER_GENERIC_TIMEOUT
// would tie every backend reached through the shim to one wait, which is the
// same "one frozen decision for everybody" defect the settings layer exists to
// remove — just relocated from a compiled constant to a single variable.
//
// The keys below are written out in full rather than recomputed with
// settings.Key(), because a test that derives its expectation from the helper
// its subject uses agrees with that subject by construction. The convention is
// pinned independently by pkg/settings/keys_contract_test.go.
const (
	envNvidiaTimeoutKey  = "LLMPROVIDER_NVIDIA_TIMEOUT"
	envUpstageTimeoutKey = "LLMPROVIDER_UPSTAGE_TIMEOUT"
	envGenericTimeoutKey = "LLMPROVIDER_GENERIC_TIMEOUT"
)

func TestTimeoutIsKeyedByTheCallersProviderName(t *testing.T) {
	t.Setenv(envNvidiaTimeoutKey, "")
	t.Setenv(envUpstageTimeoutKey, "")

	assert.Equal(t, DefaultTimeout,
		NewGenericProvider("nvidia", "k", "https://endpoint.invalid/v1", "synthetic-model-a").httpClient.Timeout,
		"with nothing set the compiled fallback must still apply")

	t.Setenv(envNvidiaTimeoutKey, "23s")
	assert.Equal(t, 23*time.Second,
		NewGenericProvider("nvidia", "k", "https://endpoint.invalid/v1", "synthetic-model-a").httpClient.Timeout,
		"%s did not reach the constructor", envNvidiaTimeoutKey)

	// The load-bearing half: one backend's variable must not move another's.
	assert.Equal(t, DefaultTimeout,
		NewGenericProvider("upstage", "k", "https://endpoint.invalid/v1", "synthetic-model-a").httpClient.Timeout,
		"%s leaked into a different backend - the key is not per-provider", envNvidiaTimeoutKey)
}

// A guard against the regression that would look like a simplification:
// collapsing the caller's name back into a single literal key.
func TestThereIsNoSingleGenericTimeoutKey(t *testing.T) {
	t.Setenv(envNvidiaTimeoutKey, "")
	t.Setenv(envGenericTimeoutKey, "29s")

	assert.Equal(t, DefaultTimeout,
		NewGenericProvider("nvidia", "k", "https://endpoint.invalid/v1", "synthetic-model-a").httpClient.Timeout,
		"%s must have no effect - this shim keys by the caller's provider name", envGenericTimeoutKey)
}

// baseURL and model are NOT resolved through settings here, on purpose: this
// constructor has no compiled default for either, so there is nothing to
// unfreeze. Adding a resolution would invent a default where the API
// deliberately has none, and a caller that passed an empty string would
// silently start talking somewhere. Pin that, so the "consistency" refactor
// that adds it has to argue with a test.
func TestEndpointAndModelComeOnlyFromTheCaller(t *testing.T) {
	t.Setenv("LLMPROVIDER_NVIDIA_BASE_URL", "https://leaked.invalid/v1")
	t.Setenv("LLMPROVIDER_NVIDIA_MODEL", "leaked-model")

	p := NewGenericProvider("nvidia", "k", "", "")
	assert.Equal(t, "", p.baseURL, "this constructor has no endpoint default and must not acquire one")
	assert.Equal(t, "", p.model, "this constructor has no model default and must not acquire one")
}
