package junie

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// The variables an operator actually exports, written out in full rather than
// recomputed with settings.Key(): a test that derives its expectation from the
// same helper its subject uses agrees with that subject by construction. The
// convention is pinned independently by pkg/settings/keys_contract_test.go.
const (
	envModelKey   = "LLMPROVIDER_JUNIE_MODEL"
	envTimeoutKey = "LLMPROVIDER_JUNIE_TIMEOUT"
)

const fixtureModel = "synthetic-model-a"

func clearEnv(t *testing.T) {
	t.Helper()
	t.Setenv(envModelKey, "")
	t.Setenv(envTimeoutKey, "")
}

func TestDefaultConfigIsOverridableFromTheEnvironment(t *testing.T) {
	clearEnv(t)
	c := DefaultJunieConfig()
	assert.Equal(t, DefaultJunieModel, c.Model, "with nothing set the compiled fallback must still apply")
	assert.Equal(t, DefaultJunieTimeout, c.Timeout)

	t.Setenv(envModelKey, fixtureModel)
	t.Setenv(envTimeoutKey, "31s")
	c = DefaultJunieConfig()
	assert.Equal(t, fixtureModel, c.Model, "%s did not reach DefaultJunieConfig", envModelKey)
	assert.Equal(t, 31*time.Second, c.Timeout, "%s did not reach DefaultJunieConfig", envTimeoutKey)
}

// A caller that builds a JunieConfig itself never passes through
// DefaultJunieConfig, so resolving in only one of the two places is exactly how
// an override comes to work on some construction paths and not others.
//
// The Model half also fixes a defect that was worse than a frozen default:
// there was no empty-check for Model at all, so a caller-built config with no
// Model produced a provider with an EMPTY model string.
func TestCallerBuiltConfigIsResolvedToo(t *testing.T) {
	clearEnv(t)
	p := NewJunieProvider(JunieConfig{})
	assert.Equal(t, DefaultJunieModel, p.model, "an unset model must not stay empty")
	assert.Equal(t, DefaultJunieTimeout, p.timeout)

	t.Setenv(envModelKey, fixtureModel)
	t.Setenv(envTimeoutKey, "37s")
	p = NewJunieProvider(JunieConfig{})
	assert.Equal(t, fixtureModel, p.model, "%s did not reach NewJunieProvider", envModelKey)
	assert.Equal(t, 37*time.Second, p.timeout, "%s did not reach NewJunieProvider", envTimeoutKey)
}

func TestExplicitConfigOutranksTheEnvironment(t *testing.T) {
	clearEnv(t)
	t.Setenv(envModelKey, "from-env")
	t.Setenv(envTimeoutKey, "41s")

	p := NewJunieProvider(JunieConfig{Model: "caller-model", Timeout: 3 * time.Second})
	assert.Equal(t, "caller-model", p.model)
	assert.Equal(t, 3*time.Second, p.timeout)
}

func TestUnparseableTimeoutFallsBackInsteadOfDisablingTheTimeout(t *testing.T) {
	clearEnv(t)
	for _, bad := range []string{"soon", "0s", "-5s", "180"} {
		t.Setenv(envTimeoutKey, bad)
		assert.Equal(t, DefaultJunieTimeout, NewJunieProvider(JunieConfig{}).timeout,
			"%s=%q must fall back, never disable the timeout", envTimeoutKey, bad)
	}
}
