package codestral

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// The variable an operator actually exports, written out in full rather than
// recomputed with settings.Key(). Deriving the key from the same helper the
// constructor uses would make this test agree with the subject by construction:
// a Key() that silently changed convention would move BOTH sides together and
// these assertions would keep passing while every operator's exported variable
// stopped being read. Spelling it out is what lets this test come back RED for
// that defect instead of quietly green. The convention itself — that Key()
// produces exactly this string — is pinned independently by TestKeyConvention
// in pkg/settings.
const envModelKey = "LLMPROVIDER_CODESTRAL_MODEL"

// The F23 gate for this adapter: the compiled default model must be a FALLBACK,
// not a frozen decision. Before pkg/settings existed, `if model == "" { model =
// CodestralModel }` had nothing behind it, so a vendor retiring or rate-capping
// that model could only be answered by editing this file and cutting a release.
//
// Both halves are asserted. A rewiring that ALWAYS reads the environment would
// break every caller relying on the documented default, so the first half
// pins the fallback; the second proves the override reaches the constructor.
func TestDefaultModelIsOverridableFromTheEnvironment(t *testing.T) {
	key := envModelKey

	t.Setenv(key, "")
	assert.Equal(t, CodestralModel, NewCodestralProvider("k", "", "").model,
		"with nothing set the compiled fallback must still apply")

	t.Setenv(key, "operator-chosen-model")
	assert.Equal(t, "operator-chosen-model", NewCodestralProvider("k", "", "").model,
		"%s did not reach the constructor - the default is still frozen", key)
}

func TestExplicitModelOutranksTheEnvironment(t *testing.T) {
	t.Setenv(envModelKey, "from-env")
	assert.Equal(t, "caller-model", NewCodestralProvider("k", "", "caller-model").model,
		"an explicit argument must outrank the environment")
}
