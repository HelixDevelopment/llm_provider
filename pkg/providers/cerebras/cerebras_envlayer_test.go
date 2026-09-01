package cerebras

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"digital.vasic.llmprovider/pkg/settings"
)

// The F23 gate for this adapter: the compiled default model must be a FALLBACK,
// not a frozen decision. Before pkg/settings existed, `if model == "" { model =
// CerebrasModel }` had nothing behind it, so a vendor retiring or rate-capping
// that model could only be answered by editing this file and cutting a release.
//
// Both halves are asserted. A rewiring that ALWAYS reads the environment would
// break every caller relying on the documented default, so the first half
// pins the fallback; the second proves the override reaches the constructor.
func TestDefaultModelIsOverridableFromTheEnvironment(t *testing.T) {
	key := settings.Key("cerebras", settings.SuffixModel)

	t.Setenv(key, "")
	assert.Equal(t, CerebrasModel, NewCerebrasProvider("k", "", "").model,
		"with nothing set the compiled fallback must still apply")

	t.Setenv(key, "operator-chosen-model")
	assert.Equal(t, "operator-chosen-model", NewCerebrasProvider("k", "", "").model,
		"%s did not reach the constructor - the default is still frozen", key)
}

func TestExplicitModelOutranksTheEnvironment(t *testing.T) {
	t.Setenv(settings.Key("cerebras", settings.SuffixModel), "from-env")
	assert.Equal(t, "caller-model", NewCerebrasProvider("k", "", "caller-model").model,
		"an explicit argument must outrank the environment")
}
