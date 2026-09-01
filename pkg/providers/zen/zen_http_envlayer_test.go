package zen

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"digital.vasic.llmprovider/pkg/settings"
)

// The F24 gate for this adapter. `http://localhost:4096` was a frozen literal
// in three places — the struct's doc comment, DefaultZenHTTPConfig, and
// NewZenHTTPProvider's own empty-check — with no environment layer at all,
// even though this same file already read OPENCODE_SERVER_PASSWORD from the
// environment two fields away. An opencode server on another host or port is
// an ordinary operator change; it must not require editing this file.

func clearZenEnv(t *testing.T) {
	t.Helper()
	t.Setenv(settings.Key("zen", settings.SuffixBaseURL), "")
	t.Setenv(settings.Key("zen", settings.SuffixModel), "")
	t.Setenv(settings.Key("zen", settings.SuffixTimeout), "")
	t.Setenv(EnvZenBaseURLAlias, "")
}

func TestZenHTTPDefaultsWhenTheEnvironmentIsClear(t *testing.T) {
	clearZenEnv(t)
	cfg := DefaultZenHTTPConfig()
	assert.Equal(t, DefaultZenBaseURL, cfg.BaseURL)
	assert.Equal(t, DefaultZenModel, cfg.Model)
	assert.Equal(t, DefaultZenTimeout, cfg.Timeout)
	assert.Equal(t, DefaultZenUsername, cfg.Username)
	assert.Equal(t, DefaultZenMaxTokens, cfg.MaxTokens)
}

func TestZenHTTPEndpointIsOverridable(t *testing.T) {
	clearZenEnv(t)
	t.Setenv(settings.Key("zen", settings.SuffixBaseURL), "http://inference.internal:4096")
	assert.Equal(t, "http://inference.internal:4096", DefaultZenHTTPConfig().BaseURL)
}

func TestZenHTTPAcceptsOpencodesOwnEndpointVariable(t *testing.T) {
	clearZenEnv(t)
	t.Setenv(EnvZenBaseURLAlias, "http://from-opencode:9999")
	assert.Equal(t, "http://from-opencode:9999", DefaultZenHTTPConfig().BaseURL)

	// …and this module's own convention outranks the alias when both are set.
	t.Setenv(settings.Key("zen", settings.SuffixBaseURL), "http://canonical:1")
	assert.Equal(t, "http://canonical:1", DefaultZenHTTPConfig().BaseURL)
}

func TestZenHTTPModelAndTimeoutAreOverridable(t *testing.T) {
	clearZenEnv(t)
	t.Setenv(settings.Key("zen", settings.SuffixModel), "operator-chosen-model")
	t.Setenv(settings.Key("zen", settings.SuffixTimeout), "9m")
	cfg := DefaultZenHTTPConfig()
	assert.Equal(t, "operator-chosen-model", cfg.Model)
	assert.Equal(t, 9*time.Minute, cfg.Timeout)
}

func TestNewZenHTTPProviderFillsBlanksFromTheEnvironment(t *testing.T) {
	// The constructor has its own empty-checks, separate from
	// DefaultZenHTTPConfig. A caller who builds a ZenHTTPConfig by hand and
	// leaves fields zero must get the same environment layer — before this,
	// that path re-froze the literals a second time.
	clearZenEnv(t)
	t.Setenv(settings.Key("zen", settings.SuffixBaseURL), "http://hand-built:4096")
	t.Setenv(settings.Key("zen", settings.SuffixModel), "hand-built-model")
	t.Setenv(settings.Key("zen", settings.SuffixTimeout), "42s")

	p := NewZenHTTPProvider(ZenHTTPConfig{})
	assert.Equal(t, "http://hand-built:4096", p.baseURL)
	assert.Equal(t, "hand-built-model", p.model)
}

func TestZenHTTPExplicitConfigOutranksTheEnvironment(t *testing.T) {
	clearZenEnv(t)
	t.Setenv(settings.Key("zen", settings.SuffixBaseURL), "http://from-env:1")
	p := NewZenHTTPProvider(ZenHTTPConfig{BaseURL: "http://explicit:2", Model: "m"})
	assert.Equal(t, "http://explicit:2", p.baseURL)
}
