package settings

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestKeyConvention(t *testing.T) {
	assert.Equal(t, "LLMPROVIDER_OLLAMA_MODEL", Key("ollama", SuffixModel))
	assert.Equal(t, "LLMPROVIDER_ZEN_HTTP_BASE_URL", Key("zen_http", SuffixBaseURL))
	assert.Equal(t, "LLMPROVIDER_ZEN_HTTP_TIMEOUT", Key("Zen HTTP", SuffixTimeout))
	assert.Equal(t, "LLMPROVIDER_AI21_MODEL", Key("ai21", SuffixModel))
}

func TestFallbackWhenNothingIsSet(t *testing.T) {
	t.Setenv(Key("cerebras", SuffixModel), "")
	assert.Equal(t, "llama3.1-8b", Model("cerebras", "llama3.1-8b"))
}

func TestCanonicalKeyWins(t *testing.T) {
	t.Setenv(Key("cerebras", SuffixModel), "from-canonical")
	assert.Equal(t, "from-canonical", Model("cerebras", "compiled-default"))
}

func TestAliasIsConsultedAfterTheCanonicalKey(t *testing.T) {
	t.Setenv(Key("ollama", SuffixBaseURL), "")
	t.Setenv("OLLAMA_HOST", "http://from-alias:11434")
	assert.Equal(t, "http://from-alias:11434",
		BaseURL("ollama", "http://localhost:11434", "OLLAMA_HOST"))

	// And the module's own convention outranks the alias when both are set.
	t.Setenv(Key("ollama", SuffixBaseURL), "http://from-canonical:1234")
	assert.Equal(t, "http://from-canonical:1234",
		BaseURL("ollama", "http://localhost:11434", "OLLAMA_HOST"))
}

func TestEmptyCountsAsUnset(t *testing.T) {
	// This is what makes a test hermetic without unsetting process env vars.
	t.Setenv(Key("mistral", SuffixModel), "")
	assert.Equal(t, "mistral-large-latest", Model("mistral", "mistral-large-latest"))
}

func TestTimeoutParsing(t *testing.T) {
	t.Setenv(Key("ollama", SuffixTimeout), "3m")
	assert.Equal(t, 3*time.Minute, Timeout("ollama", 120*time.Second))
}

func TestUnparseableTimeoutFallsBackRatherThanMeaningForever(t *testing.T) {
	// http.Client.Timeout == 0 means "no timeout"; honouring a typo literally
	// would turn an attempt to SHORTEN a timeout into an unbounded request.
	for _, bad := range []string{"two minutes", "0s", "-5s", "180"} {
		t.Setenv(Key("ollama", SuffixTimeout), bad)
		assert.Equal(t, 120*time.Second, Timeout("ollama", 120*time.Second),
			"value %q must fall back, not be honoured", bad)
	}
}

func TestSourceReportsProvenance(t *testing.T) {
	t.Setenv(Key("ollama", SuffixBaseURL), "")
	t.Setenv("OLLAMA_HOST", "")
	assert.Equal(t, "", Source("ollama", SuffixBaseURL, "OLLAMA_HOST"),
		"nothing set -> no source, the compiled fallback is in force")

	t.Setenv("OLLAMA_HOST", "http://x:1")
	assert.Equal(t, "OLLAMA_HOST", Source("ollama", SuffixBaseURL, "OLLAMA_HOST"))

	t.Setenv(Key("ollama", SuffixBaseURL), "http://y:2")
	assert.Equal(t, Key("ollama", SuffixBaseURL),
		Source("ollama", SuffixBaseURL, "OLLAMA_HOST"))
}

func TestAliasesAreTriedInOrder(t *testing.T) {
	t.Setenv(Key("x", SuffixModel), "")
	t.Setenv("FIRST_ALIAS", "")
	t.Setenv("SECOND_ALIAS", "from-second")
	assert.Equal(t, "from-second", Model("x", "fallback", "FIRST_ALIAS", "SECOND_ALIAS"))

	t.Setenv("FIRST_ALIAS", "from-first")
	assert.Equal(t, "from-first", Model("x", "fallback", "FIRST_ALIAS", "SECOND_ALIAS"))
}

func TestLookupDoesNotMutateTheCallersAliasSlice(t *testing.T) {
	// lookup() prepends the canonical key to the alias list. Doing that with a
	// bare append onto the caller's slice would clobber the caller's backing
	// array whenever it had spare capacity.
	aliases := make([]string, 1, 4)
	aliases[0] = "ALIAS_ONE"
	_ = Model("x", "fallback", aliases...)
	assert.Equal(t, []string{"ALIAS_ONE"}, aliases)
}
