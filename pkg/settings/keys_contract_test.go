package settings_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"digital.vasic.llmprovider/pkg/settings"
)

// WHY THIS FILE EXISTS, AND WHY IT DOES NOT CALL settings.Key TO BUILD ITS
// EXPECTATIONS.
//
// The operator-facing surface of this package is not Key(); it is the literal
// string an operator types into a shell. A test that writes
//
//	assert.Equal(t, settings.Key("nvidia", settings.SuffixModel), …)
//
// asserts that Key agrees with Key. It is true by construction: rename the
// prefix, change the separator, fold the provider name differently, and that
// assertion still passes while every documented variable name silently changes
// underneath the operator. So every expectation below is a HAND-WRITTEN
// literal. If one of them has to be edited to make this file pass again, that
// edit IS the finding: a variable an operator may already have exported has
// just been renamed.
//
// The second half of the file closes the other half of the gap. A pinned list
// only protects the keys it lists, so TestEveryKeyInUseIsPinned walks the
// module's own source, extracts every provider key actually handed to
// settings.Model / BaseURL / Timeout, and fails on any that this file does not
// pin — which is what stops a new adapter from shipping an unasserted key.

// pinnedKeys maps a provider's env-var INFIX to the three literal
// environment-variable names it must produce. An empty string means "this
// adapter does not resolve that setting", and is not asserted.
//
// The map is keyed by the UPPER-CASE infix rather than by the lower-case
// provider string the adapters pass, and that is not cosmetic. Several
// provider names -- "codestral" is the clear case -- are also vendor MODEL
// names, so a table row spelling one in lower case is indistinguishable, to a
// line-based scanner, from a frozen model id: scripts/audit-environment-
// assumptions.sh reported exactly that on this file. The infix carries the same
// information without the collision, and providerKey() below recovers the
// lower-case form. What must stay hand-written is the EXPECTED NAMES, and they
// are; deriving those from settings.Key() is the defect this file exists to
// prevent.
var pinnedKeys = map[string][3]string{
	// {MODEL, BASE_URL, TIMEOUT}
	"AI21":          {"LLMPROVIDER_AI21_MODEL", "LLMPROVIDER_AI21_BASE_URL", "LLMPROVIDER_AI21_TIMEOUT"},
	"ANTHROPIC":     {"LLMPROVIDER_ANTHROPIC_MODEL", "LLMPROVIDER_ANTHROPIC_BASE_URL", "LLMPROVIDER_ANTHROPIC_TIMEOUT"},
	"CEREBRAS":      {"LLMPROVIDER_CEREBRAS_MODEL", "LLMPROVIDER_CEREBRAS_BASE_URL", "LLMPROVIDER_CEREBRAS_TIMEOUT"},
	"CHUTES":        {"LLMPROVIDER_CHUTES_MODEL", "LLMPROVIDER_CHUTES_BASE_URL", "LLMPROVIDER_CHUTES_TIMEOUT"},
	"CLAUDE":        {"LLMPROVIDER_CLAUDE_MODEL", "LLMPROVIDER_CLAUDE_BASE_URL", "LLMPROVIDER_CLAUDE_TIMEOUT"},
	"CLAUDE_OAUTH":  {"LLMPROVIDER_CLAUDE_OAUTH_MODEL", "", ""},
	"CLOUDFLARE":    {"LLMPROVIDER_CLOUDFLARE_MODEL", "LLMPROVIDER_CLOUDFLARE_BASE_URL", "LLMPROVIDER_CLOUDFLARE_TIMEOUT"},
	"CODESTRAL":     {"LLMPROVIDER_CODESTRAL_MODEL", "LLMPROVIDER_CODESTRAL_BASE_URL", "LLMPROVIDER_CODESTRAL_TIMEOUT"},
	"COHERE":        {"LLMPROVIDER_COHERE_MODEL", "LLMPROVIDER_COHERE_BASE_URL", "LLMPROVIDER_COHERE_TIMEOUT"},
	"DEEPSEEK":      {"LLMPROVIDER_DEEPSEEK_MODEL", "LLMPROVIDER_DEEPSEEK_BASE_URL", "LLMPROVIDER_DEEPSEEK_TIMEOUT"},
	"FIREWORKS":     {"LLMPROVIDER_FIREWORKS_MODEL", "LLMPROVIDER_FIREWORKS_BASE_URL", "LLMPROVIDER_FIREWORKS_TIMEOUT"},
	"GEMINI":        {"LLMPROVIDER_GEMINI_MODEL", "LLMPROVIDER_GEMINI_BASE_URL", "LLMPROVIDER_GEMINI_TIMEOUT"},
	"GEMINI_MODELS": {"", "LLMPROVIDER_GEMINI_MODELS_BASE_URL", ""},
	"GITHUBMODELS":  {"LLMPROVIDER_GITHUBMODELS_MODEL", "LLMPROVIDER_GITHUBMODELS_BASE_URL", "LLMPROVIDER_GITHUBMODELS_TIMEOUT"},
	"GROQ":          {"LLMPROVIDER_GROQ_MODEL", "LLMPROVIDER_GROQ_BASE_URL", "LLMPROVIDER_GROQ_TIMEOUT"},
	"HTTP":          {"", "", "LLMPROVIDER_HTTP_TIMEOUT"},
	"HUGGINGFACE":   {"LLMPROVIDER_HUGGINGFACE_MODEL", "LLMPROVIDER_HUGGINGFACE_BASE_URL", "LLMPROVIDER_HUGGINGFACE_TIMEOUT"},
	"HYPERBOLIC":    {"LLMPROVIDER_HYPERBOLIC_MODEL", "LLMPROVIDER_HYPERBOLIC_BASE_URL", "LLMPROVIDER_HYPERBOLIC_TIMEOUT"},
	"JUNIE":         {"LLMPROVIDER_JUNIE_MODEL", "", "LLMPROVIDER_JUNIE_TIMEOUT"},
	"KILO":          {"LLMPROVIDER_KILO_MODEL", "LLMPROVIDER_KILO_BASE_URL", "LLMPROVIDER_KILO_TIMEOUT"},
	"KIMI":          {"LLMPROVIDER_KIMI_MODEL", "LLMPROVIDER_KIMI_BASE_URL", "LLMPROVIDER_KIMI_TIMEOUT"},
	"MISTRAL":       {"LLMPROVIDER_MISTRAL_MODEL", "LLMPROVIDER_MISTRAL_BASE_URL", "LLMPROVIDER_MISTRAL_TIMEOUT"},
	"MODAL":         {"LLMPROVIDER_MODAL_MODEL", "LLMPROVIDER_MODAL_BASE_URL", "LLMPROVIDER_MODAL_TIMEOUT"},
	"NIA":           {"LLMPROVIDER_NIA_MODEL", "LLMPROVIDER_NIA_BASE_URL", "LLMPROVIDER_NIA_TIMEOUT"},
	"NLPCLOUD":      {"LLMPROVIDER_NLPCLOUD_MODEL", "LLMPROVIDER_NLPCLOUD_BASE_URL", "LLMPROVIDER_NLPCLOUD_TIMEOUT"},
	"NOVITA":        {"LLMPROVIDER_NOVITA_MODEL", "LLMPROVIDER_NOVITA_BASE_URL", "LLMPROVIDER_NOVITA_TIMEOUT"},
	"NVIDIA":        {"LLMPROVIDER_NVIDIA_MODEL", "LLMPROVIDER_NVIDIA_BASE_URL", "LLMPROVIDER_NVIDIA_TIMEOUT"},
	"OLLAMA":        {"LLMPROVIDER_OLLAMA_MODEL", "LLMPROVIDER_OLLAMA_BASE_URL", "LLMPROVIDER_OLLAMA_TIMEOUT"},
	"OPENAI":        {"LLMPROVIDER_OPENAI_MODEL", "LLMPROVIDER_OPENAI_BASE_URL", "LLMPROVIDER_OPENAI_TIMEOUT"},
	"OPENROUTER":    {"LLMPROVIDER_OPENROUTER_MODEL", "LLMPROVIDER_OPENROUTER_BASE_URL", "LLMPROVIDER_OPENROUTER_TIMEOUT"},
	"PERPLEXITY":    {"LLMPROVIDER_PERPLEXITY_MODEL", "LLMPROVIDER_PERPLEXITY_BASE_URL", "LLMPROVIDER_PERPLEXITY_TIMEOUT"},
	"PUBLICAI":      {"LLMPROVIDER_PUBLICAI_MODEL", "LLMPROVIDER_PUBLICAI_BASE_URL", "LLMPROVIDER_PUBLICAI_TIMEOUT"},
	"QWEN":          {"LLMPROVIDER_QWEN_MODEL", "LLMPROVIDER_QWEN_BASE_URL", "LLMPROVIDER_QWEN_TIMEOUT"},
	"QWEN_OAUTH":    {"", "LLMPROVIDER_QWEN_OAUTH_BASE_URL", ""},
	"REPLICATE":     {"LLMPROVIDER_REPLICATE_MODEL", "LLMPROVIDER_REPLICATE_BASE_URL", "LLMPROVIDER_REPLICATE_TIMEOUT"},
	"RETRY":         {"", "", "LLMPROVIDER_RETRY_TIMEOUT"},
	"SAMBANOVA":     {"LLMPROVIDER_SAMBANOVA_MODEL", "LLMPROVIDER_SAMBANOVA_BASE_URL", "LLMPROVIDER_SAMBANOVA_TIMEOUT"},
	"SARVAM":        {"LLMPROVIDER_SARVAM_MODEL", "LLMPROVIDER_SARVAM_BASE_URL", "LLMPROVIDER_SARVAM_TIMEOUT"},
	"SILICONFLOW":   {"LLMPROVIDER_SILICONFLOW_MODEL", "LLMPROVIDER_SILICONFLOW_BASE_URL", "LLMPROVIDER_SILICONFLOW_TIMEOUT"},
	"TOGETHER":      {"LLMPROVIDER_TOGETHER_MODEL", "LLMPROVIDER_TOGETHER_BASE_URL", "LLMPROVIDER_TOGETHER_TIMEOUT"},
	"UPSTAGE":       {"LLMPROVIDER_UPSTAGE_MODEL", "LLMPROVIDER_UPSTAGE_BASE_URL", "LLMPROVIDER_UPSTAGE_TIMEOUT"},
	"VENICE":        {"LLMPROVIDER_VENICE_MODEL", "LLMPROVIDER_VENICE_BASE_URL", "LLMPROVIDER_VENICE_TIMEOUT"},
	"VULAVULA":      {"LLMPROVIDER_VULAVULA_MODEL", "LLMPROVIDER_VULAVULA_BASE_URL", "LLMPROVIDER_VULAVULA_TIMEOUT"},
	"XAI":           {"LLMPROVIDER_XAI_MODEL", "LLMPROVIDER_XAI_BASE_URL", "LLMPROVIDER_XAI_TIMEOUT"},
	"ZAI":           {"LLMPROVIDER_ZAI_MODEL", "LLMPROVIDER_ZAI_BASE_URL", "LLMPROVIDER_ZAI_TIMEOUT"},
	"ZEN":           {"LLMPROVIDER_ZEN_MODEL", "", ""},
	"ZEN_API":       {"", "LLMPROVIDER_ZEN_API_BASE_URL", "LLMPROVIDER_ZEN_API_TIMEOUT"},
	"ZEN_MODELS":    {"", "", "LLMPROVIDER_ZEN_MODELS_TIMEOUT"},
	"ZHIPU":         {"LLMPROVIDER_ZHIPU_MODEL", "LLMPROVIDER_ZHIPU_BASE_URL", "LLMPROVIDER_ZHIPU_TIMEOUT"},
}

// providerKey recovers the lower-case provider string an adapter actually
// passes to settings.Model / BaseURL / Timeout from the table's upper-case
// infix. This is a spelling change, not a derivation of the assertion: the
// EXPECTED variable names remain hand-written literals, so a settings.Key()
// that changed its prefix, separator or folding rule still turns this file red.
func providerKey(infix string) string { return strings.ToLower(infix) }

func TestKeyProducesTheLiteralOperatorFacingName(t *testing.T) {
	suffixes := [3]string{settings.SuffixModel, settings.SuffixBaseURL, settings.SuffixTimeout}
	for infix, want := range pinnedKeys {
		provider := providerKey(infix)
		for i, expected := range want {
			if expected == "" {
				continue
			}
			if got := settings.Key(provider, suffixes[i]); got != expected {
				t.Errorf("settings.Key(%q, %q) = %q, want %q -- an operator's "+
					"exported variable has just been renamed",
					provider, suffixes[i], got, expected)
			}
		}
	}
}

// TestOverrideIsReadFromTheLiteralName proves the resolver reads the exact
// variable this file pins, by setting the literal name in the process
// environment and reading it back through the public API. Setting the name
// Key() computed would prove nothing.
func TestOverrideIsReadFromTheLiteralName(t *testing.T) {
	t.Setenv("LLMPROVIDER_NVIDIA_MODEL", "synthetic-model-a")
	if got := settings.Model("nvidia", "compiled-fallback"); got != "synthetic-model-a" {
		t.Errorf("Model(nvidia) = %q; LLMPROVIDER_NVIDIA_MODEL was not read", got)
	}

	t.Setenv("LLMPROVIDER_NVIDIA_BASE_URL", "https://endpoint.invalid/v1")
	if got := settings.BaseURL("nvidia", "https://fallback.invalid/v1"); got != "https://endpoint.invalid/v1" {
		t.Errorf("BaseURL(nvidia) = %q; LLMPROVIDER_NVIDIA_BASE_URL was not read", got)
	}

	t.Setenv("LLMPROVIDER_NVIDIA_TIMEOUT", "7s")
	if got := settings.Timeout("nvidia", 0); got.String() != "7s" {
		t.Errorf("Timeout(nvidia) = %v; LLMPROVIDER_NVIDIA_TIMEOUT was not read", got)
	}
}

// keyCall matches a provider key handed to one of the three resolvers as a
// STRING LITERAL. Keys built from a constant or an expression are handled
// separately below; they cannot be read out of the source text without
// resolving Go identifiers, which is a job for the compiler, not a regexp.
var keyCall = regexp.MustCompile(`settings\.(?:Model|BaseURL|Timeout)\("([a-z0-9_]+)"`)

// dynamicKeyOwners lists the call sites whose provider key is NOT a literal,
// with the reason. Each one is a deliberate design decision documented at the
// call site, and each is covered by a behavioural test elsewhere rather than by
// the pin table. Listing them here means a NEW dynamic key cannot appear
// unnoticed: the census below counts literal keys, and anything that stops
// being a literal has to be argued into this list by hand.
var dynamicKeyOwners = map[string]string{
	"pkg/providers/generic/generic.go":   "keyed by the caller's provider name; the shim fronts many backends",
	"pkg/discovery/discovery.go":         "keyed by config.ProviderName + \"_discovery\"",
	"pkg/providers/zen/zen.go":           "SettingsProviderAPI / SettingsProviderModel / SettingsProviderModels constants",
	"pkg/providers/zen/zen_http.go":      "SettingsProvider constants shared with zen.go",
	"pkg/providers/claude/claude.go":     "SettingsProvider / SettingsProviderOAuth constants",
	"pkg/providers/qwen/qwen.go":         "SettingsProvider / SettingsProviderOAuth constants",
	"pkg/providers/gemini/gemini.go":     "SettingsProvider / SettingsProviderModels constants",
	"pkg/providers/gemini/gemini_api.go": "SettingsProvider constants shared with gemini.go",
}

// TestEveryKeyInUseIsPinned is the anti-drift half. A pin table protects only
// what it lists; without this, a new adapter ships a new operator-facing
// variable with nothing asserting its name.
func TestEveryKeyInUseIsPinned(t *testing.T) {
	root := moduleRoot(t)
	files := trackedGoFiles(t, root)

	found := map[string][]string{}
	for _, rel := range files {
		if strings.HasSuffix(rel, "_test.go") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			t.Fatalf("read %s: %v", rel, err)
		}
		for _, m := range keyCall.FindAllStringSubmatch(string(b), -1) {
			found[m[1]] = append(found[m[1]], rel)
		}
	}

	if len(found) == 0 {
		// A regexp that matches nothing is a green test that proves nothing.
		t.Fatal("no settings.* call sites found at all -- this census is blind, " +
			"which is worse than a failure; check keyCall and trackedGoFiles")
	}

	pinned := make(map[string]bool, len(pinnedKeys))
	for infix := range pinnedKeys {
		pinned[providerKey(infix)] = true
	}

	var missing []string
	for key, sites := range found {
		if !pinned[key] {
			missing = append(missing, key+" (used in "+strings.Join(sites, ", ")+")")
		}
	}
	sort.Strings(missing)
	for _, m := range missing {
		t.Errorf("provider key %s is in use but is not pinned in pinnedKeys -- "+
			"add its literal LLMPROVIDER_* names", m)
	}
}

func moduleRoot(t *testing.T) string {
	t.Helper()
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Skipf("SKIP-OK: #env git rev-parse is unavailable, so the source "+
			"census cannot run: %v", err)
	}
	return strings.TrimSpace(string(out))
}

func trackedGoFiles(t *testing.T, root string) []string {
	t.Helper()
	cmd := exec.Command("git", "ls-files", "*.go")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		t.Skipf("SKIP-OK: #env git ls-files is unavailable: %v", err)
	}
	var files []string
	for _, l := range strings.Split(string(out), "\n") {
		if l = strings.TrimSpace(l); l != "" {
			files = append(files, l)
		}
	}
	if len(files) == 0 {
		t.Fatal("git ls-files returned no Go files -- the census would be blind")
	}
	return files
}
