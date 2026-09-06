package junie_test

import (
	"os"
	"testing"
	"time"

	"digital.vasic.llmprovider/pkg/providers/junie"
)

func TestJunieCLIConfig_Default(t *testing.T) {
	config := junie.DefaultJunieCLIConfig()
	if config.MaxTokens != 4096 {
		t.Errorf("Default MaxTokens should be 4096, got %d", config.MaxTokens)
	}
	if config.Model != "junie-1" {
		t.Errorf("Default Model should be junie-1, got %s", config.Model)
	}
}

func TestDefaultJunieConfig(t *testing.T) {
	config := junie.DefaultJunieConfig()
	if config.Timeout != 180*time.Second {
		t.Errorf("Default timeout should be 180s")
	}
	if config.MaxTokens != 8192 {
		t.Errorf("Default MaxTokens should be 8192")
	}
	if config.Model != "sonnet" {
		t.Errorf("Default Model should be sonnet")
	}
}

func TestDefaultJunieACPConfig(t *testing.T) {
	config := junie.DefaultJunieACPConfig()
	if config.MaxTokens != 4096 {
		t.Errorf("Default MaxTokens should be 4096, got %d", config.MaxTokens)
	}
	if config.Model != "junie-1" {
		t.Errorf("Default Model should be junie-1, got %s", config.Model)
	}
}

func TestNewJunieCLIProvider(t *testing.T) {
	config := junie.DefaultJunieCLIConfig()
	p := junie.NewJunieCLIProvider(config)
	if p.GetCurrentModel() == "" {
		t.Errorf("Default model should be set")
	}
}

func TestNewJunieACPProvider(t *testing.T) {
	config := junie.DefaultJunieACPConfig()
	p := junie.NewJunieACPProvider(config)
	if p.GetCurrentModel() == "" {
		t.Errorf("Default model should be set")
	}
}

func TestNewJunieProvider(t *testing.T) {
	config := junie.DefaultJunieConfig()
	p := junie.NewJunieProvider(config)
	if p.GetCurrentModel() != "sonnet" {
		t.Errorf("Default model should be sonnet")
	}
}

func TestJunieCLIProvider_GetName(t *testing.T) {
	p := junie.NewJunieCLIProvider(junie.DefaultJunieCLIConfig())
	name := p.GetName()
	if name != "junie-cli" {
		t.Errorf("Expected name junie-cli, got %s", name)
	}
}

func TestJunieCLIProvider_GetProviderType(t *testing.T) {
	p := junie.NewJunieCLIProvider(junie.DefaultJunieCLIConfig())
	providerType := p.GetProviderType()
	if providerType != "cli" {
		t.Errorf("Expected provider type cli, got %s", providerType)
	}
}

func TestJunieCLIProvider_GetCurrentModel(t *testing.T) {
	config := junie.DefaultJunieCLIConfig()
	p := junie.NewJunieCLIProvider(config)
	model := p.GetCurrentModel()
	// Stub always returns "junie-1"
	if model != "junie-1" {
		t.Errorf("Expected model junie-1, got %s", model)
	}
}

func TestJunieCLIProvider_SetModel(t *testing.T) {
	p := junie.NewJunieCLIProvider(junie.DefaultJunieCLIConfig())
	// SetModel is a no-op on the stub; just ensure no panic
	p.SetModel("gemini-pro")
	// Stub still returns "junie-1"
	if p.GetCurrentModel() != "junie-1" {
		t.Errorf("Expected model junie-1, got %s", p.GetCurrentModel())
	}
}

func TestJunieACPProvider_GetCurrentModel(t *testing.T) {
	p := junie.NewJunieACPProvider(junie.DefaultJunieACPConfig())
	model := p.GetCurrentModel()
	if model != "junie-1" {
		t.Errorf("Expected model junie-1, got %s", model)
	}
}

func TestJunieACPProvider_IsAvailable(t *testing.T) {
	p := junie.NewJunieACPProvider(junie.DefaultJunieACPConfig())
	if p.IsAvailable() {
		t.Errorf("Expected ACP to not be available in standalone module")
	}
}

func TestJunieProvider_GetName(t *testing.T) {
	p := junie.NewJunieProvider(junie.DefaultJunieConfig())
	name := p.GetName()
	if name != "junie" {
		t.Errorf("Expected name junie, got %s", name)
	}
}

func TestJunieProvider_GetProviderType(t *testing.T) {
	p := junie.NewJunieProvider(junie.DefaultJunieConfig())
	providerType := p.GetProviderType()
	if providerType != "junie" {
		t.Errorf("Expected provider type junie, got %s", providerType)
	}
}

func TestJunieProvider_SupportedModels(t *testing.T) {
	p := junie.NewJunieProvider(junie.DefaultJunieConfig())
	caps := p.GetCapabilities()
	if len(caps.SupportedModels) == 0 {
		t.Errorf("Expected at least one model, got %d", len(caps.SupportedModels))
	}
	for _, model := range caps.SupportedModels {
		if model == "" {
			t.Errorf("Model should not be empty")
		}
	}
}

func TestJunieProvider_BYOKProviders(t *testing.T) {
	p := junie.NewJunieProvider(junie.DefaultJunieConfig())
	byok := p.GetBYOKProviders()
	// BYOK providers depend on env vars being set; just verify no panic
	for _, provider := range byok {
		if provider == "" {
			t.Errorf("BYOK provider should not be empty")
		}
	}
}

func TestIsJunieInstalled(t *testing.T) {
	// In this standalone llmprovider module the Junie CLI is not bundled;
	// junie_cli_stub.go is the sole definition and MUST report "not installed".
	// Assert the observable contract deterministically rather than logging.
	installed := junie.IsJunieInstalled()
	if installed {
		t.Fatalf("IsJunieInstalled() = true, want false: standalone module ships the CLI stub and cannot have a real Junie CLI installed")
	}
}

func TestIsJunieAuthenticated(t *testing.T) {
	// Authentication cannot succeed without an installed CLI; the stub MUST
	// report "not authenticated". Assert the contract, and assert the
	// invariant that an un-installed CLI is never reported as authenticated.
	authenticated := junie.IsJunieAuthenticated()
	if authenticated {
		t.Fatalf("IsJunieAuthenticated() = true, want false: standalone module ships the CLI stub and cannot be authenticated")
	}
	if junie.IsJunieInstalled() == false && authenticated {
		t.Fatalf("invariant violated: Junie reported authenticated while not installed")
	}
}

func TestJunieProvider_GetCapabilities(t *testing.T) {
	p := junie.NewJunieProvider(junie.DefaultJunieConfig())
	caps := p.GetCapabilities()
	if len(caps.SupportedModels) == 0 {
		t.Errorf("Expected at least one model, got %d", len(caps.SupportedModels))
	}
	if !caps.SupportsStreaming {
		t.Errorf("Expected streaming support")
	}
	if !caps.SupportsTools {
		t.Errorf("Expected tools support")
	}
}

func TestJunieProvider_ValidateConfig(t *testing.T) {
	p := junie.NewJunieProvider(junie.DefaultJunieConfig())
	valid, issues := p.ValidateConfig(nil)

	installed := junie.IsJunieInstalled()
	credentialed := os.Getenv("JUNIE_API_KEY") != "" || junie.IsJunieAuthenticated()

	if installed && credentialed {
		// This branch used to contain nothing but a t.Logf, so on a machine
		// where Junie IS installed and credentialed the test asserted nothing
		// whatsoever. ValidateConfig appends an issue only when the CLI is
		// missing or no credential can be found, so with both present its
		// contract is an unconditional valid=true with no issues.
		if !valid {
			t.Errorf("Expected valid config when Junie is installed and credentialed; issues: %v", issues)
		}
		if len(issues) != 0 {
			t.Errorf("Expected no issues when Junie is installed and credentialed, got: %v", issues)
		}
		return
	}

	if valid {
		t.Errorf("Expected invalid config when Junie not available")
	}
	if len(issues) == 0 {
		t.Errorf("Expected issues when Junie not available")
	}
	if !installed {
		found := false
		for _, issue := range issues {
			if issue == "junie command not installed" {
				found = true
			}
		}
		if !found {
			t.Errorf("Expected a 'junie command not installed' issue, got: %v", issues)
		}
	}
}

// TestJunieCLIStubs_AreStubsInThisModule records, as an executable fact, WHY
// the installed+credentialed branch of TestJunieProvider_ValidateConfig does
// not run here: junie_cli_stub.go hardcodes both predicates to false, so that
// branch is statically dead in this module. That is why an assertion-free
// branch survived there unnoticed.
//
// If the Junie CLI is ever really wired up, this test fails and points at the
// branch that then needs real coverage, instead of that branch silently
// resuming its old habit of checking nothing.
func TestJunieCLIStubs_AreStubsInThisModule(t *testing.T) {
	if junie.IsJunieInstalled() {
		t.Errorf("IsJunieInstalled is a stub returning false in this module; " +
			"it now returns true, so TestJunieProvider_ValidateConfig's " +
			"installed branch is live and needs reviewing")
	}
	if junie.IsJunieAuthenticated() {
		t.Errorf("IsJunieAuthenticated is a stub returning false in this module; " +
			"it now returns true, so the credentialed branch is live and " +
			"needs reviewing")
	}
}
