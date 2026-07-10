package llmprovider

import (
	"testing"
	"time"
)

// TestHealthMonitor_ForceCheckBeforeStart_MustNotPanic reproduces a nil-context
// panic in the PUBLIC API surface.
//
// An external caller of this module can only reach the monitor through exported
// methods: NewHealthMonitor -> RegisterProvider -> ForceCheck. The unexported
// hm.ctx field is nil until Start() is called, and ForceCheck() -> checkProvider()
// does context.WithTimeout(hm.ctx, ...), which panics ("cannot create context
// from nil parent") when hm.ctx is nil.
//
// The pre-existing in-package TestHealthMonitor_ForceCheck papers over this by
// assigning hm.ctx = context.Background() directly — an assignment no external
// caller can make because the field is unexported. This test uses ONLY the
// exported API, exactly as a real consumer would.
func TestHealthMonitor_ForceCheckBeforeStart_MustNotPanic(t *testing.T) {
	config := HealthMonitorConfig{
		HealthyThreshold: 1,
		Timeout:          5 * time.Second,
		Enabled:          false,
	}
	hm := NewHealthMonitor(config)
	hm.RegisterProvider("test", &mockProvider{})

	// No Start(), no package-internal hm.ctx assignment — purely the public API.
	err := hm.ForceCheck("test")
	if err != nil {
		t.Fatalf("ForceCheck returned error: %v", err)
	}

	health, ok := hm.GetHealth("test")
	if !ok {
		t.Fatal("expected a health record after ForceCheck")
	}
	if health.Status != HealthStatusHealthy {
		t.Fatalf("expected healthy after successful check, got %s", health.Status)
	}
}
