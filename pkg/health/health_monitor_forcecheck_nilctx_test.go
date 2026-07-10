package health

import (
	"testing"
	"time"
)

// TestHealthMonitor_ForceCheckBeforeStart_MustNotPanic reproduces a nil-context
// panic reachable purely through the exported API: NewHealthMonitor ->
// RegisterProvider -> ForceCheck. hm.ctx is nil until Start() runs, and
// checkProvider does context.WithTimeout(hm.ctx, ...) which panics on a nil
// parent. The pre-existing in-package test masks this by setting the unexported
// hm.ctx field directly, which an external caller cannot do.
func TestHealthMonitor_ForceCheckBeforeStart_MustNotPanic(t *testing.T) {
	config := HealthMonitorConfig{
		HealthyThreshold: 1,
		Timeout:          5 * time.Second,
		Enabled:          false,
	}
	hm := NewHealthMonitor(config)
	hm.RegisterProvider("test", &mockProvider{})

	// Public API only — no hm.ctx assignment.
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
