package llmprovider

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"digital.vasic.llmprovider/pkg/models"
	"github.com/stretchr/testify/assert"
)

// mockProvider is a mock LLM provider for testing
type mockProvider struct {
	healthErr   error
	healthDelay time.Duration
	mu          sync.Mutex
	checkCount  int
}

func (m *mockProvider) Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error) {
	return &models.LLMResponse{}, nil
}

func (m *mockProvider) CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error) {
	ch := make(chan *models.LLMResponse)
	close(ch)
	return ch, nil
}

func (m *mockProvider) HealthCheck() error {
	m.mu.Lock()
	m.checkCount++
	delay := m.healthDelay
	err := m.healthErr
	m.mu.Unlock()

	if delay > 0 {
		time.Sleep(delay)
	}
	return err
}

func (m *mockProvider) GetCapabilities() *models.ProviderCapabilities {
	return &models.ProviderCapabilities{}
}

func (m *mockProvider) ValidateConfig(config map[string]interface{}) (bool, []string) {
	return true, nil
}

func (m *mockProvider) SetHealthError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.healthErr = err
}

func (m *mockProvider) GetCheckCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.checkCount
}

func TestDefaultHealthMonitorConfig(t *testing.T) {
	config := DefaultHealthMonitorConfig()

	assert.Equal(t, 30*time.Second, config.CheckInterval)
	assert.Equal(t, 2, config.HealthyThreshold)
	assert.Equal(t, 3, config.UnhealthyThreshold)
	assert.Equal(t, 10*time.Second, config.Timeout)
	assert.True(t, config.Enabled)
}

func TestHealthMonitor_RegisterProvider(t *testing.T) {
	hm := NewDefaultHealthMonitor()
	provider := &mockProvider{}

	hm.RegisterProvider("test-provider", provider)

	health, exists := hm.GetHealth("test-provider")
	assert.True(t, exists)
	assert.Equal(t, "test-provider", health.ProviderID)
	assert.Equal(t, HealthStatusUnknown, health.Status)
}

func TestHealthMonitor_UnregisterProvider(t *testing.T) {
	hm := NewDefaultHealthMonitor()
	provider := &mockProvider{}

	hm.RegisterProvider("test-provider", provider)
	hm.UnregisterProvider("test-provider")

	_, exists := hm.GetHealth("test-provider")
	assert.False(t, exists)
}

func TestHealthMonitor_CheckProvider_Healthy(t *testing.T) {
	config := HealthMonitorConfig{
		CheckInterval:      100 * time.Millisecond,
		HealthyThreshold:   1,
		UnhealthyThreshold: 2,
		Timeout:            5 * time.Second,
		Enabled:            true,
	}
	hm := NewHealthMonitor(config)
	provider := &mockProvider{}

	hm.RegisterProvider("test-provider", provider)
	hm.Start()
	defer hm.Stop()

	// Wait for health check
	time.Sleep(200 * time.Millisecond)

	health, _ := hm.GetHealth("test-provider")
	assert.Equal(t, HealthStatusHealthy, health.Status)
	assert.True(t, health.SuccessCount > 0)
}

func TestHealthMonitor_CheckProvider_Unhealthy(t *testing.T) {
	config := HealthMonitorConfig{
		CheckInterval:      50 * time.Millisecond,
		HealthyThreshold:   1,
		UnhealthyThreshold: 2,
		Timeout:            5 * time.Second,
		Enabled:            true,
	}
	hm := NewHealthMonitor(config)
	provider := &mockProvider{healthErr: errors.New("connection failed")}

	hm.RegisterProvider("test-provider", provider)
	hm.Start()
	defer hm.Stop()

	// Wait for multiple health checks
	time.Sleep(200 * time.Millisecond)

	health, _ := hm.GetHealth("test-provider")
	assert.Equal(t, HealthStatusUnhealthy, health.Status)
	assert.Equal(t, "connection failed", health.LastError)
}

func TestHealthMonitor_GetAllHealth(t *testing.T) {
	hm := NewDefaultHealthMonitor()

	hm.RegisterProvider("provider1", &mockProvider{})
	hm.RegisterProvider("provider2", &mockProvider{})
	hm.RegisterProvider("provider3", &mockProvider{})

	allHealth := hm.GetAllHealth()
	assert.Len(t, allHealth, 3)
	assert.Contains(t, allHealth, "provider1")
	assert.Contains(t, allHealth, "provider2")
	assert.Contains(t, allHealth, "provider3")
}

func TestHealthMonitor_GetHealthyProviders(t *testing.T) {
	config := HealthMonitorConfig{
		CheckInterval:      50 * time.Millisecond,
		HealthyThreshold:   1,
		UnhealthyThreshold: 2,
		Timeout:            5 * time.Second,
		Enabled:            true,
	}
	hm := NewHealthMonitor(config)

	healthyProvider := &mockProvider{}
	unhealthyProvider := &mockProvider{healthErr: errors.New("error")}

	hm.RegisterProvider("healthy", healthyProvider)
	hm.RegisterProvider("unhealthy", unhealthyProvider)

	hm.Start()
	defer hm.Stop()

	// Wait for health checks
	time.Sleep(200 * time.Millisecond)

	healthy := hm.GetHealthyProviders()
	assert.Contains(t, healthy, "healthy")
	assert.NotContains(t, healthy, "unhealthy")
}

func TestHealthMonitor_IsHealthy(t *testing.T) {
	config := HealthMonitorConfig{
		CheckInterval:      50 * time.Millisecond,
		HealthyThreshold:   1,
		UnhealthyThreshold: 2,
		Timeout:            5 * time.Second,
		Enabled:            true,
	}
	hm := NewHealthMonitor(config)

	provider := &mockProvider{}
	hm.RegisterProvider("test", provider)

	hm.Start()
	defer hm.Stop()

	time.Sleep(100 * time.Millisecond)

	assert.True(t, hm.IsHealthy("test"))
	assert.False(t, hm.IsHealthy("nonexistent"))
}

func TestHealthMonitor_RecordSuccess(t *testing.T) {
	config := HealthMonitorConfig{
		HealthyThreshold:   2,
		UnhealthyThreshold: 3,
		Enabled:            false, // Don't auto-check
	}
	hm := NewHealthMonitor(config)

	provider := &mockProvider{}
	hm.RegisterProvider("test", provider)

	// Initially unknown
	health, _ := hm.GetHealth("test")
	assert.Equal(t, HealthStatusUnknown, health.Status)

	// Record successes
	hm.RecordSuccess("test")
	hm.RecordSuccess("test")

	health, _ = hm.GetHealth("test")
	assert.Equal(t, HealthStatusHealthy, health.Status)
	assert.Equal(t, int64(2), health.SuccessCount)
}

func TestHealthMonitor_RecordFailure(t *testing.T) {
	config := HealthMonitorConfig{
		HealthyThreshold:   2,
		UnhealthyThreshold: 2,
		Enabled:            false,
	}
	hm := NewHealthMonitor(config)

	provider := &mockProvider{}
	hm.RegisterProvider("test", provider)

	// Record failures
	hm.RecordFailure("test", errors.New("error 1"))
	hm.RecordFailure("test", errors.New("error 2"))

	health, _ := hm.GetHealth("test")
	assert.Equal(t, HealthStatusUnhealthy, health.Status)
	assert.Equal(t, int64(2), health.FailureCount)
	assert.Equal(t, "error 2", health.LastError)
}

func TestHealthMonitor_StatusTransition(t *testing.T) {
	config := HealthMonitorConfig{
		CheckInterval:      50 * time.Millisecond,
		HealthyThreshold:   1,
		UnhealthyThreshold: 2,
		Timeout:            5 * time.Second,
		Enabled:            true,
	}
	hm := NewHealthMonitor(config)

	provider := &mockProvider{}
	hm.RegisterProvider("test", provider)

	statusChanges := make([]struct {
		old HealthStatus
		new HealthStatus
	}, 0)
	var mu sync.Mutex

	hm.AddListener(func(providerID string, oldStatus, newStatus HealthStatus) {
		mu.Lock()
		statusChanges = append(statusChanges, struct {
			old HealthStatus
			new HealthStatus
		}{oldStatus, newStatus})
		mu.Unlock()
	})

	hm.Start()
	time.Sleep(100 * time.Millisecond)

	// Should transition to healthy
	mu.Lock()
	assert.Len(t, statusChanges, 1)
	assert.Equal(t, HealthStatusUnknown, statusChanges[0].old)
	assert.Equal(t, HealthStatusHealthy, statusChanges[0].new)
	mu.Unlock()

	// Make provider fail
	provider.SetHealthError(errors.New("failure"))

	time.Sleep(200 * time.Millisecond)
	hm.Stop()

	// Should have transitioned through degraded to unhealthy
	mu.Lock()
	assert.True(t, len(statusChanges) > 1)
	mu.Unlock()
}

func TestHealthMonitor_AggregateHealth(t *testing.T) {
	config := HealthMonitorConfig{
		HealthyThreshold:   1,
		UnhealthyThreshold: 1,
		Enabled:            false,
	}
	hm := NewHealthMonitor(config)

	hm.RegisterProvider("healthy1", &mockProvider{})
	hm.RegisterProvider("healthy2", &mockProvider{})
	hm.RegisterProvider("unhealthy", &mockProvider{healthErr: errors.New("error")})

	// Mark some as healthy
	hm.RecordSuccess("healthy1")
	hm.RecordSuccess("healthy2")
	hm.RecordFailure("unhealthy", errors.New("error"))

	agg := hm.GetAggregateHealth()

	assert.Equal(t, 3, agg.TotalProviders)
	assert.Equal(t, 2, agg.HealthyProviders)
	assert.Equal(t, 1, agg.UnhealthyProviders)
	assert.Equal(t, HealthStatusDegraded, agg.OverallStatus)
}

func TestHealthMonitor_AggregateHealth_AllHealthy(t *testing.T) {
	config := HealthMonitorConfig{
		HealthyThreshold: 1,
		Enabled:          false,
	}
	hm := NewHealthMonitor(config)

	hm.RegisterProvider("p1", &mockProvider{})
	hm.RegisterProvider("p2", &mockProvider{})

	hm.RecordSuccess("p1")
	hm.RecordSuccess("p2")

	agg := hm.GetAggregateHealth()
	assert.Equal(t, HealthStatusHealthy, agg.OverallStatus)
}

func TestHealthMonitor_AggregateHealth_AllUnhealthy(t *testing.T) {
	config := HealthMonitorConfig{
		UnhealthyThreshold: 1,
		Enabled:            false,
	}
	hm := NewHealthMonitor(config)

	hm.RegisterProvider("p1", &mockProvider{})
	hm.RegisterProvider("p2", &mockProvider{})

	hm.RecordFailure("p1", errors.New("error"))
	hm.RecordFailure("p2", errors.New("error"))

	agg := hm.GetAggregateHealth()
	assert.Equal(t, HealthStatusUnhealthy, agg.OverallStatus)
}

func TestHealthMonitor_ForceCheck(t *testing.T) {
	config := HealthMonitorConfig{
		HealthyThreshold: 1,
		Timeout:          5 * time.Second,
		Enabled:          false,
	}
	hm := NewHealthMonitor(config)

	provider := &mockProvider{}
	hm.RegisterProvider("test", provider)

	err := hm.ForceCheck("test")
	assert.NoError(t, err)

	health, _ := hm.GetHealth("test")
	assert.Equal(t, HealthStatusHealthy, health.Status)
	assert.Equal(t, 1, provider.GetCheckCount())
}

func TestHealthMonitor_Timeout(t *testing.T) {
	config := HealthMonitorConfig{
		CheckInterval:      50 * time.Millisecond,
		HealthyThreshold:   1,
		UnhealthyThreshold: 1,
		Timeout:            50 * time.Millisecond, // Very short timeout
		Enabled:            true,
	}
	hm := NewHealthMonitor(config)

	// Provider that takes too long
	provider := &mockProvider{healthDelay: 200 * time.Millisecond}
	hm.RegisterProvider("slow", provider)

	hm.Start()
	time.Sleep(150 * time.Millisecond)
	hm.Stop()

	health, _ := hm.GetHealth("slow")
	assert.Equal(t, HealthStatusUnhealthy, health.Status)
}

func TestHealthMonitor_StartStop(t *testing.T) {
	config := HealthMonitorConfig{
		CheckInterval: 50 * time.Millisecond,
		Enabled:       true,
	}
	hm := NewHealthMonitor(config)

	assert.False(t, hm.IsRunning())

	hm.Start()
	assert.True(t, hm.IsRunning())

	hm.Stop()
	time.Sleep(100 * time.Millisecond) // Give time for goroutine to exit
	assert.False(t, hm.IsRunning())
}

func TestHealthMonitor_DisabledDoesNotStart(t *testing.T) {
	config := HealthMonitorConfig{
		Enabled: false,
	}
	hm := NewHealthMonitor(config)

	hm.Start()
	assert.False(t, hm.IsRunning())
}

func TestHealthMonitor_GetConfig(t *testing.T) {
	config := HealthMonitorConfig{
		CheckInterval:      1 * time.Minute,
		HealthyThreshold:   5,
		UnhealthyThreshold: 10,
		Timeout:            30 * time.Second,
		Enabled:            true,
	}
	hm := NewHealthMonitor(config)

	retrieved := hm.GetConfig()
	assert.Equal(t, config.CheckInterval, retrieved.CheckInterval)
	assert.Equal(t, config.HealthyThreshold, retrieved.HealthyThreshold)
	assert.Equal(t, config.UnhealthyThreshold, retrieved.UnhealthyThreshold)
}

func TestHealthMonitor_ConcurrentAccess(t *testing.T) {
	config := HealthMonitorConfig{
		CheckInterval:      10 * time.Millisecond,
		HealthyThreshold:   1,
		UnhealthyThreshold: 3,
		Timeout:            1 * time.Second,
		Enabled:            true,
	}
	hm := NewHealthMonitor(config)

	for i := 0; i < 10; i++ {
		hm.RegisterProvider("provider"+string(rune('0'+i)), &mockProvider{})
	}

	hm.Start()

	// Concurrent access
	const iterations = 100
	var wg sync.WaitGroup
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			_ = hm.GetAllHealth()
			_ = hm.GetHealthyProviders()
			_ = hm.GetAggregateHealth()
			hm.RecordSuccess("provider0")
			hm.RecordFailure("provider1", errors.New("test"))
		}(i)
	}

	wg.Wait()
	hm.Stop()

	// Observable post-concurrency invariants (not just -race coverage):
	// all registered providers remain tracked and the manual records survived
	// the concurrent mutation (>= because the background monitor loop may add
	// a few extra checks, but it can never drop any of the 100 manual records).
	agg := hm.GetAggregateHealth()
	assert.Equal(t, 10, agg.TotalProviders, "all 10 registered providers must remain tracked after concurrent access")

	h0, ok0 := hm.GetHealth("provider0")
	if assert.True(t, ok0, "provider0 must be retrievable after concurrent RecordSuccess") {
		assert.GreaterOrEqual(t, h0.SuccessCount, int64(iterations), "provider0 must record at least %d successes", iterations)
	}
	h1, ok1 := hm.GetHealth("provider1")
	if assert.True(t, ok1, "provider1 must be retrievable after concurrent RecordFailure") {
		assert.GreaterOrEqual(t, h1.FailureCount, int64(iterations), "provider1 must record at least %d failures", iterations)
	}
}
