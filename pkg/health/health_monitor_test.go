package health

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"digital.vasic.llmprovider/pkg/models"
	"github.com/stretchr/testify/assert"
)

type mockProvider struct {
	healthErr   error
	healthDelay time.Duration
	mu          sync.Mutex
	checkCount  int
}

func (m *mockProvider) Complete(_ context.Context, _ *models.LLMRequest) (*models.LLMResponse, error) {
	return &models.LLMResponse{}, nil
}
func (m *mockProvider) CompleteStream(_ context.Context, _ *models.LLMRequest) (<-chan *models.LLMResponse, error) {
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
func (m *mockProvider) ValidateConfig(_ map[string]interface{}) (bool, []string) {
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
	hm.RegisterProvider("test-provider", &mockProvider{})
	hm.UnregisterProvider("test-provider")
	_, exists := hm.GetHealth("test-provider")
	assert.False(t, exists)
}

func TestHealthMonitor_CheckProvider_Healthy(t *testing.T) {
	config := HealthMonitorConfig{
		CheckInterval: 100 * time.Millisecond, HealthyThreshold: 1,
		UnhealthyThreshold: 2, Timeout: 5 * time.Second, Enabled: true,
	}
	hm := NewHealthMonitor(config)
	hm.RegisterProvider("test-provider", &mockProvider{})
	hm.Start()
	defer hm.Stop()
	time.Sleep(200 * time.Millisecond)
	health, _ := hm.GetHealth("test-provider")
	assert.Equal(t, HealthStatusHealthy, health.Status)
	assert.True(t, health.SuccessCount > 0)
}

func TestHealthMonitor_CheckProvider_Unhealthy(t *testing.T) {
	config := HealthMonitorConfig{
		CheckInterval: 50 * time.Millisecond, HealthyThreshold: 1,
		UnhealthyThreshold: 2, Timeout: 5 * time.Second, Enabled: true,
	}
	hm := NewHealthMonitor(config)
	hm.RegisterProvider("test-provider", &mockProvider{healthErr: errors.New("connection failed")})
	hm.Start()
	defer hm.Stop()
	time.Sleep(200 * time.Millisecond)
	health, _ := hm.GetHealth("test-provider")
	assert.Equal(t, HealthStatusUnhealthy, health.Status)
	assert.Equal(t, "connection failed", health.LastError)
}

func TestHealthMonitor_GetAllHealth(t *testing.T) {
	hm := NewDefaultHealthMonitor()
	hm.RegisterProvider("p1", &mockProvider{})
	hm.RegisterProvider("p2", &mockProvider{})
	hm.RegisterProvider("p3", &mockProvider{})
	allHealth := hm.GetAllHealth()
	assert.Len(t, allHealth, 3)
}

func TestHealthMonitor_RecordSuccess(t *testing.T) {
	config := HealthMonitorConfig{HealthyThreshold: 2, UnhealthyThreshold: 3, Enabled: false}
	hm := NewHealthMonitor(config)
	hm.RegisterProvider("test", &mockProvider{})
	health, _ := hm.GetHealth("test")
	assert.Equal(t, HealthStatusUnknown, health.Status)
	hm.RecordSuccess("test")
	hm.RecordSuccess("test")
	health, _ = hm.GetHealth("test")
	assert.Equal(t, HealthStatusHealthy, health.Status)
}

func TestHealthMonitor_RecordFailure(t *testing.T) {
	config := HealthMonitorConfig{HealthyThreshold: 2, UnhealthyThreshold: 2, Enabled: false}
	hm := NewHealthMonitor(config)
	hm.RegisterProvider("test", &mockProvider{})
	hm.RecordFailure("test", errors.New("error 1"))
	hm.RecordFailure("test", errors.New("error 2"))
	health, _ := hm.GetHealth("test")
	assert.Equal(t, HealthStatusUnhealthy, health.Status)
	assert.Equal(t, int64(2), health.FailureCount)
}

func TestHealthMonitor_AggregateHealth(t *testing.T) {
	config := HealthMonitorConfig{HealthyThreshold: 1, UnhealthyThreshold: 1, Enabled: false}
	hm := NewHealthMonitor(config)
	hm.RegisterProvider("h1", &mockProvider{})
	hm.RegisterProvider("h2", &mockProvider{})
	hm.RegisterProvider("u1", &mockProvider{})
	hm.RecordSuccess("h1")
	hm.RecordSuccess("h2")
	hm.RecordFailure("u1", errors.New("error"))
	agg := hm.GetAggregateHealth()
	assert.Equal(t, 3, agg.TotalProviders)
	assert.Equal(t, 2, agg.HealthyProviders)
	assert.Equal(t, 1, agg.UnhealthyProviders)
	assert.Equal(t, HealthStatusDegraded, agg.OverallStatus)
}

func TestHealthMonitor_StartStop(t *testing.T) {
	config := HealthMonitorConfig{CheckInterval: 50 * time.Millisecond, Enabled: true}
	hm := NewHealthMonitor(config)
	assert.False(t, hm.IsRunning())
	hm.Start()
	assert.True(t, hm.IsRunning())
	hm.Stop()
	time.Sleep(100 * time.Millisecond)
	assert.False(t, hm.IsRunning())
}

func TestHealthMonitor_DisabledDoesNotStart(t *testing.T) {
	hm := NewHealthMonitor(HealthMonitorConfig{Enabled: false})
	hm.Start()
	assert.False(t, hm.IsRunning())
}

func TestHealthMonitor_ForceCheck(t *testing.T) {
	config := HealthMonitorConfig{HealthyThreshold: 1, Timeout: 5 * time.Second, Enabled: false}
	hm := NewHealthMonitor(config)
	provider := &mockProvider{}
	hm.RegisterProvider("test", provider)
	err := hm.ForceCheck("test")
	assert.NoError(t, err)
	health, _ := hm.GetHealth("test")
	assert.Equal(t, HealthStatusHealthy, health.Status)
	assert.Equal(t, 1, provider.GetCheckCount())
}

func TestHealthMonitor_ConcurrentAccess(t *testing.T) {
	config := HealthMonitorConfig{
		CheckInterval: 10 * time.Millisecond, HealthyThreshold: 1,
		UnhealthyThreshold: 3, Timeout: 1 * time.Second, Enabled: true,
	}
	hm := NewHealthMonitor(config)
	for i := 0; i < 10; i++ {
		hm.RegisterProvider("provider"+string(rune('0'+i)), &mockProvider{})
	}
	hm.Start()
	const iterations = 100
	var wg sync.WaitGroup
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = hm.GetAllHealth()
			_ = hm.GetHealthyProviders()
			_ = hm.GetAggregateHealth()
			hm.RecordSuccess("provider0")
			hm.RecordFailure("provider1", errors.New("test"))
		}()
	}
	wg.Wait()
	hm.Stop()

	// Observable post-concurrency invariants (not just -race coverage):
	// every registered provider is still tracked, and the success/failure
	// counters reflect AT LEAST the records the goroutines applied — proving
	// the concurrent RecordSuccess/RecordFailure calls were not lost under the
	// mutex (the background monitor loop may add a few more, hence >=, but it
	// can never lose any of the 100 manual records).
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
