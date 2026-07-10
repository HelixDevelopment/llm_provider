// Package health provides health monitoring for multiple LLM providers.
package health

import (
	"context"
	"sync"
	"time"

	"digital.vasic.llmprovider/pkg/provider"
)

// HealthStatus represents the health state of a provider
type HealthStatus string

const (
	HealthStatusHealthy   HealthStatus = "healthy"
	HealthStatusUnhealthy HealthStatus = "unhealthy"
	HealthStatusUnknown   HealthStatus = "unknown"
	HealthStatusDegraded  HealthStatus = "degraded"
)

// ProviderHealth contains health information for a single provider
type ProviderHealth struct {
	ProviderID       string        `json:"provider_id"`
	Status           HealthStatus  `json:"status"`
	LastCheck        time.Time     `json:"last_check"`
	LastSuccess      time.Time     `json:"last_success,omitempty"`
	LastError        string        `json:"last_error,omitempty"`
	ConsecutiveFails int           `json:"consecutive_fails"`
	ConsecutiveOK    int           `json:"consecutive_ok"`
	Latency          time.Duration `json:"latency,omitempty"`
	CheckCount       int64         `json:"check_count"`
	SuccessCount     int64         `json:"success_count"`
	FailureCount     int64         `json:"failure_count"`
}

// HealthMonitorConfig configures the health monitor
type HealthMonitorConfig struct {
	CheckInterval      time.Duration
	HealthyThreshold   int
	UnhealthyThreshold int
	Timeout            time.Duration
	Enabled            bool
}

// DefaultHealthMonitorConfig returns sensible defaults
func DefaultHealthMonitorConfig() HealthMonitorConfig {
	return HealthMonitorConfig{
		CheckInterval:      30 * time.Second,
		HealthyThreshold:   2,
		UnhealthyThreshold: 3,
		Timeout:            10 * time.Second,
		Enabled:            true,
	}
}

// HealthMonitor monitors the health of multiple LLM providers
type HealthMonitor struct {
	mu        sync.RWMutex
	providers map[string]provider.LLMProvider
	health    map[string]*ProviderHealth
	config    HealthMonitorConfig
	ctx       context.Context
	cancel    context.CancelFunc
	running   bool
	listeners []HealthListener
}

// HealthListener is called when provider health changes
type HealthListener func(providerID string, oldStatus, newStatus HealthStatus)

// NewHealthMonitor creates a new health monitor
func NewHealthMonitor(config HealthMonitorConfig) *HealthMonitor {
	return &HealthMonitor{
		providers: make(map[string]provider.LLMProvider),
		health:    make(map[string]*ProviderHealth),
		config:    config,
		listeners: make([]HealthListener, 0),
	}
}

// NewDefaultHealthMonitor creates a health monitor with default config
func NewDefaultHealthMonitor() *HealthMonitor {
	return NewHealthMonitor(DefaultHealthMonitorConfig())
}

// RegisterProvider registers a provider for health monitoring
func (hm *HealthMonitor) RegisterProvider(providerID string, p provider.LLMProvider) {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	hm.providers[providerID] = p
	hm.health[providerID] = &ProviderHealth{
		ProviderID: providerID,
		Status:     HealthStatusUnknown,
	}
}

// UnregisterProvider removes a provider from health monitoring
func (hm *HealthMonitor) UnregisterProvider(providerID string) {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	delete(hm.providers, providerID)
	delete(hm.health, providerID)
}

// AddListener adds a listener for health status changes
func (hm *HealthMonitor) AddListener(listener HealthListener) {
	hm.mu.Lock()
	defer hm.mu.Unlock()
	hm.listeners = append(hm.listeners, listener)
}

// Start begins the health monitoring loop
func (hm *HealthMonitor) Start() {
	hm.mu.Lock()
	if hm.running || !hm.config.Enabled {
		hm.mu.Unlock()
		return
	}
	hm.running = true
	hm.ctx, hm.cancel = context.WithCancel(context.Background())
	hm.mu.Unlock()

	hm.checkAllProviders()
	go hm.monitorLoop()
}

// Stop stops the health monitoring loop
func (hm *HealthMonitor) Stop() {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	if !hm.running {
		return
	}
	hm.running = false
	if hm.cancel != nil {
		hm.cancel()
	}
}

// IsRunning returns true if the monitor is running
func (hm *HealthMonitor) IsRunning() bool {
	hm.mu.RLock()
	defer hm.mu.RUnlock()
	return hm.running
}

func (hm *HealthMonitor) monitorLoop() {
	ticker := time.NewTicker(hm.config.CheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-hm.ctx.Done():
			return
		case <-ticker.C:
			hm.checkAllProviders()
		}
	}
}

func (hm *HealthMonitor) checkAllProviders() {
	hm.mu.RLock()
	providers := make(map[string]provider.LLMProvider, len(hm.providers))
	for id, p := range hm.providers {
		providers[id] = p
	}
	hm.mu.RUnlock()

	var wg sync.WaitGroup
	for id, p := range providers {
		wg.Add(1)
		go func(providerID string, prov provider.LLMProvider) {
			defer wg.Done()
			hm.checkProvider(providerID, prov)
		}(id, p)
	}
	wg.Wait()
}

func (hm *HealthMonitor) checkProvider(providerID string, p provider.LLMProvider) {
	// hm.ctx is only set by Start(). ForceCheck() may be invoked before Start()
	// (a one-shot on-demand check is a legitimate public-API use), so fall back
	// to context.Background() rather than panicking on a nil parent context.
	parent := hm.ctx
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithTimeout(parent, hm.config.Timeout)
	defer cancel()

	start := time.Now()

	errChan := make(chan error, 1)
	go func() {
		errChan <- p.HealthCheck()
	}()

	var err error
	select {
	case <-ctx.Done():
		err = ctx.Err()
	case err = <-errChan:
	}

	latency := time.Since(start)

	hm.mu.Lock()
	defer hm.mu.Unlock()

	health, exists := hm.health[providerID]
	if !exists {
		return
	}

	oldStatus := health.Status
	health.LastCheck = time.Now()
	health.Latency = latency
	health.CheckCount++

	if err != nil {
		health.ConsecutiveFails++
		health.ConsecutiveOK = 0
		health.FailureCount++
		health.LastError = err.Error()

		if health.ConsecutiveFails >= hm.config.UnhealthyThreshold {
			health.Status = HealthStatusUnhealthy
		} else if health.Status == HealthStatusHealthy {
			health.Status = HealthStatusDegraded
		}
	} else {
		health.ConsecutiveFails = 0
		health.ConsecutiveOK++
		health.SuccessCount++
		health.LastSuccess = time.Now()
		health.LastError = ""

		if health.ConsecutiveOK >= hm.config.HealthyThreshold || health.Status == HealthStatusUnknown {
			health.Status = HealthStatusHealthy
		}
	}

	if oldStatus != health.Status {
		for _, listener := range hm.listeners {
			go listener(providerID, oldStatus, health.Status)
		}
	}
}

// GetHealth returns the health status of a specific provider
func (hm *HealthMonitor) GetHealth(providerID string) (*ProviderHealth, bool) {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	health, exists := hm.health[providerID]
	if !exists {
		return nil, false
	}
	cp := *health
	return &cp, true
}

// GetAllHealth returns health status for all providers
func (hm *HealthMonitor) GetAllHealth() map[string]*ProviderHealth {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	result := make(map[string]*ProviderHealth, len(hm.health))
	for id, health := range hm.health {
		cp := *health
		result[id] = &cp
	}
	return result
}

// GetHealthyProviders returns IDs of all healthy providers
func (hm *HealthMonitor) GetHealthyProviders() []string {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	var healthy []string
	for id, health := range hm.health {
		if health.Status == HealthStatusHealthy {
			healthy = append(healthy, id)
		}
	}
	return healthy
}

// IsHealthy returns true if the specified provider is healthy
func (hm *HealthMonitor) IsHealthy(providerID string) bool {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	health, exists := hm.health[providerID]
	if !exists {
		return false
	}
	return health.Status == HealthStatusHealthy
}

// RecordSuccess manually records a successful operation for a provider
func (hm *HealthMonitor) RecordSuccess(providerID string) {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	health, exists := hm.health[providerID]
	if !exists {
		return
	}

	health.ConsecutiveFails = 0
	health.ConsecutiveOK++
	health.SuccessCount++
	health.LastSuccess = time.Now()

	if health.Status != HealthStatusHealthy && health.ConsecutiveOK >= hm.config.HealthyThreshold {
		oldStatus := health.Status
		health.Status = HealthStatusHealthy
		for _, listener := range hm.listeners {
			go listener(providerID, oldStatus, HealthStatusHealthy)
		}
	}
}

// RecordFailure manually records a failed operation for a provider
func (hm *HealthMonitor) RecordFailure(providerID string, err error) {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	health, exists := hm.health[providerID]
	if !exists {
		return
	}

	health.ConsecutiveFails++
	health.ConsecutiveOK = 0
	health.FailureCount++
	if err != nil {
		health.LastError = err.Error()
	}

	if health.ConsecutiveFails >= hm.config.UnhealthyThreshold {
		oldStatus := health.Status
		if oldStatus != HealthStatusUnhealthy {
			health.Status = HealthStatusUnhealthy
			for _, listener := range hm.listeners {
				go listener(providerID, oldStatus, HealthStatusUnhealthy)
			}
		}
	} else if health.Status == HealthStatusHealthy {
		oldStatus := health.Status
		health.Status = HealthStatusDegraded
		for _, listener := range hm.listeners {
			go listener(providerID, oldStatus, HealthStatusDegraded)
		}
	}
}

// GetAggregateHealth returns overall system health summary
func (hm *HealthMonitor) GetAggregateHealth() AggregateHealth {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	agg := AggregateHealth{
		TotalProviders: len(hm.health),
		Providers:      make(map[string]HealthStatus),
	}

	for id, health := range hm.health {
		agg.Providers[id] = health.Status
		switch health.Status {
		case HealthStatusHealthy:
			agg.HealthyProviders++
		case HealthStatusDegraded:
			agg.DegradedProviders++
		case HealthStatusUnhealthy:
			agg.UnhealthyProviders++
		case HealthStatusUnknown:
			agg.UnknownProviders++
		}
	}

	if agg.HealthyProviders == agg.TotalProviders {
		agg.OverallStatus = HealthStatusHealthy
	} else if agg.UnhealthyProviders == agg.TotalProviders {
		agg.OverallStatus = HealthStatusUnhealthy
	} else if agg.HealthyProviders > 0 {
		agg.OverallStatus = HealthStatusDegraded
	} else {
		agg.OverallStatus = HealthStatusUnknown
	}

	return agg
}

// AggregateHealth contains overall health summary
type AggregateHealth struct {
	OverallStatus      HealthStatus            `json:"overall_status"`
	TotalProviders     int                     `json:"total_providers"`
	HealthyProviders   int                     `json:"healthy_providers"`
	DegradedProviders  int                     `json:"degraded_providers"`
	UnhealthyProviders int                     `json:"unhealthy_providers"`
	UnknownProviders   int                     `json:"unknown_providers"`
	Providers          map[string]HealthStatus `json:"providers"`
}

// ForceCheck forces an immediate health check for a specific provider
func (hm *HealthMonitor) ForceCheck(providerID string) error {
	hm.mu.RLock()
	p, exists := hm.providers[providerID]
	hm.mu.RUnlock()

	if !exists {
		return nil
	}

	hm.checkProvider(providerID, p)
	return nil
}

// GetConfig returns the current monitor configuration
func (hm *HealthMonitor) GetConfig() HealthMonitorConfig {
	return hm.config
}
