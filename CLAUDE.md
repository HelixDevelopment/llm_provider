<!-- BEGIN constitution-inheritance pointer (managed) -->
## INHERITED FROM Helix Constitution

This module is a submodule of a project that includes the Helix
Constitution submodule. All rules in `constitution/CLAUDE.md` and the
`constitution/Constitution.md` it references apply unconditionally.
Locate the constitution submodule from any arbitrary nested depth
using its `find_constitution.sh` helper.

Canonical reference: https://github.com/HelixDevelopment/HelixConstitution
<!-- END constitution-inheritance pointer (managed) -->
# CLAUDE.md - LLMProvider Module

## INHERITED FROM constitution/CLAUDE.md

All rules in `constitution/CLAUDE.md` (and the `constitution/Constitution.md` it references) apply unconditionally. This file's rules below extend them — they MUST NOT weaken any inherited rule. Use `constitution/find_constitution.sh` from the parent project root to resolve the absolute path of the submodule from any nested location.

## Definition of Done

This module inherits the parent project's universal Definition of Done — see the root
agent manuals and `docs/development/definition-of-done.md`. In one line: **no
task is done without pasted output from a real run of the real system in the
same session as the change.** Coverage and green suites are not evidence.

### Acceptance demo for this module

```bash
# Circuit breaker + health monitor + retry policy for provider fault tolerance
cd LLMProvider && GOMAXPROCS=2 nice -n 19 go test -count=1 -race -v \
  -run 'TestDefaultCircuitBreakerConfig|TestHealthMonitor_|TestDefaultRetryConfig' ./pkg/...
```
Expect: PASS; breaker opens after 3 consecutive failures, recovers after cooldown. `LLMProvider/README.md` shows the full `LLMProvider` interface.

## Module Overview

`digital.vasic.llmprovider` is a generic, reusable Go module providing LLM provider abstractions and utilities. It defines the core `LLMProvider` interface and common patterns for building LLM provider implementations, including circuit breakers, health monitoring, retry logic, and lazy loading. The module is designed for AI/LLM applications that need to integrate multiple LLM providers with fault tolerance and observability.

**Module path**: `digital.vasic.llmprovider`
**Go version**: 1.25.3+
**Dependencies**: `digital.vasic.models`, `github.com/sirupsen/logrus`
**Test Dependencies**: `github.com/stretchr/testify`

## Build & Test

```bash
go build ./...
go test ./... -count=1 -race
go test ./... -short              # Unit tests only
go test -v -run TestCircuitBreaker ./...
gofmt -w .
go vet ./...
go mod tidy
go mod verify
```

## Code Style

- Standard Go conventions, `gofmt` formatting
- Imports grouped: stdlib, third-party, internal (blank line separated)
- Line length ≤ 100 characters
- Naming: `camelCase` private, `PascalCase` exported, acronyms all-caps
- Errors: always check, wrap with `fmt.Errorf("...: %w", err)`
- Tests: table-driven, `testify`, naming `Test<Struct>_<Method>_<Scenario>`

## Package Structure

| Package | Purpose |
|---------|---------|
| `llmprovider` (root) | Core types: `LLMProvider` interface, circuit breaker, health monitor, retry config, lazy provider, and associated utilities. This is the only package. |

## Dependency Graph

```
digital.vasic.models
    ↓
digital.vasic.llmprovider
    ↓
github.com/sirupsen/logrus
```

## Key Files

| File | Purpose |
|------|---------|
| `provider.go` | `LLMProvider` interface definition with `Complete`, `CompleteStream`, `HealthCheck`, `GetCapabilities`, `ValidateConfig` methods |
| `circuit_breaker.go` | Circuit breaker implementation with closed/open/half-open states, failure counting, timeout |
| `health_monitor.go` | Health monitoring with configurable thresholds, intervals, status transitions |
| `retry.go` | Retry logic with exponential backoff, jitter, HTTP status code detection |
| `types.go` | Empty file (types moved to models module) |
| `circuit_breaker_test.go` / `health_monitor_test.go` / `retry_test.go` / `types_test.go` | Corresponding test files |
| `go.mod` | Module definition and dependencies |
| `README.md` | User-facing documentation with quick start |

## Key Interfaces

- `LLMProvider`: Interface for LLM provider implementations with `Complete`, `CompleteStream`, `HealthCheck`, `GetCapabilities`, `ValidateConfig`
- `CircuitBreaker`: Wraps an `LLMProvider` with fault tolerance (closed/open/half-open states)
- `HealthMonitor`: Tracks provider health with configurable thresholds and intervals
- `RetryConfig`: Configurable retry logic with exponential backoff and jitter
- `LazyProvider`: Lazy initialization of providers with optional event publishing

## Core Components

### LLMProvider Interface

The foundational interface that all LLM provider implementations must satisfy:

```go
type LLMProvider interface {
    Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error)
    CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error)
    HealthCheck() error
    GetCapabilities() *models.ProviderCapabilities
    ValidateConfig(config map[string]interface{}) (bool, []string)
}
```

### Circuit Breaker

Prevents cascading failures when providers are unhealthy:
- **Closed**: Normal operation, requests pass through
- **Open**: Provider is failing, requests are short-circuited
- **Half-Open**: Testing if provider has recovered

### Health Monitor

Tracks provider health with:
- Configurable check intervals and timeouts
- Consecutive failure/success thresholds
- Health status transitions (healthy, degraded, unhealthy, unknown)
- Listener support for health status changes

### Retry Logic

Configurable retry with:
- Exponential backoff with configurable multiplier
- Jitter to prevent thundering herd
- HTTP status code detection (429, 500, 502, 503, 504)
- Context cancellation support

### Lazy Provider

Lazy initialization pattern:
- Deferred provider initialization until first use
- Configurable timeout and retry attempts
- Optional event bus integration for provider lifecycle events

## Dependencies

- **digital.vasic.models**: For `LLMRequest`, `LLMResponse`, `ProviderCapabilities` types
- **github.com/sirupsen/logrus**: For structured logging in circuit breaker
- **Standard library**: `context`, `sync`, `time`, `net/http`, etc.

## Thread Safety

- `CircuitBreaker` uses `sync.RWMutex` for thread-safe state management
- `HealthMonitor` uses atomic operations for status updates
- `CircuitBreakerManager` is thread-safe for concurrent provider management
- `RetryConfig` is immutable after creation
- `LazyProvider` uses `sync.Once` for thread-safe lazy initialization
- All exported methods are safe for concurrent use unless otherwise documented

## Example Usage

```go
import (
    "context"
    "digital.vasic.llmprovider"
    "digital.vasic.llmprovider/pkg/models"
)

func main() {
    provider := // create your provider implementation
    cb := llmprovider.NewDefaultCircuitBreaker("my-provider", provider)

    req := &models.LLMRequest{
        Prompt: "Hello, world!",
        MaxTokens: 100,
    }

    resp, err := cb.Complete(context.Background(), req)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(resp.Text)
}
```

## Agent Coordination Guide

### Division of Work

When multiple agents work on this module simultaneously, divide work by component categories:

1. **Interface Agent** -- Owns `LLMProvider` interface definition and related methods. Changes affect all provider implementations.
2. **Circuit Breaker Agent** -- Owns circuit breaker implementation and `CircuitBreakerManager`. Affects fault tolerance.
3. **Health Monitor Agent** -- Owns health monitoring, thresholds, status transitions.
4. **Retry Logic Agent** -- Owns retry configuration, exponential backoff, HTTP status detection.
5. **Lazy Provider Agent** -- Owns lazy initialization pattern and deferred provider creation.

### Coordination Rules

- **`LLMProvider` interface changes** require coordination with all agents as this is the foundational interface.
- **Circuit breaker behavior changes** affect fault tolerance across all providers.
- **Health monitor thresholds** affect provider health detection and automatic failover.
- **Retry logic changes** affect error recovery and rate limiting handling.
- **Lazy provider changes** affect initialization patterns and startup performance.

### Safe Parallel Changes

These changes can be made simultaneously without coordination:
- Adding new methods to existing structs (if they don't affect interface)
- Adding new configuration options with sensible defaults
- Adding new helper functions
- Updating documentation
- Adding new test cases
- Improving error messages or logging

### Changes Requiring Coordination

- Modifying `LLMProvider` interface method signatures (breaks all implementations)
- Changing circuit breaker state machine logic (affects fault tolerance)
- Modifying health monitor status transition thresholds (affects provider health detection)
- Changing retry exponential backoff formula (affects error recovery timing)
- Modifying lazy provider initialization semantics (affects startup behavior)

## Commit Conventions

Follow Conventional Commits with llmprovider scope:

```
feat(llmprovider): add new method to LLMProvider interface for batch processing
feat(circuit): add configurable failure threshold to circuit breaker
feat(health): add degraded state to health monitor
feat(retry): add jitter to exponential backoff
fix(circuit): correct race condition in state transition
test(health): add edge case tests for health monitor thresholds
docs(llmprovider): update API reference for new methods
refactor(retry): extract common backoff calculation functions
```

## Integration with the parent project

This module is extracted from the parent project's `internal/llm` package. In the parent project, provider implementations (Claude, DeepSeek, Gemini, etc.) implement the `LLMProvider` interface and use these utilities for fault tolerance and observability. Provider implementations should import this module and implement the `LLMProvider` interface; the module is designed for zero-dependency provider implementations (implementations only need this module and models). Circuit breaker and health monitor can be composed around any `LLMProvider` implementation.

## Integration Seams

| Direction | Sibling modules |
|-----------|-----------------|
| Upstream (this module imports) | Models |
| Downstream (these import this module) | DebateOrchestrator, HelixLLM |

*Siblings* means other project-owned modules at the parent project repo root. The root parent-project app and external systems are not listed here — the list above is intentionally scoped to module-to-module seams, because drift *between* sibling modules is where the "tests pass, product broken" class of bug most often lives. See the root agent manuals for the rules that keep these seams contract-tested.

## Resources

- `README.md` — user-facing documentation with quick start.
- This file's siblings (one per supported CLI) carry the same ruleset —
  keep them in sync with this file.

---

## Constitutional Anti-Bluff Forensic Anchor (CONST-035 / §11.9, inherited)

> Verbatim user mandate: *"We had been in position that all tests do execute with success and all Challenges as well, but in reality the most of the features does not work and can't be used! This MUST NOT be the case and execution of tests and Challenges MUST guarantee the quality, the completion and full usability by end users of the product!"*
>
> Operative rule: **The bar for shipping is not "tests pass" but "users can use the feature."** Every PASS in this codebase MUST carry positive runtime evidence captured during execution. Metadata-only / configuration-only / absence-of-error / grep-based PASS without runtime evidence are critical defects regardless of how green the summary line looks. No false-success results are tolerable.

This anchor is inherited from the Helix Constitution (`constitution/Constitution.md` §11.9 / CONST-035); resolve it via `constitution/find_constitution.sh` from the parent project root. This submodule stays fully decoupled and project-not-aware (§11.4.28) — this is generic governance inheritance only, never project-specific context.
