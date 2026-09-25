<!-- BEGIN constitution-inheritance pointer (managed) -->
## INHERITED FROM Helix Constitution

This module is a submodule of a project that includes the Helix
Constitution submodule. All rules in `constitution/GEMINI.md` and the
`constitution/Constitution.md` it references apply unconditionally.
Locate the constitution submodule from any arbitrary nested depth
using its `find_constitution.sh` helper.

Canonical reference: https://github.com/HelixDevelopment/HelixConstitution
<!-- END constitution-inheritance pointer (managed) -->
# GEMINI.md - LLMProvider Module

## INHERITED FROM constitution/GEMINI.md

All rules in `constitution/GEMINI.md` (and the `constitution/Constitution.md` it references) apply unconditionally. This file's rules below extend them — they MUST NOT weaken any inherited rule. Use `constitution/find_constitution.sh` from the parent project root to resolve the absolute path of the submodule from any nested location.

**The inheritance is conditional. Both cases are stated; neither is assumed.**
When this module is consumed inside a project that includes the Helix
Constitution submodule, the inherited rules are authoritative for every topic
not covered here. When this module is consumed standalone — cloned on its own,
with no constitution reachable in any parent — there is nothing to inherit, and
**only the module-local rules below apply**.

### Locating the base file: a resolver, never a path

`constitution/GEMINI.md` in the heading above is the **canonical name of the base
file**, written exactly as the constitution's own examples write it. It is not
a filesystem path relative to this module, and it must not be rewritten into
one:

- a consuming project may mount the constitution under more than one layout,
  and this module cannot know which one it got;
- the same commit of this module can be checked out at two different depths at
  the same time, so no single relative path is correct for both;
- a standalone clone has no constitution anywhere, so any hardcoded path would
  simply dangle.

Resolve it at run time with the constitution's own parent-walk resolver,
**`find_constitution.sh`**. It walks up the parent chain trying each layout the
constitution supports, follows
`git rev-parse --show-superproject-working-tree` out of nested submodules so it
works from any nested depth, and exits non-zero with an explicit error when no
constitution is reachable — which is precisely the standalone case above.

This file therefore hardcodes **no** parent-project path and **no**
depth-dependent path, keeping the module project-not-aware, decoupled and
reusable per §11.4.28(B). Agent tooling with a native file-import syntax must
not turn the heading into one: an `@constitution/GEMINI.md` import resolves
relative to *this* file, so inside a module it points at a path that does not
exist and silently resolves to nothing.

This carrier is read by Gemini CLI. Its three siblings (one per supported
CLI) carry the same body — only the self-referencing file name and this reader
line vary per file. Keep them in lockstep (§11.4.157).

## Definition of Done

This module inherits the consuming project's universal Definition of Done from
that project's root agent manuals. In one line: **no task is done without
pasted output from a real run of the real system in the same session as the
change.** Coverage and green suites are not evidence.

### Acceptance demo for this module

```bash
# Circuit breaker + health monitor + retry policy for provider fault tolerance
# (run from this module's root)
GOMAXPROCS=2 nice -n 19 go test -count=1 -race -v \
  -run 'TestDefaultCircuitBreakerConfig|TestHealthMonitor_|TestDefaultRetryConfig' ./pkg/...
```
Expect: PASS; breaker opens after 3 consecutive failures, recovers after cooldown. `README.md` shows the full `LLMProvider` interface.

## Module Overview

`digital.vasic.llmprovider` is a standalone, reusable Go library: **one
abstraction over many LLM backends**, plus the operational primitives needed to
call them safely — circuit breakers, health monitoring, retry logic, tiered
model discovery, and environment-driven credential and settings resolution. It
has no `main`, no server and no configuration file of its own, and it is
deliberately project-not-aware (§11.4.28(B)) — it carries no consuming
project's vocabulary, domain types or filesystem paths, and none may be added.

**Module path**: `digital.vasic.llmprovider`
**Go version**: 1.25.3 (per `go.mod`)
**Dependencies** (per `go.mod`, re-derive with `go list -m all`): `github.com/sirupsen/logrus`, `gopkg.in/yaml.v3`
**Test Dependencies**: `github.com/stretchr/testify`

## Build & Test

```bash
go build ./...
go test ./... -count=1 -race
go test ./... -short              # Unit tests only
go test -v -run TestCircuitBreaker ./...
gofmt -l .
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

Measured in this tree — re-derive with `ls pkg/`, `ls -d pkg/providers/*/ | wc -l`
and `go build ./...`:

| Package | What it holds |
|---------|---------------|
| `.` (`package llmprovider`) | the `LLMProvider` interface plus the original single-package circuit breaker, health monitor and retry implementations |
| `pkg/provider` | the same contract as a package-scoped interface for adapters to import |
| `pkg/providers/<name>` | 43 concrete backend adapters, one directory per backend |
| `pkg/circuit` | circuit breaker — closed / open / half-open, with configurable thresholds |
| `pkg/health` | health monitoring across providers — healthy / degraded / unhealthy / unknown |
| `pkg/retry` | retry with exponential backoff and jitter for LLM API calls |
| `pkg/models` | the shared request / response / capability types the interface speaks |
| `pkg/discovery` | tiered dynamic model discovery |
| `pkg/apikeys` | this module's single authority for resolving per-provider credentials from the environment |
| `pkg/settings` | this module's single authority for per-provider NON-credential configuration from the environment — default model, endpoint, timeout |
| `pkg/http` | HTTP client with retry for provider APIs |
| `pkg/i18n` | YAML-backed message bundles and translator for provider-facing strings |

Two of the 43 adapters matter to a consumer who would rather not write a new
one: `pkg/providers/generic` speaks the OpenAI-compatible wire format, so any
backend that also speaks it needs no new package at all, and
`pkg/providers/ollama` targets a locally hosted runtime.

## Dependency Graph

```
github.com/sirupsen/logrus   gopkg.in/yaml.v3
            ↓                       ↓
       digital.vasic.llmprovider  (root + pkg/...)
            ↑
  pkg/models ← pkg/provider ← pkg/providers/<name>
```

The root package and every adapter import `digital.vasic.llmprovider/pkg/models`
for the request / response / capability types.

## Key Files

| File | Purpose |
|------|---------|
| `provider.go` | `LLMProvider` interface definition with `Complete`, `CompleteStream`, `HealthCheck`, `GetCapabilities`, `ValidateConfig` methods |
| `circuit_breaker.go` | Circuit breaker implementation with closed/open/half-open states, failure counting, timeout, and `CircuitBreakerManager` |
| `health_monitor.go` | Health monitoring with configurable thresholds, intervals, status transitions |
| `retry.go` | Retry logic with exponential backoff, jitter, HTTP status code detection |
| `types.go` | Comment-only stub — the types formerly here live in `pkg/models` |
| `circuit_breaker_test.go` / `health_monitor_test.go` / `retry_test.go` / `types_test.go` | Corresponding test files |
| `pkg/` | The package tree tabulated above |
| `go.mod` | Module definition and dependencies |
| `README.md` | User-facing documentation with quick start |

## Key Interfaces

- `LLMProvider`: Interface for LLM provider implementations with `Complete`, `CompleteStream`, `HealthCheck`, `GetCapabilities`, `ValidateConfig`
- `CircuitBreaker`: Wraps an `LLMProvider` with fault tolerance (closed/open/half-open states); `CircuitBreakerManager` manages one per provider
- `HealthMonitor`: Tracks provider health with configurable thresholds and intervals
- `RetryConfig`: Configurable retry logic with exponential backoff and jitter

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

## Thread Safety

- `CircuitBreaker` uses `sync.RWMutex` for thread-safe state management
- `HealthMonitor` uses `sync.RWMutex` for status updates (re-derive with `grep -n 'sync\.' health_monitor.go`)
- `CircuitBreakerManager` is thread-safe for concurrent provider management
- `RetryConfig` is immutable after creation
- All exported methods are safe for concurrent use unless otherwise documented; any new exported symbol either is safe too, or says plainly in its doc comment that it is not

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

## Module-local rules

These extend the inherited base rules; they weaken nothing.

1. The `LLMProvider` interface is this module's contract. Changing a method
   signature breaks all 43 adapters in a single edit. Extend by adding a type,
   an option or a wrapper — never by widening the interface in place.
2. A new backend is a new `pkg/providers/<name>/` directory, lowercase, one
   backend per package (§11.4.29). It is never a new branch inside an existing
   adapter's `switch`.
3. Nothing consumer-specific may enter this module — no consuming project's
   domain vocabulary, no its paths, no its types (§11.4.28(B)). A feature that
   cannot be described without naming a particular consumer belongs in that
   consumer, not here.
4. Credentials are resolved at run time through `pkg/apikeys`, and are never
   hardcoded, never logged and never committed (§11.4.10). Non-credential
   defaults (model, endpoint, timeout) are resolved through `pkg/settings` with
   an environment override layer — never frozen into an adapter. Test fixtures
   carry synthetic keys only, and no fixture may be derived from real user
   content.
5. `pkg/circuit`, `pkg/health` and the retry helpers are safe for concurrent
   use. Any new exported symbol either is safe too, or says plainly in its doc
   comment that it is not.
6. Every claim about this module carries the command that re-derives it and
   that command's output (§11.4, §11.4.6). "The tests pass" is not evidence
   that a backend works; a run against it is.

## Agent Coordination Guide

### Division of Work

When multiple agents work on this module simultaneously, divide work by component categories:

1. **Interface Agent** -- Owns `LLMProvider` interface definition and related methods. Changes affect all provider implementations.
2. **Circuit Breaker Agent** -- Owns circuit breaker implementation and `CircuitBreakerManager`. Affects fault tolerance.
3. **Health Monitor Agent** -- Owns health monitoring, thresholds, status transitions.
4. **Retry Logic Agent** -- Owns retry configuration, exponential backoff, HTTP status detection.
5. **Adapter Agent(s)** -- Own one or more `pkg/providers/<name>/` adapters plus the `pkg/apikeys` / `pkg/settings` resolution they depend on.

### Coordination Rules

- **`LLMProvider` interface changes** require coordination with all agents as this is the foundational interface.
- **Circuit breaker behavior changes** affect fault tolerance across all providers.
- **Health monitor thresholds** affect provider health detection and automatic failover.
- **Retry logic changes** affect error recovery and rate limiting handling.
- **`pkg/apikeys` / `pkg/settings` resolution changes** affect every adapter's credential and default-model/endpoint/timeout behaviour.

### Safe Parallel Changes

These changes can be made simultaneously without coordination:
- Adding new methods to existing structs (if they don't affect interface)
- Adding new configuration options with sensible defaults
- Adding new helper functions
- Adding a new `pkg/providers/<name>/` adapter
- Updating documentation
- Adding new test cases
- Improving error messages or logging

### Changes Requiring Coordination

- Modifying `LLMProvider` interface method signatures (breaks all implementations)
- Changing circuit breaker state machine logic (affects fault tolerance)
- Modifying health monitor status transition thresholds (affects provider health detection)
- Changing retry exponential backoff formula (affects error recovery timing)
- Changing the `pkg/apikeys` / `pkg/settings` environment-variable conventions (affects every adapter and every operator's environment)

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

## Publishing (upstreams/)

This repository is published to more than one organisation on purpose: the
canonical lineage is `git@github.com:HelixDevelopment/LLMProvider.git`
(`upstreams/github.sh`, and this module's registered submodule URL in its
consuming projects), with `vasic-digital` mirrors on GitHub and GitLab
(`upstreams/vasic_digital_github.sh`, `upstreams/gitlab.sh`,
`upstreams/vasic_digital_gitlab.sh`). Every recipe MUST name a repository this
checkout already has as a configured remote — never a URL the repo does not
know — and every mirror is integrated by fetch + merge onto the latest canonical
tip, then fast-forward pushed; force-push is forbidden (§11.4.113).
`challenges/scripts/upstreams_recipe_origin_challenge.sh` enforces this and
derives every expectation from `git remote -v` at run time.

## Resources

- `README.md` — user-facing documentation with quick start.
- This file's siblings (one per supported CLI) carry the same ruleset —
  keep them in sync with this file.

## Honest boundaries (§11.4.6)

- The cascaded constitutional corpus that the agent carriers of this repository
  previously each carried inline is **not** reproduced here. It is inherited by
  reference through the pointer above (§11.4.28 / §11.4.177) and additionally
  retained in this repository's own `CONSTITUTION.md`. Nothing was dropped from
  the repository; it stopped being duplicated four times.
- The committed `.html` and `.pdf` exports of the agent carriers were last
  regenerated at commit `e65e431`, while their Markdown sources have moved on
  since. They are therefore stale, and this repository ships no export
  generator to regenerate them. Re-derive with
  `git log -1 --format=%h -- <file>`.
- The package table above is measured; earlier carrier text that described this
  module as having exactly one package, an empty `types.go`, a dependency on a
  `digital.vasic.models` module, and a `LazyProvider` type is withdrawn rather
  than restated — none of the four is present in this tree (re-derive with
  `ls pkg/`, `cat types.go`, `grep require -A5 go.mod`, `grep -rn LazyProvider --include=*.go .`).

---

## Constitutional Anti-Bluff Forensic Anchor (CONST-035 / §11.9, inherited)

> Verbatim user mandate: *"We had been in position that all tests do execute with success and all Challenges as well, but in reality the most of the features does not work and can't be used! This MUST NOT be the case and execution of tests and Challenges MUST guarantee the quality, the completion and full usability by end users of the product!"*
>
> Operative rule: **The bar for shipping is not "tests pass" but "users can use the feature."** Every PASS in this codebase MUST carry positive runtime evidence captured during execution. Metadata-only / configuration-only / absence-of-error / grep-based PASS without runtime evidence are critical defects regardless of how green the summary line looks. No false-success results are tolerable.

This anchor is inherited from the Helix Constitution (`constitution/Constitution.md` §11.9 / CONST-035); resolve it via `constitution/find_constitution.sh` from the parent project root. This submodule stays fully decoupled and project-not-aware (§11.4.28) — this is generic governance inheritance only, never project-specific context.
