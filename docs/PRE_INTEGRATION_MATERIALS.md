# LLMProvider — Pre-Integration Materials

**Revision:** 1
**Last modified:** 2026-07-15T11:19:00Z
**Purpose:** Consolidated pre-integration materials (gate before any integration/deployment work).

> Scope: this document consolidates and verifies the pre-integration materials
> for the `digital.vasic.llmprovider` Go sub-system located at
> `tools/helixqa/llm_provider`. Every statement below is grounded in the real
> repository files cited inline. Where a fact could not be determined from the
> repository it is marked `UNKNOWN:`. No credential values appear here — the
> API-key interface is described by environment-variable NAME only.

---

## 1. Purpose / What it is

`digital.vasic.llmprovider` is a **generic, reusable Go module providing LLM
provider abstractions and utilities** (`doc.go:1-3`, `README.md`). It defines the
core `LLMProvider` interface plus common fault-tolerance/observability patterns:
circuit breaker, health monitoring, retry with exponential backoff, and lazy
loading (`doc.go:5-11`, `provider.go:9-15`).

It is a **library consumed by a parent project**, not a standalone service. Per
`docs/ARCHITECTURE.md` ("Integration with the parent project"), the parent
registers **43 LLM providers**, each wrapped with a circuit breaker, and uses the
health monitor for startup verification and the circuit-breaker status for
ensemble/debate routing.

**Provider backends implemented** — 43 packages under `pkg/providers/`
(`ls pkg/providers/`, count = 43):

`ai21`, `anthropic`, `cerebras`, `chutes`, `claude`, `cloudflare`, `codestral`,
`cohere`, `deepseek`, `fireworks`, `gemini`, `generic`, `githubmodels`, `groq`,
`huggingface`, `hyperbolic`, `junie`, `kilo`, `kimi`, `mistral`, `modal`, `nia`,
`nlpcloud`, `novita`, `nvidia`, `ollama`, `openai`, `openrouter`, `perplexity`,
`publicai`, `qwen`, `replicate`, `sambanova`, `sarvam`, `siliconflow`, `together`,
`upstage`, `venice`, `vulavula`, `xai`, `zai`, `zen`, `zhipu`.

Each backend implements the `LLMProvider` interface (e.g.
`pkg/providers/claude/claude.go:840` `func (p *ClaudeProvider) HealthCheck()`,
`pkg/providers/generic/generic.go:270`). A `generic` backend
(`pkg/providers/generic/generic.go`) provides a reusable OpenAI-compatible base.

---

## 2. Architecture overview

The module is layered: an application registers providers, each wrapped by a
`CircuitBreaker`, with a `HealthMonitor` observing all of them (`docs/ARCHITECTURE.md`
"High-Level Architecture").

**Provider interface** (`provider.go:9-15`):

```go
type LLMProvider interface {
    Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error)
    CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error)
    HealthCheck() error
    GetCapabilities() *models.ProviderCapabilities
    ValidateConfig(config map[string]interface{}) (bool, []string)
}
```

**Circuit breaker** (`circuit_breaker.go:14-52`, also mirrored under
`pkg/circuit/circuit_breaker.go`): three states — `closed` / `open` / `half_open`
(`circuit_breaker.go:17-21`). `DefaultCircuitBreakerConfig()` returns
`FailureThreshold=5, SuccessThreshold=2, Timeout=30s, HalfOpenMaxRequests=3`
(`circuit_breaker.go:44-52`). Errors `ErrCircuitOpen` and
`ErrCircuitHalfOpenRejected` short-circuit requests when the provider is failing
(`circuit_breaker.go:30-34`). State-change listeners are notified in separate
goroutines with a 5-second timeout (`circuit_breaker.go:24-28`, `docs/ARCHITECTURE.md`).
An empty `CompleteStream` (no responses) is treated as a failure (`docs/ARCHITECTURE.md`).

**Health monitor** (`health_monitor.go:8-16`): statuses `healthy` / `unhealthy` /
`unknown` / `degraded`. Runs concurrent per-provider checks with individual
timeouts, and exposes `RecordSuccess` / `RecordFailure` for external contributors
(e.g. the circuit breaker) plus `GetAggregateHealth` for a system-wide summary
(`docs/ARCHITECTURE.md` "Health Monitor").

**Retry / fallback** (`retry.go:11-35`): `DefaultRetryConfig()` =
`MaxRetries=3, InitialDelay=1s, MaxDelay=30s, Multiplier=2.0, JitterFactor=0.1`
(`retry.go:26-34`). Backoff is exponential with jitter; retryable HTTP status
codes are 429, 500, 502, 503, 504; `context.Canceled` /
`context.DeadlineExceeded` and non-429 4xx are non-retryable
(`docs/ARCHITECTURE.md` "Retry Logic"). A `RetryableHTTPClient` wraps
`http.Client` and clones the request body per attempt. Fallback/ensemble routing
itself lives in the parent project (`docs/ARCHITECTURE.md` "Integration with the
parent project"), driven by circuit-breaker status.

**Supporting packages** (`pkg/`): `pkg/http/client.go` (an outbound HTTP client
with retry, `pkg/http/client.go:1-2,13-30`), `pkg/apikeys/`, `pkg/discovery/`,
`pkg/i18n/` (locale bundles), `pkg/models/types.go` (request/response/capability
types), `pkg/provider/`, `pkg/circuit/`, `pkg/health/`, `pkg/retry/`.

> Observation (doc-vs-code, `§11.4.6`): `doc.go:72-76`, `CLAUDE.md`, and
> `docs/ARCHITECTURE.md` describe `digital.vasic.models` as an **external**
> dependency, but `go.mod` does not require it — the `models` types are the
> **internal** package `pkg/models`, imported as
> `digital.vasic.llmprovider/pkg/models` (`provider.go:5`,
> `pkg/models/types.go`). Integrators MUST treat `models` as in-tree, not as a
> separate module to vendor.

---

## 3. Dependencies

- **Submodules:** none. There is **no `.gitmodules`** file (`ls -a`, confirmed).
- **`helix-deps.yaml`:** `schema_version: 1`, `deps: []` — own-org dependencies:
  **none** (self-declared after inspecting `go.mod`/`go.sum`, per the file's
  Catalogue-Check note). `transitive_handling.recursive: true`,
  `conflict_resolution: operator-required`, `language_specific_subtree: false`.
- **Go module:** `digital.vasic.llmprovider`, `go 1.25.3` (`go.mod:1,3`).
- **Direct Go libraries** (`go.mod:5-9`):
  - `github.com/sirupsen/logrus v1.9.3` (structured logging in the breaker)
  - `github.com/stretchr/testify v1.11.1` (test dependency)
  - `gopkg.in/yaml.v3 v3.0.1` (challenge/fixture YAML)
- **Indirect** (`go.mod:11-15`): `github.com/davecgh/go-spew v1.1.1`,
  `github.com/pmezard/go-difflib v1.0.0`, `golang.org/x/sys v0.0.0-20220715151400-c0bba94af5f8`.
- **Required API-key environment-variable NAMES** (values MUST come from a
  gitignored `.env` / secret store — never committed; names only, from
  `.env.example`): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`,
  `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `QWEN_API_KEY`,
  `OPENROUTER_API_KEY`, `CEREBRAS_API_KEY`, `COHERE_API_KEY`,
  `SILICONFLOW_API_KEY`, `HUGGINGFACE_API_KEY`, `REPLICATE_API_TOKEN`,
  `XAI_API_KEY`, `PERPLEXITY_API_KEY`, `FIREWORKS_API_KEY`, `TOGETHER_API_KEY`,
  `NVIDIA_API_KEY`, `SAMBANOVA_API_KEY`. Key handling lives in `pkg/apikeys/`.
  `.env` itself is credential-bearing and MUST stay gitignored (`.gitignore`
  present in the tree).

  > Note: `.env.example` enumerates 19 key names; the module ships 43 provider
  > backends. Backends beyond the enumerated key set take their credentials via
  > per-provider config (`ValidateConfig(config map[string]interface{})`,
  > `provider.go:14`) or a provider-specific env var. `UNKNOWN:` the exhaustive
  > env-var name for every one of the 43 backends is not centralized in a single
  > repo file — determine per backend under `pkg/providers/<name>/` at
  > integration time.

---

## 4. Deploy / Distribution design

- **Form factor:** **library / Go module** — consumed by `go get` + import, not
  deployed as a standalone binary. `README.md` install: `go get digital.vasic/llmprovider`;
  authoritative module path is `digital.vasic.llmprovider` (`go.mod:1`). Usage is
  `import "digital.vasic.llmprovider"` + wrap a provider with
  `llmprovider.NewDefaultCircuitBreaker(...)` (`README.md` Quick Start,
  `doc.go:46-68`).
- **Distribution slice / upstreams** (`upstreams/*.sh`):
  - `git@github.com:HelixDevelopment/LLMProvider.git` (`upstreams/GitHub.sh`)
  - `git@github.com:vasic-digital/LLMProvider.git` (`upstreams/VasicDigitalGitHub.sh`)
  - `git@gitlab.com:vasic-digital/LLMProvider.git`
    (`upstreams/GitLab.sh`, `upstreams/VasicDigitalGitLab.sh`)
  - `doc.go:83` cites repository `https://github.com/vasic-digital/llmprovider`.
  The module has a `vasic-digital` twin and a `HelixDevelopment` twin that share
  source parity in `pkg/circuit`/`pkg/health`/`pkg/retry`/`pkg/models` but keep
  divergent git histories (`README.md` "Cascade").
- **No container artifacts:** there is **no `Dockerfile*` and no `*compose*` file**
  in the tree (searched; only a `Makefile` is present). Containerization, if
  required by the integration target, is `UNKNOWN:` not provided by this module.
- **License:** `doc.go:82` states MIT; a `LICENSE` file is present at the module
  root. `README.md` "License" defers to the parent project.

---

## 5. Ports

`UNKNOWN: library — no own listen port.` The module opens **no listening socket**.
There is no `http.ListenAndServe`, no `net.Listen`, and no `net/http.Server`
anywhere in the non-test Go sources (scanned). The only `func main()` is
`challenges/runner/main.go:143`, a challenge/anti-bluff harness (not a server).
The single `"/health"` string in the codebase
(`pkg/providers/zen/zen_http.go:138`) is an **outbound** `http.NewRequestWithContext(ctx, "GET", p.baseURL+"/health", ...)`
— i.e. this module probing an *upstream provider's* health endpoint, not a port
this module serves. All provider traffic is **outbound HTTPS to third-party LLM
APIs** via `pkg/http/client.go` / per-provider clients.

---

## 6. Health

Health is exposed as **library APIs, not a served endpoint** (consistent with
§5). Two mechanisms:

1. **Per-provider `HealthCheck() error`** — an interface method (`provider.go:12`)
   each backend implements as an outbound liveness probe (e.g.
   `pkg/providers/claude/claude.go:840`, `pkg/providers/deepseek/deepseek.go:659`,
   `pkg/circuit/circuit_breaker.go:182`).
2. **`HealthMonitor`** (`health_monitor.go`) — periodic concurrent checks with
   configurable thresholds; status enum `healthy` / `degraded` / `unhealthy` /
   `unknown` (`health_monitor.go:11-16`), per-provider `ProviderHealth` record
   (`health_monitor.go:19-31`), plus `GetAggregateHealth` for a system-wide roll-up
   (`docs/ARCHITECTURE.md`). Defaults: `CheckInterval`, `HealthyThreshold`,
   `UnhealthyThreshold`, `Timeout`, `Enabled` via `DefaultHealthMonitorConfig()`
   (`health_monitor.go:33-45`).

An integrating service that needs an HTTP `/health` endpoint MUST expose it
itself and delegate to `HealthMonitor.GetAggregateHealth()` — this module does not
provide one.

---

## 7. How it boots

**Consumed as a library — no standalone entrypoint.** Integration boot sequence
(`README.md` Quick Start, `doc.go:46-68`):

1. `import "digital.vasic.llmprovider"` + `.../pkg/models`.
2. Construct a concrete `LLMProvider` backend (from `pkg/providers/<name>/`),
   supplying its API key/config.
3. Wrap it: `cb := llmprovider.NewDefaultCircuitBreaker("<id>", provider)`.
4. Optionally attach a `HealthMonitor` and `RetryConfig`.
5. Call `cb.Complete(ctx, req)` / `cb.CompleteStream(ctx, req)`.

The only runnable command in the repo is the **challenge/verification harness**,
not a production entrypoint:

- `go run ./challenges/runner/` — exercises the real `circuit.CircuitBreaker` +
  `health.HealthMonitor` + `retry` across 5 locale fixtures (en/de/es/ja/sr),
  exit 0 on success (`challenges/runner/main.go:1-20`, `README.md` "How to run").
- `./challenges/llmprovider_describe_challenge.sh normal|mutate` — paired-mutation
  Challenge (exit 0 normal, 99 mutate).

**Build/test entrypoints** (`Makefile`): `make build` (`go build ./...`),
`make test` (`go test ./... -race -count=1`), `make test-core`,
`make test-providers`, `make vet`, plus DoD gates `make no-silent-skips` /
`make demo-all` (`Makefile`).

---

## 8. Materials status (verify pass)

Every material listed below was read from the repository during this pass; each
row cites the real artefact. No values were invented; no credential values are
included.

| # | Material | Present? | Evidence / citation | Notes |
|---|----------|----------|---------------------|-------|
| 1 | Purpose / interface | ✅ VERIFIED | `doc.go`, `provider.go:9-15`, `README.md` | `LLMProvider` 5-method interface |
| 2 | Architecture doc | ✅ VERIFIED | `docs/ARCHITECTURE.md` (+ `.html`/`.pdf`) | breaker/health/retry state machines |
| 3 | API reference | ✅ VERIFIED | `docs/API_REFERENCE.md` (+ `.html`/`.pdf`) | present in `docs/` |
| 4 | Examples | ✅ VERIFIED | `docs/EXAMPLES.md` (+ `.html`/`.pdf`) | present |
| 5 | Dependency manifest | ✅ VERIFIED | `helix-deps.yaml` (`deps: []`), `go.mod` | zero own-org deps; 3 direct libs |
| 6 | Submodule wiring | ✅ VERIFIED (none) | no `.gitmodules` | self-contained module |
| 7 | Credential interface | ✅ VERIFIED (names only) | `.env.example`, `pkg/apikeys/` | 19 key NAMES; `.env` gitignored |
| 8 | Provider backend set | ✅ VERIFIED | `pkg/providers/` (43 dirs) | 43 backends enumerated in §1 |
| 9 | Circuit-breaker config | ✅ VERIFIED | `circuit_breaker.go:44-52` | root default `FailureThreshold=5` |
| 10 | Retry config | ✅ VERIFIED | `retry.go:26-34` | `MaxRetries=3`, backoff 2.0, jitter 0.1 |
| 11 | Health config | ✅ VERIFIED | `health_monitor.go:8-45` | 4-state status enum |
| 12 | Ports | ✅ VERIFIED (none) | non-test source scan | library, no listener |
| 13 | Distribution / upstreams | ✅ VERIFIED | `upstreams/*.sh`, `doc.go:83` | github + gitlab, two twins |
| 14 | Container artifacts | ⚠️ NONE | no `Dockerfile*`/`compose*` | `UNKNOWN:` not provided by module |
| 15 | Anti-bluff / challenge stack | ✅ VERIFIED | `README.md`, `challenges/`, `challenges/runner/main.go` | 4-layer, paired mutation |

**Open observations to resolve at integration time (`§11.4.6` — flagged, not invented):**

- **O1 — `models` dependency framing:** `doc.go`/`CLAUDE.md`/`ARCHITECTURE.md`
  call `digital.vasic.models` external, but it is the internal `pkg/models`
  (`provider.go:5`, `go.mod` requires no such module). Treat as in-tree.
- **O2 — circuit-breaker default value:** root `DefaultCircuitBreakerConfig()`
  sets `FailureThreshold=5` (`circuit_breaker.go:47`), whereas `README.md`'s
  anti-bluff section states the `challenges/runner` fixtures assert
  `FailureThreshold=3`. The `pkg/circuit` default was not independently read in
  this pass — `UNKNOWN:` the `pkg/circuit.DefaultConfig` value; the two config
  surfaces (root package vs `pkg/circuit`) must be reconciled by the integrator.
- **O3 — full per-backend credential map:** `.env.example` enumerates 19 key
  names for 43 backends. `UNKNOWN:` the exhaustive env-var/config key for every
  backend — resolve per `pkg/providers/<name>/`.

---

### Verdict

**HAS-VERIFIED.** The pre-integration materials already exist in the repository
(README, ARCHITECTURE, API_REFERENCE, EXAMPLES, `helix-deps.yaml`, `.env.example`,
`challenges/`, `Makefile`, per-package sources) and were verified against the real
Go source during this pass. This document consolidates them and cites each claim.
Three doc-vs-code observations (O1–O3) and the container-artifact gap (row 14) are
recorded as items to reconcile at integration time — none were gap-filled with
invented data; unresolved facts are marked `UNKNOWN:`.

---

## Sources verified

All facts cite files under `tools/helixqa/llm_provider/` read on 2026-07-15:
`go.mod`, `helix-deps.yaml`, `.env.example`, `provider.go`, `doc.go`,
`circuit_breaker.go`, `retry.go`, `health_monitor.go`, `README.md`,
`docs/ARCHITECTURE.md`, `Makefile`, `pkg/http/client.go`,
`challenges/runner/main.go`, `pkg/providers/` (directory listing, 43 entries),
`upstreams/*.sh`, `pkg/providers/zen/zen_http.go`. No external network sources
were consulted; every statement is grounded in the in-repo source of truth.
