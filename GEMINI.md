# GEMINI.md — LLMProvider

## INHERITED FROM constitution/GEMINI.md

**The inheritance below is conditional. Both cases are stated; neither is
assumed.**

When this module is consumed inside a project that includes the Helix
Constitution submodule, the rules in `constitution/GEMINI.md` — and in the
`constitution/Constitution.md` it references — are authoritative for every
topic not covered here. The module-local rules below extend them; they never
weaken or override them.

When this module is consumed standalone — cloned on its own, with no
constitution reachable in any parent — there is nothing to inherit, and **only
the module-local rules below apply**.

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

Canonical reference:
<https://github.com/HelixDevelopment/HelixConstitution>

## Module-local notes

This carrier is read by Gemini CLI.

See [`README.md`](README.md) for what this module is and how it is used.
Module-specific rules go below this line; they extend the inherited base rules
and never weaken them.

### What this module is

`digital.vasic.llmprovider` is a standalone, reusable Go library: **one
abstraction over many LLM backends**, plus the operational primitives needed to
call them safely. It has no `main`, no server and no configuration file of its
own, and it is deliberately project-not-aware (§11.4.28(B)) — it carries no
consuming project's vocabulary, domain types or filesystem paths, and none may
be added.

The contract every backend satisfies is a five-method interface:

```go
type LLMProvider interface {
    Complete(ctx context.Context, req *models.LLMRequest) (*models.LLMResponse, error)
    CompleteStream(ctx context.Context, req *models.LLMRequest) (<-chan *models.LLMResponse, error)
    HealthCheck() error
    GetCapabilities() *models.ProviderCapabilities
    ValidateConfig(config map[string]interface{}) (bool, []string)
}
```

Package map, measured in this tree — re-derive with `ls pkg/` and
`go build ./...`:

| Package | What it holds |
|---|---|
| `.` (`package llmprovider`) | the `LLMProvider` interface plus the original single-package circuit breaker, health monitor and retry implementations |
| `pkg/provider` | the same contract as a package-scoped interface for adapters to import |
| `pkg/providers/<name>` | 43 concrete backend adapters, one directory per backend |
| `pkg/circuit` | circuit breaker — closed / open / half-open, with configurable thresholds |
| `pkg/health` | health monitoring across providers — healthy / degraded / unhealthy / unknown |
| `pkg/models` | the shared request / response / capability types the interface speaks |
| `pkg/discovery` | tiered dynamic model discovery |
| `pkg/apikeys` | this module's single authority for resolving per-provider credentials from the environment |
| `pkg/http` | HTTP client with retry for provider APIs |
| `pkg/i18n` | YAML-backed message bundles and translator for provider-facing strings |

Two of the 43 adapters matter to a consumer who would rather not write a new
one: `pkg/providers/generic` speaks the OpenAI-compatible wire format, so any
backend that also speaks it needs no new package at all, and
`pkg/providers/ollama` targets a locally hosted runtime.

### Module-local rules

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
   hardcoded, never logged and never committed (§11.4.10). Test fixtures carry
   synthetic keys only, and no fixture may be derived from real user content.
5. `pkg/circuit`, `pkg/health` and the retry helpers are safe for concurrent
   use. Any new exported symbol either is safe too, or says plainly in its doc
   comment that it is not.
6. Every claim about this module carries the command that re-derives it and
   that command's output (§11.4, §11.4.6). "The tests pass" is not evidence
   that a backend works; a run against it is.

### Build and test

```bash
go build ./...
go test ./... -count=1 -race
go vet ./...
gofmt -l .
```

### Honest boundaries (§11.4.6)

- The cascaded constitutional corpus that the agent carriers of this repository
  previously each carried inline is **not** reproduced here. It is inherited by
  reference through the pointer above (§11.4.28 / §11.4.177) and additionally
  retained in this repository's own `CONSTITUTION.md`. Nothing was dropped from
  the repository; it stopped being duplicated four times.
- The committed `.html` and `.pdf` exports of the agent carriers were last
  regenerated at commit `e65e431` while their Markdown sources moved on at
  `949a575`. They were therefore already stale before this rewrite, and this
  change does not regenerate them — this repository ships no export generator.
  Re-derive with `git log -1 --format=%h -- <file>`.
- The package table above is measured; the pre-existing carrier text that
  described this module as having exactly one package, with `types.go` empty
  and no `pkg/` tree, is withdrawn rather than restated.
