# LLMProvider — Constitution

## INHERITED FROM constitution/Constitution.md

All rules in `constitution/Constitution.md` (and the `constitution/Constitution.md` it references) apply unconditionally. This file's rules below extend them — they MUST NOT weaken any inherited rule. Use `constitution/find_constitution.sh` from the parent project root to resolve the absolute path of the submodule from any nested location.

> **Inherits from:** `HelixConstitution/Constitution.md` — all universal
> clauses in that document apply unconditionally. Rules below extend or
> tighten them; they MUST NOT weaken any inherited clause.

> **Status:** Active. This document is the project's authoritative
> rule set. When a rule here conflicts with `CLAUDE.md`, `AGENTS.md`,
> or any guide, the Constitution wins.

## Mission

See README.md.

## Mandatory Standards

1. **Reproducibility:** every change is reproducible from a clean
   clone (`git clone <repo> && <project bootstrap>`); no hidden steps.
2. **Tests track behavior, not code:** test what the user-visible
   behavior is, not what the implementation looks like.
3. **No silent skips, no silent mocks above unit tests.**
4. **Conventional Commits** for all commits.
5. **SSH-only for git operations** (`git@…`); HTTPS prohibited.

## Numbered Rules

<!-- Rules are numbered CONST-NNN. New rules append. Removed rules
     keep their number with a "**Retired:** …" line. -->


## Definition of Done

A change is done when:

1. The code change is committed.
2. All project-level tests pass on a clean clone.
3. All challenges in `challenges/scripts/` pass on the running host.
4. Governance docs (`CONSTITUTION.md`, `AGENTS.md`, `CLAUDE.md`) are
   coherent with the change.

## See also

- `README.md` — project overview, quickstart.
- `AGENTS.md` — guidance for AI coding agents (Codex, Cursor, etc.).
- `CLAUDE.md` — guidance specifically for Claude Code.
- `docs/HOST_POWER_MANAGEMENT.md` — CONST-033 background and runbook.



---

## Constitutional Anti-Bluff Forensic Anchor (CONST-035 / §11.9, inherited)

> Verbatim user mandate: *"We had been in position that all tests do execute with success and all Challenges as well, but in reality the most of the features does not work and can't be used! This MUST NOT be the case and execution of tests and Challenges MUST guarantee the quality, the completion and full usability by end users of the product!"*
>
> Operative rule: **The bar for shipping is not "tests pass" but "users can use the feature."** Every PASS in this codebase MUST carry positive runtime evidence captured during execution. Metadata-only / configuration-only / absence-of-error / grep-based PASS without runtime evidence are critical defects regardless of how green the summary line looks. No false-success results are tolerable.

This anchor is inherited from the Helix Constitution (`constitution/Constitution.md` §11.9 / CONST-035); resolve it via `constitution/find_constitution.sh` from the parent project root. This submodule stays fully decoupled and project-not-aware (§11.4.28) — this is generic governance inheritance only, never project-specific context.
