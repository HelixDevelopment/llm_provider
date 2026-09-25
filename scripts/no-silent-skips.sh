#!/usr/bin/env bash
# Fail if a test skip directive carries no reviewed SKIP-OK annotation.
#
# A skipped test is invisible debt. This gate makes that debt loud.
#
# ── WHAT CHANGED ON 2026-09-01, AND WHY ──────────────────────────────────────
#
# This gate used to demand `SKIP-OK: #<digits>`, a ticket reference. Measured
# on 2026-09-01: the repository contained 54 skip directives and 54 SKIP-OK
# annotations, and the regex matched NONE of them — every annotation in the tree
# is a categorical slug (`#short-mode`, `#opt-in-live-probe`, …), not a number.
# The gate was therefore permanently red at 54, and `make ci-validate-all` had
# already routed around it through `no-silent-skips-warn` (WARN_ONLY=1). A gate
# whose required form nothing in the tree uses is not enforcing a rule; it is
# a red light everyone has learned to drive through, and that is worse than no
# gate at all because its redness carries no information.
#
# Three repairs were available and two were rejected:
#
#   RETIRE IT — rejected. The rule is real: a silently skipped test is a test
#   suite lying about its coverage. The implementation was wrong, not the rule.
#
#   RENUMBER THE ANNOTATIONS to `#<digits>` — rejected, and this is the one that
#   looks reasonable. This repository has no issue tracker wired to it, so the
#   54 numbers would have to be invented. That produces 54 annotations that LOOK
#   traceable and reference nothing — a bluff in exactly the shape §11.4 forbids,
#   and strictly worse than the slugs, which at least say WHY.
#
#   FIX THE REGEX — adopted, but NOT as a wildcard. Widening to
#   `SKIP-OK: #[A-Za-z0-9_-]+` would accept `#todo`, `#x`, `#later`: the gate
#   would go green and stop meaning anything, which is how a gate dies quietly
#   instead of loudly. So the accepted annotations are a CLOSED VOCABULARY,
#   declared below. A new skip must be argued into one of these categories, or
#   the vocabulary must be extended by an edit somebody reviews. That is the
#   same closed-vocabulary discipline the umbrella's check-registry.tsv uses for
#   its row types.
#
# ── DEBT IS ACCEPTED BUT NEVER SILENT ────────────────────────────────────────
#
# `#legacy-untriaged` is in the vocabulary because refusing it would only push
# those skips back to unannotated. But accepting it silently would launder the
# debt, so it is counted and printed on EVERY run, the way the umbrella's
# verify-check-registry.sh prints its DEBT rows and still exits 0. A DEBT line
# is not compliance and must never be reported as one. `--strict` makes it fail.
#
# ── THE ANNOTATION MUST BE ON THE SKIP'S OWN LINE ────────────────────────────
#
# This gate is line-based on purpose: crediting an annotation found N lines away
# would let one annotation cover a different, unannotated skip below it. Two
# ai21 sites had their annotation on the continuation line of a wrapped string
# and were reported as violations — correctly. They were fixed by moving the
# annotation to the `t.Skip(` line, not by teaching the gate to look further.
#
# ── EXIT CODES — three-valued, and 2 is NEVER a pass ─────────────────────────
#   0  every skip carries a vocabulary annotation (debt may still be printed)
#   1  at least one skip is unannotated or uses an unknown category
#   2  COULD NOT DETERMINE — the scan could not be performed at all
#
# Usage:
#   scripts/no-silent-skips.sh                 # report; debt does not fail
#   scripts/no-silent-skips.sh --strict        # declared debt fails too
#   scripts/no-silent-skips.sh --prove-failure # §1.1 paired mutation proof
#
# Env:
#   NO_SILENT_SKIPS_WARN_ONLY=1   — log violations but exit 0 (transition mode)
#   NO_SILENT_SKIPS_EXCLUDES=...  — colon-separated extra directory names to skip

set -uo pipefail
cd "$(dirname "$0")/.." || { echo "no-silent-skips: cannot reach repo root" >&2; exit 2; }

MODE=scan
STRICT=0
ROOT=.
for arg in "$@"; do
  case "$arg" in
    --strict)        STRICT=1 ;;
    --prove-failure) MODE=prove ;;
    --help|-h)       sed -n '2,68p' "$0"; exit 0 ;;
    --*) echo "no-silent-skips: unknown option '$arg'" >&2; exit 2 ;;
    # A bare argument is the ROOT to scan. It exists so --prove-failure can
    # point this gate at a throwaway fixture tree: a proof that ran against the
    # real repository would measure the repository, not the gate.
    *)   ROOT=$arg ;;
  esac
done
[ -d "$ROOT" ] || { echo "no-silent-skips: '$ROOT' is not a directory" >&2; exit 2; }

command -v grep >/dev/null 2>&1 || { echo "no-silent-skips: no grep on PATH" >&2; exit 2; }

# ── The closed vocabulary ────────────────────────────────────────────────────
# Each entry is a category an annotation may name, with what it asserts.
#   short-mode          the test is excluded from `go test -short`; it runs in
#                       the full suite, so the coverage is not lost
#   opt-in-live-probe   the test calls a real vendor endpoint and needs a real
#                       credential; it must never be part of the default suite
#   integration-mode-only  needs a running dependency the unit suite has no
#                       business starting
#   env                 a host capability the test needs is absent, and its
#                       absence is not a defect in the subject
#   legacy-untriaged    DEBT: nobody has yet decided why this skips. Counted
#                       and printed on every run; --strict makes it fail.
VOCABULARY=(short-mode opt-in-live-probe integration-mode-only env legacy-untriaged)
DEBT_CATEGORIES=(legacy-untriaged)

vocab_alternation() { local IFS='|'; printf '%s' "${VOCABULARY[*]}"; }
debt_alternation()  { local IFS='|'; printf '%s' "${DEBT_CATEGORIES[*]}"; }

PATTERNS='t\.Skip\(|@Ignore\b|\bxit\(|\.skip\(|@pytest\.mark\.skip|@unittest\.skip|#\[ignore\]|XCTSkipIf'
INCLUDES=(--include='*.go' --include='*.kt' --include='*.kts' --include='*.java'
          --include='*.ts' --include='*.tsx' --include='*.js' --include='*.jsx'
          --include='*.py' --include='*.swift' --include='*.rs')

EXCLUDES=(--exclude-dir=.git --exclude-dir=vendor --exclude-dir=node_modules
          --exclude-dir=external --exclude-dir=target --exclude-dir=build
          --exclude-dir=.gradle --exclude-dir=.idea --exclude-dir=dist
          --exclude-dir=releases --exclude-dir=reports --exclude-dir=test-results
          --exclude-dir=.next --exclude-dir=.nuxt --exclude-dir=coverage
          --exclude-dir=.venv --exclude-dir=__pycache__)

if [ -n "${NO_SILENT_SKIPS_EXCLUDES:-}" ]; then
  IFS=':' read -r -a extras <<< "$NO_SILENT_SKIPS_EXCLUDES"
  for d in "${extras[@]}"; do
    [ -n "$d" ] && EXCLUDES+=("--exclude-dir=$d")
  done
fi

# ── The scan ─────────────────────────────────────────────────────────────────
# Emits: <count-of-skips> <count-of-debt> <violations-text>
scan() {
  local root=${1:-.}
  local all annotated
  all=$(grep -rnE "$PATTERNS" "${INCLUDES[@]}" "${EXCLUDES[@]}" "$root" 2>/dev/null || true)

  # A scan that finds NOTHING is not a pass: this repository is known to contain
  # skip directives, so an empty result means the scan itself is blind — a wrong
  # root, a swallowed error, an --include list that stopped matching. Report it
  # as COULD NOT DETERMINE rather than printing OK.
  if [ -z "$all" ]; then
    echo "no-silent-skips: the scan matched ZERO skip directives anywhere." >&2
    echo "That is not a pass — it means the scan is blind. Check the includes," >&2
    echo "the excludes, and that '$root' is the repository root." >&2
    return 2
  fi

  annotated=$(printf '%s\n' "$all" | grep -E "SKIP-OK: #($(vocab_alternation))\b" || true)
  VIOLATIONS=$(printf '%s\n' "$all" | grep -vE "SKIP-OK: #($(vocab_alternation))\b" || true)
  DEBT=$(printf '%s\n' "$annotated" | grep -E "SKIP-OK: #($(debt_alternation))\b" || true)
  TOTAL=$(printf '%s\n' "$all" | grep -c . )
  return 0
}

# ── §1.1 PAIRED MUTATION PROOF ───────────────────────────────────────────────
# Seed each defect this gate is supposed to catch into a temporary tree and
# require rc=1; then require rc=0 on a clean fixture. A gate never seen to fail
# is not evidence.
if [ "$MODE" = prove ]; then
  tmp=$(mktemp -d) || { echo "prove: mktemp failed" >&2; exit 2; }
  trap 'rm -rf "$tmp"' EXIT
  rc=0

  self=$PWD/scripts/no-silent-skips.sh

  probe() { # <label> <file-content> <expected-rc>
    local label=$1 content=$2 want=$3 got
    rm -rf "$tmp/x"; mkdir -p "$tmp/x"
    printf '%s\n' "$content" > "$tmp/x/probe_test.go"
    bash "$self" "$tmp/x" >/dev/null 2>&1
    got=$?
    if [ "$got" = "$want" ]; then
      echo "PASS  $label (rc=$got as required)"
    else
      echo "FAIL  $label (rc=$got, required $want)"; rc=1
    fi
  }
  probe "an UNANNOTATED skip must fail" \
        'func TestX(t *testing.T) { t.Skip("no reason given") }' 1
  probe "a skip annotated with the OLD ticket form must fail (that form is gone)" \
        'func TestX(t *testing.T) { t.Skip("x") } // SKIP-OK: #1234' 1
  probe "a skip annotated OUTSIDE the vocabulary must fail (no wildcard)" \
        'func TestX(t *testing.T) { t.Skip("x") } // SKIP-OK: #whatever' 1
  probe "an annotation on a DIFFERENT line must not be credited" \
        'func TestX(t *testing.T) {
	t.Skip("x")
	// SKIP-OK: #short-mode
}' 1
  probe "a vocabulary annotation on the skip line must pass" \
        'func TestX(t *testing.T) { t.Skip("x") } // SKIP-OK: #short-mode' 0
  probe "a DEBT annotation passes by default but is counted" \
        'func TestX(t *testing.T) { t.Skip("x") } // SKIP-OK: #legacy-untriaged' 0

  # A tree with no skip directives at all must be UNDETERMINED, not OK.
  rm -rf "$tmp/x"; mkdir -p "$tmp/x"
  printf 'func TestX(t *testing.T) {}\n' > "$tmp/x/probe_test.go"
  bash "$self" "$tmp/x" >/dev/null 2>&1
  if [ $? = 2 ]; then
    echo "PASS  a scan that matches nothing is UNDETERMINED, not a pass (rc=2)"
  else
    echo "FAIL  a scan that matches nothing did not return 2"; rc=1
  fi

  echo
  [ "$rc" = 0 ] && echo "no-silent-skips --prove-failure: all probes behaved as required" \
                || echo "no-silent-skips --prove-failure: the gate does NOT catch what it claims"
  exit "$rc"
fi

# ── Report ───────────────────────────────────────────────────────────────────
VIOLATIONS=; DEBT=; TOTAL=0
scan "$ROOT" || exit 2

if [ -n "$VIOLATIONS" ]; then
  count=$(printf '%s\n' "$VIOLATIONS" | grep -c .)
  {
    echo "❌ $count of $TOTAL skip directive(s) carry no reviewed SKIP-OK annotation."
    echo
    printf '%s\n' "$VIOLATIONS" | head -30
    [ "$count" -gt 30 ] && echo "... ($((count - 30)) more)"
    echo
    echo "Annotate the skip line itself with one of:"
    for v in "${VOCABULARY[@]}"; do echo "    // SKIP-OK: #$v"; done
    echo "or remove the skip. The vocabulary is CLOSED: extend it in this script,"
    echo "with the reason, rather than inventing a category at the call site."
  } >&2
  if [ "${NO_SILENT_SKIPS_WARN_ONLY:-0}" = "1" ]; then
    echo "(warn-only mode — set NO_SILENT_SKIPS_WARN_ONLY=0 to fail the build)" >&2
    exit 0
  fi
  exit 1
fi

debt_count=0
[ -n "$DEBT" ] && debt_count=$(printf '%s\n' "$DEBT" | grep -c .)
if [ "$debt_count" -gt 0 ]; then
  {
    echo "⚠️  $debt_count of $TOTAL skip(s) are DECLARED DEBT (#$(debt_alternation))."
    echo "   Declared is not resolved. Each one is a test nobody has triaged:"
    printf '%s\n' "$DEBT" | head -20
    [ "$debt_count" -gt 20 ] && echo "   ... ($((debt_count - 20)) more)"
  } >&2
  if [ "$STRICT" = 1 ]; then
    echo "   --strict: declared debt is a failure." >&2
    exit 1
  fi
fi

echo "no-silent-skips: OK — $TOTAL skip directive(s), all annotated from the closed vocabulary ($debt_count declared debt)"
exit 0
