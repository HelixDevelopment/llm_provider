#!/usr/bin/env bash
# no_suspend_calls_challenge.sh — CONST-033 source-tree gate.
#
# Wraps scripts/host-power-management/check-no-suspend-calls.sh as a challenge:
# asserts this repository's tree contains zero forbidden host-power-management
# invocations. The repository root is exactly two levels above this file
# (<root>/challenges/scripts/); it is never searched for upwards, so a scanner
# that is missing from THIS repository cannot be replaced by one found in an
# enclosing directory (which used to scan, and pass, a different tree).
#
# Usage:
#   bash no_suspend_calls_challenge.sh
#   bash no_suspend_calls_challenge.sh --prove-failure
#
# Exit (three-valued; the scanner's own contract, passed through; 2 is NEVER a pass):
#   0 = clean — the scanner read the whole tree and found nothing
#   1 = violations (a violation outranks an unreadable path)
#   2 = COULD NOT DETERMINE — an unknown argument, the scanner missing from this
#       repository, or the scanner unable to read part of the tree
#
# --prove-failure (§1.1 paired proof, VSC-369; harness lib/const050b_proof.sh):
# builds throw-away repository trees under $TMPDIR holding a copy of THIS file
# and of this repository's scanner, and runs the copy: a clean tree must pass
# (0); a planted call must exit 1 naming its file, also when another file is
# unreadable; a tree whose scanner is missing (with one in an enclosing
# directory), a tree with an unreadable file and an unknown argument must exit 2.

set -uo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

PROVE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prove-failure) PROVE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "COULD NOT DETERMINE: unknown argument '$1' — this challenge takes only --prove-failure." >&2
       exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
SCANNER_REL="scripts/host-power-management/check-no-suspend-calls.sh"

prove_failure() {
  if [[ ! -r "$HERE/lib/const050b_proof.sh" ]]; then
    echo "prove-failure: cannot determine — $HERE/lib/const050b_proof.sh is missing or unreadable"
    return 2
  fi
  # shellcheck source=lib/const050b_proof.sh
  source "$HERE/lib/const050b_proof.sh"
  local real_scanner
  real_scanner="$(cd "$HERE/../.." && pwd)/$SCANNER_REL"
  if [[ ! -r "$real_scanner" ]]; then
    echo "prove-failure: cannot determine — this repository's scanner $real_scanner is missing"
    return 2
  fi
  pf_begin "$SELF" "no-suspend-calls challenge" grep || return $?
  local d c
  # ns_tree NAME [noscanner] — a planted repository tree; prints its root.
  ns_tree() {
    local r="$PF_T/case.$1/repo"
    mkdir -p "$r/challenges/scripts" "$r/src" || return 1
    cp -- "$SELF" "$r/challenges/scripts/" || return 1
    if [[ "${2-}" != noscanner ]]; then
      mkdir -p "$r/scripts/host-power-management" && cp -- "$real_scanner" "$r/$SCANNER_REL" || return 1
    fi
    printf 'echo hello\n' > "$r/src/app.sh" || return 1
    printf '%s\n' "$r"
  }
  ns_gate() { pf_run "$1/challenges/scripts/$(basename "$SELF")" "${@:2}"; }
  if d="$(ns_tree c1)"; then
    pf_expect "C1 clean tree" 0 "=== summary: PASS" ns_gate "$d"
  else pf_undet "C1" "could not build the tree"; fi
  if d="$(ns_tree m1)" && printf 'systemctl suspend\n' > "$d/src/deploy.sh"; then
    pf_expect "M1 a planted suspend call" 1 "src/deploy.sh" ns_gate "$d"
  else pf_undet "M1" "could not build the tree"; fi
  if d="$(ns_tree u1 noscanner)" && mkdir -p "$PF_T/case.u1/scripts/host-power-management" &&
     cp -- "$real_scanner" "$PF_T/case.u1/$SCANNER_REL"; then
    pf_expect "U1 scanner missing here, present in an enclosing directory" 2 "cannot locate" ns_gate "$d"
  else pf_undet "U1" "could not build the tree"; fi
  if d="$(ns_tree u2)" && chmod 000 "$d/src/app.sh"; then
    if [[ -r "$d/src/app.sh" ]]; then
      pf_undet "U2/M2" "chmod 000 does not deny reads here (running as root?)"
    else
      pf_expect "U2 an unreadable file is UNDETERMINED, not FAIL" 2 "=== summary: UNDETERMINED" ns_gate "$d"
      if d="$(ns_tree m2)" && chmod 000 "$d/src/app.sh" && printf 'loginctl hibernate\n' > "$d/src/deploy.sh"; then
        pf_expect "M2 a finding outranks an unreadable file" 1 "src/deploy.sh" ns_gate "$d"
      else pf_undet "M2" "could not build the tree"; fi
    fi
  else pf_undet "U2/M2" "could not build the tree"; fi
  if d="$(ns_tree u3)"; then
    pf_expect "U3 unknown argument" 2 "unknown argument" ns_gate "$d" -- --no-such-flag
  else pf_undet "U3" "could not build the tree"; fi
  pf_end
}

if [[ "$PROVE" -eq 1 ]]; then
  prove_failure
  exit $?
fi

# ── live challenge body ──────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
SCANNER="$PROJECT_ROOT/$SCANNER_REL"
echo "=== no_suspend_calls_challenge ==="
if [[ ! -f "$SCANNER" || ! -r "$SCANNER" ]]; then
  echo "COULD NOT DETERMINE: cannot locate $SCANNER_REL in this repository ($PROJECT_ROOT)" >&2
  echo "=== summary: UNDETERMINED (scanner missing) ==="
  exit 2
fi
echo "Scanner: $SCANNER"
echo "Root:    $PROJECT_ROOT"
echo

bash "$SCANNER" "$PROJECT_ROOT"
rc=$?
echo
case "$rc" in
  0) echo "=== summary: PASS ===" ;;
  1) echo "=== summary: FAIL ===" ;;
  *) echo "=== summary: UNDETERMINED (scanner rc $rc) — NOT a pass ==="; rc=2 ;;
esac
exit "$rc"
