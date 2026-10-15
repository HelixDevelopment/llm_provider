#!/usr/bin/env bash
# ui_terminal_interaction_challenge.sh — anti-bluff UI Challenge per CONST-035
# + CONST-050(B); cascaded per CONST-051(A). Drives the binary at <PREFIX>_BIN
# (<PREFIX> = C50_PREFIX in the module-identity block below) and asserts that
# no user-hostile crash text (a Go panic, a goroutine trace, a runtime error, a
# segmentation fault, a fatal error) ever reaches the terminal.
#
# Four steps, each of which must pass (the executed-step floor: a step that was
# skipped or deleted is a finding, never a quiet pass):
#   [1/4] the binary is set and executable
#   [2/4] --help (or -h) prints non-empty help without crash text
#   [3/4] --version (or -v) prints no crash text
#   [4/4] an unknown flag is refused without crash text and without hanging
#         (exit < 124 within UI_TIMEOUT_SEC)
#
# Usage:
#   bash ui_terminal_interaction_challenge.sh
#   bash ui_terminal_interaction_challenge.sh --prove-failure
#
# Exit (three-valued; a finding outranks a could-not-determine; 2 is NEVER a pass):
#   0 = the binary was exercised and never leaked crash text
#   1 = a real finding — crash text reached the terminal, help was empty, the
#       binary hung or crashed on an unknown flag, or a step was skipped
#       (executed-step floor)
#   2 = COULD NOT DETERMINE — an unknown argument, the binary unset or not an
#       executable file, UI_TIMEOUT_SEC not a positive integer, or no timeout(1).
#
# --prove-failure (§1.1 paired proof, VSC-369; harness lib/const050b_proof.sh):
# runs copies of THIS file against fake binaries. A well-behaved binary must
# pass (0); a panicking --help, a panic followed by > 64 KiB of help text (the
# SIGPIPE case a `printf | grep -q` check misses), an empty help, a fatal
# --version, a panicking unknown flag and a hanging unknown flag must each exit
# 1 naming their step; no binary, an absent binary, a non-executable file, a
# malformed timeout and an unknown argument must exit 2; a deleted step
# (planted in a copy) trips the floor (1).

set -uo pipefail

# ── module identity: the ONLY lines that differ between the copies of this file ──
C50_MODULE="LLMProvider"
C50_PREFIX="LLMPROVIDER_VD"

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

prove_failure() {
    if [[ ! -r "$HERE/lib/const050b_proof.sh" ]]; then
        echo "prove-failure: cannot determine — $HERE/lib/const050b_proof.sh is missing or unreadable"
        return 2
    fi
    # shellcheck source=lib/const050b_proof.sh
    source "$HERE/lib/const050b_proof.sh"
    pf_begin "$SELF" "UI challenge" || return $?
    local B="${C50_PREFIX}_BIN"
    local g_from c bin knobs=(UI_TIMEOUT_SEC=3)
    if c="$(pf_copy c1)" && pf_bin bin clean; then
        pf_expect "C1 well-behaved binary passes all 4 steps" 0 "UI Challenge: PASSED" pf_run "$c" "$B=$bin" "${knobs[@]}"
    else pf_undet "C1" "could not write the fake binary"; fi
    if c="$(pf_copy m1)" && pf_bin bin panic-help; then
        pf_expect "M1 --help prints a Go panic" 1 "--help leaked: panic:" pf_run "$c" "$B=$bin" "${knobs[@]}"
    else pf_undet "M1" "could not write the fake binary"; fi
    if c="$(pf_copy m2)" && pf_bin bin big-panic-help; then
        pf_expect "M2 a panic followed by > 64 KiB of help text" 1 "--help leaked: panic:" pf_run "$c" "$B=$bin" "${knobs[@]}"
    else pf_undet "M2" "could not write the fake binary"; fi
    if c="$(pf_copy m3)" && pf_bin bin empty-help; then
        pf_expect "M3 empty help" 1 "[2/4] FAIL" pf_run "$c" "$B=$bin" "${knobs[@]}"
    else pf_undet "M3" "could not write the fake binary"; fi
    if c="$(pf_copy m4)" && pf_bin bin panic-version; then
        pf_expect "M4 --version prints a fatal error" 1 "--version leaked: fatal error:" pf_run "$c" "$B=$bin" "${knobs[@]}"
    else pf_undet "M4" "could not write the fake binary"; fi
    if c="$(pf_copy m5)" && pf_bin bin panic-bogus; then
        pf_expect "M5 an unknown flag prints a Go panic" 1 "leaked: panic:" pf_run "$c" "$B=$bin" "${knobs[@]}"
    else pf_undet "M5" "could not write the fake binary"; fi
    if c="$(pf_copy m6)" && pf_bin bin hang-bogus; then
        pf_expect "M6 an unknown flag hangs" 1 "[4/4] FAIL" pf_run "$c" "$B=$bin" UI_TIMEOUT_SEC=1
    else pf_undet "M6" "could not write the fake binary"; fi
    g_from='pass_step '; g_from+='4 "Invalid-flag"'
    if c="$(pf_copy g1)" && pf_plant "$c" "$g_from" ':' && pf_bin bin clean; then
        pf_expect "G1 a deleted step trips the executed-step floor" 1 "executed-step floor" pf_run "$c" "$B=$bin" "${knobs[@]}"
    else pf_undet "G1" "could not plant the defect"; fi
    if c="$(pf_copy u1)"; then
        pf_expect "U1 no binary configured" 2 "#env-no-binary" pf_run "$c"
    else pf_undet "U1" "could not copy"; fi
    if c="$(pf_copy u2)"; then
        pf_expect "U2 the binary path does not exist" 2 "#env-binary-unusable" pf_run "$c" "$B=$PF_T/no-such-binary" "${knobs[@]}"
    else pf_undet "U2" "could not copy"; fi
    if c="$(pf_copy u3)" && pf_bin bin clean && chmod 644 "$bin"; then
        pf_expect "U3 the binary is not executable" 2 "#env-binary-unusable" pf_run "$c" "$B=$bin" "${knobs[@]}"
    else pf_undet "U3" "could not write the fake binary"; fi
    if c="$(pf_copy u4)" && pf_bin bin clean; then
        pf_expect "U4 a timeout that is not a number" 2 "#bad-knob" pf_run "$c" "$B=$bin" UI_TIMEOUT_SEC=soon
    else pf_undet "U4" "could not write the fake binary"; fi
    if c="$(pf_copy u5)"; then
        pf_expect "U5 unknown argument" 2 "unknown argument" pf_run "$c" -- --no-such-flag
    else pf_undet "U5" "could not copy"; fi
    pf_end
}

if [[ "$PROVE" -eq 1 ]]; then
    prove_failure
    exit $?
fi

# ── live challenge body ──────────────────────────────────────────────────────
BIN_VAR="${C50_PREFIX}_BIN"
BIN_PATH="${!BIN_VAR:-}"
TIMEOUT_SEC="${UI_TIMEOUT_SEC:-30}"
USER_HOSTILE=('panic:' 'goroutine [0-9]+ \[running\]:' 'runtime error:' 'segmentation fault' 'fatal error:')
STEPS=4
STEPS_OK=0
NAME="$C50_MODULE UI Challenge"

pass_step() { STEPS_OK=$((STEPS_OK + 1)); echo "[$1/$STEPS] $2: PASS${3:+ ($3)}"; }
fail_step() { echo "[$1/$STEPS] FAIL: $2"; echo "=== $NAME: FAILED ==="; exit 1; }
undetermined() {
    echo "COULD NOT DETERMINE: $1"
    echo "  Nothing (more) was judged. $NAME is NOT known to be healthy."
    echo "=== $NAME: COULD NOT DETERMINE ($2) ==="
    exit 2
}
# assert_no_panic STEP LABEL BODY — Bash's own regex, NOT `printf ... | grep
# -qE`: under `set -o pipefail`, grep -q closing the pipe on a MATCH kills
# printf with SIGPIPE (141) and pipefail promotes it, so a leak in a body over
# ~64 KiB was missed precisely because it was there. Returns 0 explicitly on a
# clean body (the old loop returned grep's 1 and failed every clean binary).
assert_no_panic() {
    local pat
    for pat in "${USER_HOSTILE[@]}"; do
        [[ $3 =~ $pat ]] && fail_step "$1" "$2 leaked: $pat"
    done
    return 0
}

echo "=== $C50_MODULE UI Terminal-Interaction Challenge ==="
echo "  bin=$BIN_PATH timeout=${TIMEOUT_SEC}s"

command -v timeout >/dev/null 2>&1 || undetermined "'timeout' is not on PATH" "#env-no-tool"
[[ -n "$BIN_PATH" ]] || undetermined "$BIN_VAR is unset — there is no binary to drive" "#env-no-binary"
[[ -f "$BIN_PATH" && -x "$BIN_PATH" ]] ||
    undetermined "$BIN_VAR='$BIN_PATH' is not an executable file" "#env-binary-unusable"
[[ "$TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]] ||
    undetermined "UI_TIMEOUT_SEC='$TIMEOUT_SEC' is not a positive integer" "#bad-knob"
pass_step 1 "Binary present"

help_out=$(timeout "$TIMEOUT_SEC" "$BIN_PATH" --help 2>&1 || timeout "$TIMEOUT_SEC" "$BIN_PATH" -h 2>&1 || true)
assert_no_panic 2 "--help" "$help_out"
[[ -n "$help_out" ]] || fail_step 2 "empty help (--help and -h printed nothing)"
pass_step 2 "Help"

ver_out=$(timeout "$TIMEOUT_SEC" "$BIN_PATH" --version 2>&1 || timeout "$TIMEOUT_SEC" "$BIN_PATH" -v 2>&1 || true)
assert_no_panic 3 "--version" "$ver_out"
pass_step 3 "Version"

bogus=$(timeout "$TIMEOUT_SEC" "$BIN_PATH" --this-flag-does-not-exist 2>&1)
bogus_exit=$?
(( bogus_exit < 124 )) || fail_step 4 "an unknown flag hung or crashed the binary (exit $bogus_exit)"
assert_no_panic 4 "unknown-flag" "$bogus"
pass_step 4 "Invalid-flag" "exit $bogus_exit"

(( STEPS_OK == STEPS )) ||
    fail_step "$STEPS" "executed-step floor — $STEPS_OK of $STEPS steps were judged and passed"
echo
echo "=== $NAME: PASSED ==="
echo "  evidence: bin=$BIN_PATH bogus_exit=$bogus_exit"
