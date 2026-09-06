#!/usr/bin/env bash
# ux_end_to_end_flow_challenge.sh — anti-bluff UX Challenge for
# LLMProvider per CONST-035 + CONST-050(B). Cascade per CONST-051(A).
#
# ── EXIT CODES — three-valued, and 2 is NEVER a pass ─────────────────────────
#   0  the challenge ran against a real binary and every assertion passed
#   1  a real finding: the binary was there to be tested and an assertion failed
#   2  COULD NOT DETERMINE — $LLMPROVIDER_VD_BIN is unset, or does not
#      point at an executable file.
#      NOTHING was measured, so the subject is NOT known to be healthy.
#
# Why the third value exists. Every branch below that could not reach its
# binary used to print "PASSED (SKIP-OK)" and exit 0. That made a DNS blip, a
# restarting binary, an unexported variable and a typo'd URL
# indistinguishable — BY EXIT CODE — from a genuine pass, so any caller that
# reads the exit status (the only thing most callers read) was told the subject
# had been tested and was fine when in fact nothing had been tested at all.
# "Could not test" must never read as "tested and fine".

set -uo pipefail
BIN_PATH="${LLMPROVIDER_VD_BIN:-}"
TIMEOUT_SEC="${UX_TIMEOUT_SEC:-30}"
USER_HOSTILE=('panic:' 'goroutine [0-9]+ \[running\]:' 'runtime error:' 'segmentation fault' 'fatal error:')

echo "=== LLMProvider UX End-to-End Flow Challenge ==="
echo "  bin=$BIN_PATH timeout=${TIMEOUT_SEC}s"

if [[ -z "$BIN_PATH" ]] || [[ ! -x "$BIN_PATH" ]]; then
    echo "[1/5] COULD NOT DETERMINE: LLMPROVIDER_VD_BIN is unset or not executable (bin='$BIN_PATH')."
    echo "  Nothing was measured. LLMProvider UX Challenge is NOT known to be healthy."
    echo "=== LLMProvider UX Challenge: COULD NOT DETERMINE (#env-binary-missing) ==="
    exit 2
fi
echo "[1/5] Binary present: PASS"

assert_no_panic() {
    local label="$1" body="$2"
    for pat in "${USER_HOSTILE[@]}"; do
        printf '%s' "$body" | grep -qE "$pat" && { echo "  FAIL: $label leaked: $pat"; return 1; }
    done
    # An explicit success. Without it the function returned the status of the
    # LAST grep in the loop, which is 1 exactly when no hostile pattern was
    # found -- so `assert_no_panic ... || exit 1` fired on every CLEAN binary
    # and this challenge could not pass for any well-behaved subject.
    # Measured against the unmodified script with a fake binary that prints a
    # normal --help: rc=1.
    return 0
}

help_out=$(timeout "$TIMEOUT_SEC" "$BIN_PATH" --help 2>&1 || timeout "$TIMEOUT_SEC" "$BIN_PATH" -h 2>&1 || true)
assert_no_panic "--help" "$help_out" || exit 1
[[ -z "$help_out" ]] && { echo "[2/5] FAIL: empty help"; exit 1; }
echo "[2/5] Help discovery: PASS"

ver_out=$(timeout "$TIMEOUT_SEC" "$BIN_PATH" --version 2>&1 || timeout "$TIMEOUT_SEC" "$BIN_PATH" -v 2>&1 || true)
assert_no_panic "--version" "$ver_out" || exit 1
echo "[3/5] Version surface: PASS"

set +e
bogus_out=$(timeout "$TIMEOUT_SEC" "$BIN_PATH" --does-not-exist-flag 2>&1)
bogus_exit=$?
set -e
assert_no_panic "bogus" "$bogus_out" || exit 1
[[ "$bogus_exit" -ge 124 ]] && { echo "[4/5] FAIL: crashed"; exit 1; }
echo "[4/5] Graceful recovery: PASS (exit $bogus_exit)"

post=$(timeout "$TIMEOUT_SEC" "$BIN_PATH" --help 2>&1 || timeout "$TIMEOUT_SEC" "$BIN_PATH" -h 2>&1 || true)
assert_no_panic "post-error --help" "$post" || exit 1
[[ -z "$post" ]] && { echo "[5/5] FAIL"; exit 1; }
echo "[5/5] Post-error liveness: PASS"

echo
echo "=== LLMProvider UX Challenge: PASSED ==="
echo "  evidence: journey=discover→help→version→recover→post-liveness bogus_exit=$bogus_exit"
