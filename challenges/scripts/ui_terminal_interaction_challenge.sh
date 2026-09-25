#!/usr/bin/env bash
# ui_terminal_interaction_challenge.sh — anti-bluff UI Challenge for
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
TIMEOUT_SEC="${UI_TIMEOUT_SEC:-30}"
USER_HOSTILE=('panic:' 'goroutine [0-9]+ \[running\]:' 'runtime error:' 'segmentation fault' 'fatal error:')

echo "=== LLMProvider UI Terminal-Interaction Challenge ==="
echo "  bin=$BIN_PATH timeout=${TIMEOUT_SEC}s"

if [[ -z "$BIN_PATH" ]] || [[ ! -x "$BIN_PATH" ]]; then
    echo "[1/4] COULD NOT DETERMINE: LLMPROVIDER_VD_BIN is unset or not executable (bin='$BIN_PATH')."
    echo "  Nothing was measured. LLMProvider UI Challenge is NOT known to be healthy."
    echo "=== LLMProvider UI Challenge: COULD NOT DETERMINE (#env-binary-missing) ==="
    exit 2
fi
echo "[1/4] Binary present: PASS"

assert_no_panic() {
    local label="$1" body="$2"
    for pat in "${USER_HOSTILE[@]}"; do
        # Bash's own regex operator, NOT `printf ... | grep -qE`. Under the
        # `set -o pipefail` above, grep -q closing the pipe on a MATCH kills
        # printf with SIGPIPE (141) and pipefail promotes it, so the `&&` did
        # not fire — this check FAILED OPEN on a body over ~4 KiB, missing the
        # leak precisely because it was there.
        [[ $body =~ $pat ]] && { echo "  FAIL: $label leaked: $pat"; return 1; }
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
[[ -z "$help_out" ]] && { echo "[2/4] FAIL: empty help"; exit 1; }
echo "[2/4] Help: PASS"

ver_out=$(timeout "$TIMEOUT_SEC" "$BIN_PATH" --version 2>&1 || timeout "$TIMEOUT_SEC" "$BIN_PATH" -v 2>&1 || true)
assert_no_panic "--version" "$ver_out" || exit 1
echo "[3/4] Version: PASS"

set +e
bogus=$(timeout "$TIMEOUT_SEC" "$BIN_PATH" --this-flag-does-not-exist 2>&1)
bogus_exit=$?
set -e
[[ "$bogus_exit" -ge 124 ]] && { echo "[4/4] FAIL: crashed"; exit 1; }
assert_no_panic "bogus" "$bogus" || exit 1
echo "[4/4] Invalid-flag: PASS (exit $bogus_exit)"

echo
echo "=== LLMProvider UI Challenge: PASSED ==="
echo "  evidence: bin=$BIN_PATH bogus_exit=$bogus_exit"
