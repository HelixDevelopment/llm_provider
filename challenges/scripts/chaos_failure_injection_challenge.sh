#!/usr/bin/env bash
# chaos_failure_injection_challenge.sh — anti-bluff Chaos Challenge per
# CONST-035 + CONST-050(B); cascaded per CONST-051(A). Client-side fault
# injection against <PREFIX>_HEALTH_URL / <PREFIX>_CHAOS_HOST:<PREFIX>_CHAOS_PORT
# (<PREFIX> = C50_PREFIX in the module-identity block below).
#
# Six steps, each of which must pass (the executed-step floor: a step that was
# skipped or deleted is a finding, never a quiet pass):
#   [1/6] pre-chaos health answers 200, and the chaos port accepts a connection
#   [2/6] a salvo of malformed requests leaves health answering 200
#   [3/6] an oversized (8 KiB) header gets 200/400/413/414/431/494, not a hang
#         or a dropped connection
#   [4/6] five slow-loris connections are opened
#   [5/6] CHAOS_LEGIT_REQUESTS legitimate requests under that pressure reach
#         CHAOS_LEGIT_MIN_PASS_PCT % 200s (every request must be recorded)
#   [6/6] post-chaos health answers 200
#
# Usage:
#   bash chaos_failure_injection_challenge.sh
#   bash chaos_failure_injection_challenge.sh --prove-failure
#
# Exit (three-valued; a finding outranks a could-not-determine; 2 is NEVER a pass):
#   0 = the target was reached and every chaos step held
#   1 = a real finding — the target degraded under injected faults, or a step
#       was skipped (executed-step floor)
#   2 = COULD NOT DETERMINE — an unknown argument, no target configured, the
#       target not answering 200 before injection, the chaos port refusing
#       connections (no fault could be injected), a knob that is not a valid
#       number, fewer responses recorded than requests sent, or no curl/timeout.
#       Nothing (more) was judged; the target is NOT known to be resilient.
#   The pass threshold CHAOS_LEGIT_MIN_PASS_PCT has a 80 % floor: a value below it is also
#   COULD NOT DETERMINE unless CHAOS_LEGIT_MIN_PASS_PCT_BELOW_FLOOR=acknowledged (that exact
#   value; any other is refused too); then the run still runs and its
#   evidence line starts with PERMISSIVE THRESHOLD.
#
# --prove-failure (§1.1 paired proof, VSC-369; harness lib/const050b_proof.sh):
# runs copies of THIS file against loopback stub targets. A healthy stub must
# pass (0); a stub that breaks after malformed input, one that drops an
# oversized header, one that fails under the mixed load and one that dies
# right after it must each exit 1 naming their step; no target, an unknown
# argument, a refused health URL, a refused chaos port and a malformed knob or a 0 % threshold
# must exit 2; a deleted step (planted in a copy) trips the floor (1) and a
# truncated load (planted in a copy) trips the request floor (2).
# Floor cases F1-F7: one below the floor refused, exactly the floor runs and
# is not permissive, an acknowledged 50 % runs a 60 % target and says
# PERMISSIVE THRESHOLD while the default threshold calls the same target a
# finding, an override of 'yes' or 'acknowledge' is refused, 0 % stays refused
# when acknowledged.

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
    pf_begin "$SELF" "chaos challenge" curl python3 bash || return $?
    local U="${C50_PREFIX}_HEALTH_URL" H="${C50_PREFIX}_CHAOS_HOST" P="${C50_PREFIX}_CHAOS_PORT"
    local g_from c url url_PORT closed knobs=(CHAOS_LEGIT_REQUESTS=20)
    # C1: clean control
    if c="$(pf_copy c1)" && pf_stub url ok; then
        pf_expect "C1 healthy stub target passes all 6 steps" 0 "Chaos Challenge: PASSED" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}"
    else pf_undet "C1" "could not start the stub target"; fi
    # M1..M4: one broken stub target per case
    if c="$(pf_copy m1)" && pf_stub url break-on-malformed; then
        pf_expect "M1 target that breaks after malformed input" 1 "[2/6] FAIL" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}"
    else pf_undet "M1" "could not start the stub target"; fi
    if c="$(pf_copy m2)" && pf_stub url reject-oversize; then
        pf_expect "M2 target that drops an oversized header" 1 "[3/6] FAIL" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}"
    else pf_undet "M2" "could not start the stub target"; fi
    # counted GETs before the mixed load: pre-check, "GET //" of the salvo,
    # post-salvo check, oversized request = 4
    if c="$(pf_copy m3)" && pf_stub url fail-after 8; then
        pf_expect "M3 target failing under the mixed load" 1 "[5/6] FAIL" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}"
    else pf_undet "M3" "could not start the stub target"; fi
    if c="$(pf_copy m4)" && pf_stub url fail-after 24; then
        pf_expect "M4 target dead right after the chaos" 1 "[6/6] FAIL" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}"
    else pf_undet "M4" "could not start the stub target"; fi
    # G1/G2: defects planted in the copy of this file
    g_from='pass_step '; g_from+='6 "Post-chaos"'
    if c="$(pf_copy g1)" && pf_plant "$c" "$g_from" ':' && pf_stub url ok; then
        pf_expect "G1 a deleted step trips the executed-step floor" 1 "executed-step floor" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}"
    else pf_undet "G1" "could not plant the defect"; fi
    g_from='seq '; g_from+='1 "$LEGIT_REQS" |'
    if c="$(pf_copy g2)" && pf_plant "$c" "$g_from" 'seq 2 "$LEGIT_REQS" |' && pf_stub url ok; then
        pf_expect "G2 a truncated load trips the executed-request floor" 2 "executed-request floor" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}"
    else pf_undet "G2" "could not plant the defect"; fi
    # U1..U6: could not determine
    if c="$(pf_copy u1)"; then
        pf_expect "U1 no target configured" 2 "#env-no-target" pf_run "$c"
    else pf_undet "U1" "could not copy"; fi
    if c="$(pf_copy u2)" && pf_closed_port closed; then
        pf_expect "U2 unreachable target (refused loopback port)" 2 "#env-target-down" \
            pf_run "$c" "$U=http://127.0.0.1:$closed/health" "$H=127.0.0.1" "$P=$closed" "${knobs[@]}"
    else pf_undet "U2" "could not find a refused port"; fi
    if c="$(pf_copy u3)" && pf_stub url ok && pf_closed_port closed; then
        pf_expect "U3 healthy target but a refused chaos port: nothing injected" 2 "#env-chaos-port-closed" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$closed" "${knobs[@]}"
    else pf_undet "U3" "could not start the stub target"; fi
    if c="$(pf_copy u4)" && pf_stub url ok; then
        pf_expect "U4 a knob that is not a number" 2 "#bad-knob" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" CHAOS_LEGIT_REQUESTS=lots
    else pf_undet "U4" "could not start the stub target"; fi
    if c="$(pf_copy u0pct)" && pf_stub url ok; then
        pf_expect "U6 a 0 % pass threshold (every request may fail) is refused" 2 "#bad-knob" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}" CHAOS_LEGIT_MIN_PASS_PCT=0
    else pf_undet "U6" "could not start the stub target"; fi
    if c="$(pf_copy u5)"; then
        pf_expect "U7 unknown argument" 2 "unknown argument" pf_run "$c" -- --no-such-flag
    else pf_undet "U7" "could not copy"; fi
    # F1..F7: the 80 % legitimate-traffic floor and its one acknowledged override
    local K=CHAOS_LEGIT_MIN_PASS_PCT
    if c="$(pf_copy f1)" && pf_stub url ok; then
        pf_expect "F1 a threshold one below the 80 % floor (79 %) is refused" 2 "below the 80 % floor" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}" "$K=79"
    else pf_undet "F1" "could not start the stub target"; fi
    if c="$(pf_copy f2)" && pf_stub url ok; then
        pf_expect_without "F2 a threshold exactly at the floor (80 %) runs and is not permissive" 0 "Chaos Challenge: PASSED" \
            "PERMISSIVE THRESHOLD" pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}" "$K=80"
    else pf_undet "F2" "could not start the stub target"; fi
    # counted GETs: 4 before the mixed load (see M3), legit 5..24, final 25;
    # 503 for 5..12 = 12/20 (60 %)
    if c="$(pf_copy f3)" && pf_stub url fail-window 4:12; then
        pf_expect "F3 an acknowledged 50 % threshold runs a 60 % target and says PERMISSIVE THRESHOLD" 0 \
            "evidence: PERMISSIVE THRESHOLD" pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}" \
            "$K=50" "${K}_BELOW_FLOOR=acknowledged"
    else pf_undet "F3" "could not start the stub target"; fi
    if c="$(pf_copy f4)" && pf_stub url fail-window 4:12; then
        pf_expect "F4 the same 60 % target at the default threshold is a finding" 1 "[5/6] FAIL" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}"
    else pf_undet "F4" "could not start the stub target"; fi
    if c="$(pf_copy f5)" && pf_stub url ok; then
        pf_expect "F5 an override value other than 'acknowledged' (yes) is refused" 2 "only accepted value is 'acknowledged'" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}" "$K=50" "${K}_BELOW_FLOOR=yes"
    else pf_undet "F5" "could not start the stub target"; fi
    if c="$(pf_copy f6)" && pf_stub url ok; then
        pf_expect "F6 a misspelt override (acknowledge) is refused" 2 "only accepted value is 'acknowledged'" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}" "$K=50" "${K}_BELOW_FLOOR=acknowledge"
    else pf_undet "F6" "could not start the stub target"; fi
    if c="$(pf_copy f7)" && pf_stub url ok; then
        pf_expect "F7 a 0 % threshold stays refused even when acknowledged" 2 "is not a percentage in 1..100" \
            pf_run "$c" "$U=$url/health" "$H=127.0.0.1" "$P=$url_PORT" "${knobs[@]}" "$K=0" "${K}_BELOW_FLOOR=acknowledged"
    else pf_undet "F7" "could not start the stub target"; fi
    pf_end
}

if [[ "$PROVE" -eq 1 ]]; then
    prove_failure
    exit $?
fi

# ── live challenge body ──────────────────────────────────────────────────────
URL_VAR="${C50_PREFIX}_HEALTH_URL"
HOST_VAR="${C50_PREFIX}_CHAOS_HOST"
PORT_VAR="${C50_PREFIX}_CHAOS_PORT"
HEALTH_URL="${!URL_VAR:-}"
CHAOS_HOST="${!HOST_VAR:-localhost}"
CHAOS_PORT="${!PORT_VAR:-}"
LEGIT_REQS="${CHAOS_LEGIT_REQUESTS:-100}"
MIN_PCT="${CHAOS_LEGIT_MIN_PASS_PCT:-95}"
STEPS=6
STEPS_OK=0
NAME="$C50_MODULE Chaos Challenge"

pass_step() { STEPS_OK=$((STEPS_OK + 1)); echo "[$1/$STEPS] $2: PASS${3:+ ($3)}"; }
fail_step() { echo "[$1/$STEPS] FAIL: $2"; echo "=== $NAME: FAILED ==="; exit 1; }
undetermined() {
    echo "COULD NOT DETERMINE: $1"
    echo "  Nothing (more) was judged. $NAME is NOT known to be resilient."
    echo "=== $NAME: COULD NOT DETERMINE ($2) ==="
    exit 2
}
# Pass-rate floor (operator decision 2026-10-15, VSC-369 follow-up): a
# threshold below the floor is COULD NOT DETERMINE (#bad-knob) unless
# <KNOB>_BELOW_FLOOR=acknowledged — that exact value, nothing else — and then
# the run still runs and its evidence line starts with PERMISSIVE THRESHOLD.
# check_floor KNOB VALUE FLOOR (VALUE already validated as 1..100)
PERMISSIVE=""
check_floor() {
    local knob="$1" val="$2" floor="$3" ackvar="${1}_BELOW_FLOOR" ack
    ack="${!ackvar:-}"
    [[ -z "$ack" || "$ack" == acknowledged ]] ||
        undetermined "$ackvar='$ack' — the only accepted value is 'acknowledged'" "#bad-knob"
    (( val >= floor )) && return 0
    [[ "$ack" == acknowledged ]] ||
        undetermined "$knob=$val is below the $floor % floor — set $ackvar=acknowledged to run with it anyway" "#bad-knob"
    PERMISSIVE="PERMISSIVE THRESHOLD ($knob=$val below the $floor % floor, $ackvar=acknowledged) "
    echo "  WARNING: $knob=$val is below the $floor % floor; running because $ackvar=acknowledged"
}

echo "=== $C50_MODULE Chaos Failure-Injection Challenge ==="
echo "  url=$HEALTH_URL host=${CHAOS_HOST}:${CHAOS_PORT}"

for tool in curl timeout; do
    command -v "$tool" >/dev/null 2>&1 || undetermined "'$tool' is not on PATH" "#env-no-tool"
done
[[ -n "$HEALTH_URL" && -n "$CHAOS_PORT" ]] ||
    undetermined "$URL_VAR / $PORT_VAR unset — there is no target to injure" "#env-no-target"
[[ "$CHAOS_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( CHAOS_PORT <= 65535 )) ||
    undetermined "$PORT_VAR='$CHAOS_PORT' is not a TCP port" "#bad-knob"
[[ "$LEGIT_REQS" =~ ^[1-9][0-9]*$ ]] ||
    undetermined "CHAOS_LEGIT_REQUESTS='$LEGIT_REQS' is not a positive integer" "#bad-knob"
[[ "$MIN_PCT" =~ ^[1-9][0-9]*$ ]] && (( MIN_PCT <= 100 )) ||
    undetermined "CHAOS_LEGIT_MIN_PASS_PCT='$MIN_PCT' is not a percentage in 1..100" "#bad-knob"
check_floor CHAOS_LEGIT_MIN_PASS_PCT "$MIN_PCT" 80

pre=$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || pre="000"
[[ "$pre" == "200" ]] ||
    undetermined "target did not answer 200 before injection (HTTP $pre)" "#env-target-down"
# A refused chaos port used to inject NOTHING and then pass every step.
timeout 3 bash -c ": <>/dev/tcp/$CHAOS_HOST/$CHAOS_PORT" 2>/dev/null ||
    undetermined "the chaos endpoint $CHAOS_HOST:$CHAOS_PORT refused a connection — no fault can be injected" "#env-chaos-port-closed"
pass_step 1 "Pre-chaos"

for case in "BADVERB / HTTP/1.1" "GET / HTTP/9.9" "GET // HTTP/1.1" "INVALID"; do
    printf '%s\r\nHost: %s\r\n\r\n' "$case" "$CHAOS_HOST" 2>/dev/null \
        | timeout 3 bash -c "exec 3<>/dev/tcp/$CHAOS_HOST/$CHAOS_PORT && cat >&3 && cat <&3" >/dev/null 2>&1 || true
done
post=$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || post="000"
[[ "$post" == "200" ]] || fail_step 2 "health answered HTTP $post after the malformed salvo"
pass_step 2 "Malformed-salvo"

# exactly 8192 bytes (the old `tr -dc` over /dev/urandom kept only ~2 KiB)
huge=$(printf '%8192s' '' | tr ' ' 'A')
oversize=$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" -H "X-Chaos-Huge: $huge" "$HEALTH_URL" 2>/dev/null) || oversize="000"
case "$oversize" in
    200|400|413|414|431|494) pass_step 3 "Oversized" "HTTP $oversize" ;;
    *) fail_step 3 "an 8 KiB header got HTTP $oversize (want 200/400/413/414/431/494)" ;;
esac

pids=()
RES=$(mktemp) || undetermined "mktemp failed (TMPDIR='${TMPDIR:-}')" "#env-no-scratch"
cleanup() { rm -f "$RES"; [[ ${#pids[@]} -gt 0 ]] && kill "${pids[@]}" 2>/dev/null; }
trap cleanup EXIT
for _ in 1 2 3 4 5; do
    (timeout 5 bash -c "
        exec 3<>/dev/tcp/$CHAOS_HOST/$CHAOS_PORT 2>/dev/null
        printf 'GET /health HTTP/1.1\r\nHost: $CHAOS_HOST\r\nX-Slow: ' >&3 2>/dev/null
        sleep 4 2>/dev/null
        printf 'done\r\n\r\n' >&3 2>/dev/null
    " >/dev/null 2>&1 || true) &
    pids+=($!)
done
pass_step 4 "Slow-loris" "${#pids[@]} spawned"

seq 1 "$LEGIT_REQS" | xargs -n1 -P 20 -I{} \
    curl -sS -o /dev/null --max-time 5 -w "%{http_code}\n" "$HEALTH_URL" 2>/dev/null >> "$RES" || true
for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
wait 2>/dev/null || true
total=$(grep -c . "$RES")
# Executed-request floor: every request must have left a status line (curl
# writes 000 for a refused or timed-out one), else the load did not run.
(( total == LEGIT_REQS )) ||
    undetermined "executed-request floor: $total of $LEGIT_REQS requests recorded — the mixed load did not run" "#load-not-run"
ok=$(awk '$1=="200"{c++} END{print c+0}' "$RES")
pct=$((ok * 100 / total))
(( pct >= MIN_PCT )) || fail_step 5 "mixed load $ok/$total = ${pct}% < ${MIN_PCT}%"
pass_step 5 "Mixed-load" "$ok/$total ${pct}%"

final=$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || final="000"
[[ "$final" == "200" ]] || fail_step 6 "health answered HTTP $final after the chaos"
pass_step 6 "Post-chaos"

(( STEPS_OK == STEPS )) ||
    fail_step "$STEPS" "executed-step floor — $STEPS_OK of $STEPS steps were judged and passed"
echo
echo "=== $NAME: PASSED ==="
echo "  evidence: ${PERMISSIVE}legit=$total pct=${pct}% slow_loris=${#pids[@]} oversize=${oversize}"
