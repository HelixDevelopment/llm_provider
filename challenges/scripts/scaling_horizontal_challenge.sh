#!/usr/bin/env bash
# scaling_horizontal_challenge.sh — anti-bluff Scaling Challenge per CONST-035
# + CONST-050(B); cascaded per CONST-051(A). Drives the replica set listed in
# <PREFIX>_SCALING_REPLICA_URLS (comma-separated base URLs, each serving
# /health; <PREFIX> = C50_PREFIX in the module-identity block below).
#
# Six steps, each of which must pass (the executed-step floor: a step that was
# skipped or deleted is a finding, never a quiet pass):
#   [1/6] at least two DISTINCT listed replicas answer 200 (a replica listed
#         twice is one replica), and every listed replica answers: one that
#         does not is DOWN, named, and a finding — unless SCALING_ALLOW_DOWN=N
#         declares up to N down replicas (N must be below the listed count)
#   [2/6] every reached replica's health body carries "status": ok|healthy|UP
#   [3/6] SCALING_REQS_PER_REPLICA requests per replica reach
#         SCALING_MIN_PASS_PCT % 200s (every request must be recorded)
#   [4/6] every replica serves the same health body (uptime/timestamp removed)
#   [5/6] load-balancer fairness, informational: total 200s across replicas
#   [6/6] every replica still answers 200
#
# Usage:
#   bash scaling_horizontal_challenge.sh
#   bash scaling_horizontal_challenge.sh --prove-failure
#
# Exit (three-valued; a finding outranks a could-not-determine; 2 is NEVER a pass):
#   0 = >= 2 distinct replicas were reached and every scaling step held
#   1 = a real finding — more listed replicas down than SCALING_ALLOW_DOWN
#       (default 0) declares, a replica reported a wrong status, degraded,
#       diverged or died, or a step was skipped (executed-step floor)
#   2 = COULD NOT DETERMINE — an unknown argument, no replica list, fewer than
#       two distinct replicas answering 200, a knob that is not a valid number,
#       SCALING_MIN_PASS_PCT below the 90 % floor without
#       SCALING_MIN_PASS_PCT_BELOW_FLOOR=acknowledged (any other value of that
#       override is refused too), SCALING_ALLOW_DOWN not below the listed
#       count, fewer responses recorded than requests sent, or no curl.
#       Horizontal scaling was NOT exercised; nothing (more) was judged.
# With the acknowledged override the run still runs and its evidence line
# starts with PERMISSIVE THRESHOLD.
#
# --prove-failure (§1.1 paired proof, VSC-369; harness lib/const050b_proof.sh):
# runs copies of THIS file against loopback stub replicas. Two healthy stubs
# must pass (0); a replica reporting a wrong status, one failing under load,
# two replicas serving different bodies and a replica dying right after the
# load must each exit 1 naming their step; no list, one healthy replica plus a
# refused port, the same replica listed twice, a malformed knob or a 0 % threshold and an unknown
# argument must exit 2; a deleted step (planted in a copy) trips the floor (1)
# and a truncated load (planted in a copy) trips the request floor (2).
# Floor cases F1-F7: 89 % refused, 90 % runs and is not permissive, an
# acknowledged 50 % runs a 60 % replica and says PERMISSIVE THRESHOLD while the
# default threshold calls the same replica a finding, an override of 'yes' or
# 'acknowledge' is refused, 0 % stays refused when acknowledged. Down-replica
# cases D1-D5: one of three down is a finding, SCALING_ALLOW_DOWN=1 tolerates
# it, SCALING_ALLOW_DOWN=3 of 3 listed is refused, two down with 1 allowed is a
# finding, a non-numeric SCALING_ALLOW_DOWN is refused.

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
    pf_begin "$SELF" "scaling challenge" curl python3 sha256sum || return $?
    local R="${C50_PREFIX}_SCALING_REPLICA_URLS"
    # shellcheck disable=SC2034  # *_PORT is written by pf_stub (printf -v)
    local g_from c a a_PORT b b_PORT closed knobs=(SCALING_REQS_PER_REPLICA=20 SCALING_CONCURRENCY=5)
    if c="$(pf_copy c1)" && pf_stub a ok && pf_stub b ok; then
        pf_expect "C1 two healthy stub replicas pass all 6 steps" 0 "Scaling Challenge: PASSED" pf_run "$c" "$R=$a,$b" "${knobs[@]}"
    else pf_undet "C1" "could not start the stub replicas"; fi
    if c="$(pf_copy m1)" && pf_stub a ok && pf_stub b bad-schema; then
        pf_expect "M1 a replica reporting status \"down\"" 1 "[2/6] FAIL" pf_run "$c" "$R=$a,$b" "${knobs[@]}"
    else pf_undet "M1" "could not start the stub replicas"; fi
    # counted GETs on a replica before its load: topology probe, schema probe = 2
    if c="$(pf_copy m2)" && pf_stub a ok && pf_stub b fail-after 5; then
        pf_expect "M2 a replica failing under load" 1 "[3/6] FAIL" pf_run "$c" "$R=$a,$b" "${knobs[@]}"
    else pf_undet "M2" "could not start the stub replicas"; fi
    if c="$(pf_copy m3)" && pf_stub a variant one && pf_stub b variant two; then
        pf_expect "M3 replicas serving different bodies" 1 "[4/6] FAIL" pf_run "$c" "$R=$a,$b" "${knobs[@]}"
    else pf_undet "M3" "could not start the stub replicas"; fi
    # topology 1 + schema 1 + load 20 + identity 1 = 23 counted GETs
    if c="$(pf_copy m4)" && pf_stub a ok && pf_stub b fail-after 23; then
        pf_expect "M4 a replica dead right after the load" 1 "[6/6] FAIL" pf_run "$c" "$R=$a,$b" "${knobs[@]}"
    else pf_undet "M4" "could not start the stub replicas"; fi
    g_from='pass_step '; g_from+='6 "Post-scaling liveness"'
    if c="$(pf_copy g1)" && pf_plant "$c" "$g_from" ':' && pf_stub a ok && pf_stub b ok; then
        pf_expect "G1 a deleted step trips the executed-step floor" 1 "executed-step floor" pf_run "$c" "$R=$a,$b" "${knobs[@]}"
    else pf_undet "G1" "could not plant the defect"; fi
    g_from='seq '; g_from+='1 "$REQS" |'
    if c="$(pf_copy g2)" && pf_plant "$c" "$g_from" 'seq 2 "$REQS" |' && pf_stub a ok && pf_stub b ok; then
        pf_expect "G2 a truncated load trips the executed-request floor" 2 "executed-request floor" pf_run "$c" "$R=$a,$b" "${knobs[@]}"
    else pf_undet "G2" "could not plant the defect"; fi
    if c="$(pf_copy u1)"; then
        pf_expect "U1 no replica list configured" 2 "#env-no-replicas" pf_run "$c"
    else pf_undet "U1" "could not copy"; fi
    if c="$(pf_copy u2)" && pf_stub a ok && pf_closed_port closed; then
        pf_expect "U2 one healthy replica plus an unreachable one" 2 "#env-single-replica" \
            pf_run "$c" "$R=$a,http://127.0.0.1:$closed" "${knobs[@]}"
    else pf_undet "U2" "could not start the stub replica"; fi
    if c="$(pf_copy u3)" && pf_stub a ok; then
        pf_expect "U3 the same replica listed twice is one replica" 2 "#env-single-replica" pf_run "$c" "$R=$a,$a" "${knobs[@]}"
    else pf_undet "U3" "could not start the stub replica"; fi
    if c="$(pf_copy u4)" && pf_stub a ok && pf_stub b ok; then
        pf_expect "U4 a knob that is not a number" 2 "#bad-knob" pf_run "$c" "$R=$a,$b" SCALING_REQS_PER_REPLICA=x
    else pf_undet "U4" "could not start the stub replicas"; fi
    if c="$(pf_copy u0pct)" && pf_stub a ok && pf_stub b ok; then
        pf_expect "U6 a 0 % pass threshold (every request may fail) is refused" 2 "#bad-knob" \
            pf_run "$c" "$R=$a,$b" "${knobs[@]}" SCALING_MIN_PASS_PCT=0
    else pf_undet "U6" "could not start the stub target"; fi
    if c="$(pf_copy u5)"; then
        pf_expect "U7 unknown argument" 2 "unknown argument" pf_run "$c" -- --no-such-flag
    else pf_undet "U7" "could not copy"; fi
    # F1..F7: the 90 % pass-rate floor and its one acknowledged override
    local K=SCALING_MIN_PASS_PCT
    if c="$(pf_copy f1)" && pf_stub a ok && pf_stub b ok; then
        pf_expect "F1 a threshold one below the 90 % floor (89 %) is refused" 2 "below the 90 % floor" \
            pf_run "$c" "$R=$a,$b" "${knobs[@]}" "$K=89"
    else pf_undet "F1" "could not start the stub replicas"; fi
    if c="$(pf_copy f2)" && pf_stub a ok && pf_stub b ok; then
        pf_expect_without "F2 a threshold exactly at the floor (90 %) runs and is not permissive" 0 "Scaling Challenge: PASSED" \
            "PERMISSIVE THRESHOLD" pf_run "$c" "$R=$a,$b" "${knobs[@]}" "$K=90"
    else pf_undet "F2" "could not start the stub replicas"; fi
    # counted GETs on replica b: topology 1, schema 2, load 3..22, identity 23,
    # post 24; 503 for 3..10 = 12/20 (60 %)
    if c="$(pf_copy f3)" && pf_stub a ok && pf_stub b fail-window 2:10; then
        pf_expect "F3 an acknowledged 50 % threshold runs a 60 % replica and says PERMISSIVE THRESHOLD" 0 \
            "evidence: PERMISSIVE THRESHOLD" pf_run "$c" "$R=$a,$b" "${knobs[@]}" "$K=50" "${K}_BELOW_FLOOR=acknowledged"
    else pf_undet "F3" "could not start the stub replicas"; fi
    if c="$(pf_copy f4)" && pf_stub a ok && pf_stub b fail-window 2:10; then
        pf_expect "F4 the same 60 % replica at the default threshold is a finding" 1 "[3/6] FAIL" \
            pf_run "$c" "$R=$a,$b" "${knobs[@]}"
    else pf_undet "F4" "could not start the stub replicas"; fi
    if c="$(pf_copy f5)" && pf_stub a ok && pf_stub b ok; then
        pf_expect "F5 an override value other than 'acknowledged' (yes) is refused" 2 "only accepted value is 'acknowledged'" \
            pf_run "$c" "$R=$a,$b" "${knobs[@]}" "$K=50" "${K}_BELOW_FLOOR=yes"
    else pf_undet "F5" "could not start the stub replicas"; fi
    if c="$(pf_copy f6)" && pf_stub a ok && pf_stub b ok; then
        pf_expect "F6 a misspelt override (acknowledge) is refused" 2 "only accepted value is 'acknowledged'" \
            pf_run "$c" "$R=$a,$b" "${knobs[@]}" "$K=50" "${K}_BELOW_FLOOR=acknowledge"
    else pf_undet "F6" "could not start the stub replicas"; fi
    if c="$(pf_copy f7)" && pf_stub a ok && pf_stub b ok; then
        pf_expect "F7 a 0 % threshold stays refused even when acknowledged" 2 "is not a percentage in 1..100" \
            pf_run "$c" "$R=$a,$b" "${knobs[@]}" "$K=0" "${K}_BELOW_FLOOR=acknowledged"
    else pf_undet "F7" "could not start the stub replicas"; fi
    # D1..D5: a listed replica that is down, and the one declared tolerance
    # shellcheck disable=SC2034  # *_PORT is written by pf_stub (printf -v)
    local d d_PORT
    if c="$(pf_copy d1)" && pf_stub a ok && pf_stub b ok && pf_closed_port closed; then
        pf_expect "D1 a listed replica that is down is a finding" 1 "[1/6] FAIL" \
            pf_run "$c" "$R=$a,$b,http://127.0.0.1:$closed" "${knobs[@]}"
    else pf_undet "D1" "could not start the stub replicas"; fi
    if c="$(pf_copy d2)" && pf_stub a ok && pf_stub b ok && pf_closed_port closed; then
        pf_expect "D2 SCALING_ALLOW_DOWN=1 tolerates one down replica of three" 0 "Scaling Challenge: PASSED" \
            pf_run "$c" "$R=$a,$b,http://127.0.0.1:$closed" "${knobs[@]}" SCALING_ALLOW_DOWN=1
    else pf_undet "D2" "could not start the stub replicas"; fi
    if c="$(pf_copy d3)" && pf_stub a ok && pf_stub b ok && pf_closed_port closed; then
        pf_expect "D3 SCALING_ALLOW_DOWN equal to the listed count (3 of 3) is refused" 2 "is not below the 3 listed" \
            pf_run "$c" "$R=$a,$b,http://127.0.0.1:$closed" "${knobs[@]}" SCALING_ALLOW_DOWN=3
    else pf_undet "D3" "could not start the stub replicas"; fi
    # two down replicas: a refused port and a stub answering 503 to everything
    if c="$(pf_copy d4)" && pf_stub a ok && pf_stub b ok && pf_stub d fail-after 0 && pf_closed_port closed; then
        pf_expect "D4 two down replicas with SCALING_ALLOW_DOWN=1 is a finding" 1 "[1/6] FAIL" \
            pf_run "$c" "$R=$a,$b,$d,http://127.0.0.1:$closed" "${knobs[@]}" SCALING_ALLOW_DOWN=1
    else pf_undet "D4" "could not start the stub replicas"; fi
    if c="$(pf_copy d5)" && pf_stub a ok && pf_stub b ok; then
        pf_expect "D5 SCALING_ALLOW_DOWN that is not a number is refused" 2 "#bad-knob" \
            pf_run "$c" "$R=$a,$b" "${knobs[@]}" SCALING_ALLOW_DOWN=one
    else pf_undet "D5" "could not start the stub replicas"; fi
    pf_end
}

if [[ "$PROVE" -eq 1 ]]; then
    prove_failure
    exit $?
fi

# ── live challenge body ──────────────────────────────────────────────────────
REPLICAS_VAR="${C50_PREFIX}_SCALING_REPLICA_URLS"
REPLICAS="${!REPLICAS_VAR:-}"
REQS="${SCALING_REQS_PER_REPLICA:-50}"
CONC="${SCALING_CONCURRENCY:-10}"
MIN_PCT="${SCALING_MIN_PASS_PCT:-95}"
ALLOW_DOWN="${SCALING_ALLOW_DOWN:-0}"
STEPS=6
STEPS_OK=0
NAME="$C50_MODULE Scaling Challenge"

pass_step() { STEPS_OK=$((STEPS_OK + 1)); echo "[$1/$STEPS] $2: PASS${3:+ ($3)}"; }
fail_step() { echo "[$1/$STEPS] FAIL: $2"; echo "=== $NAME: FAILED ==="; exit 1; }
undetermined() {
    echo "COULD NOT DETERMINE: $1"
    echo "  Nothing (more) was judged. $NAME is NOT known to be healthy."
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

echo "=== $C50_MODULE Scaling Challenge ==="
echo "  replicas=$REPLICAS reqs=$REQS conc=$CONC pass>=${MIN_PCT}%"

for tool in curl sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || undetermined "'$tool' is not on PATH" "#env-no-tool"
done
[[ -n "$REPLICAS" ]] || undetermined "$REPLICAS_VAR is unset — no replica set to test" "#env-no-replicas"
for knob in REQS CONC; do
    [[ "${!knob}" =~ ^[1-9][0-9]*$ ]] ||
        undetermined "$knob='${!knob}' (SCALING_*) is not a positive integer" "#bad-knob"
done
[[ "$MIN_PCT" =~ ^[1-9][0-9]*$ ]] && (( MIN_PCT <= 100 )) ||
    undetermined "SCALING_MIN_PASS_PCT='$MIN_PCT' is not a percentage in 1..100" "#bad-knob"
check_floor SCALING_MIN_PASS_PCT "$MIN_PCT" 90
[[ "$ALLOW_DOWN" =~ ^(0|[1-9][0-9]*)$ ]] ||
    undetermined "SCALING_ALLOW_DOWN='$ALLOW_DOWN' is not a non-negative integer" "#bad-knob"

IFS=',' read -r -a URLS <<< "$REPLICAS"
LISTED=()
declare -A SEEN=()
for u in "${URLS[@]}"; do
    u="${u// /}"; u="${u%/}"
    [[ -z "$u" ]] && continue
    if [[ -n "${SEEN[$u]:-}" ]]; then echo "  duplicate (counted once): $u"; continue; fi
    SEEN[$u]=1
    LISTED+=("$u")
done
[[ ${#LISTED[@]} -gt 0 ]] || undetermined "$REPLICAS_VAR lists no replica URL" "#env-no-replicas"
# A declared tolerance must leave at least one listed replica that has to answer.
(( ALLOW_DOWN < ${#LISTED[@]} )) ||
    undetermined "SCALING_ALLOW_DOWN=$ALLOW_DOWN is not below the ${#LISTED[@]} listed distinct replica(s)" "#bad-knob"
REACH=()
DOWN=()
for u in "${LISTED[@]}"; do
    c=$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$u/health" 2>/dev/null) || c="000"
    if [[ "$c" == "200" ]]; then REACH+=("$u"); echo "  reachable: $u"
    else DOWN+=("$u"); echo "  DOWN (HTTP $c): $u"; fi
done
[[ ${#REACH[@]} -ge 2 ]] ||
    undetermined "only ${#REACH[@]} distinct replica(s) of ${#LISTED[@]} listed answered 200 — need at least 2" "#env-single-replica"
# A listed replica that is down is a finding unless SCALING_ALLOW_DOWN declares it.
(( ${#DOWN[@]} <= ALLOW_DOWN )) ||
    fail_step 1 "${#DOWN[@]} of ${#LISTED[@]} listed replica(s) down (${DOWN[*]}); SCALING_ALLOW_DOWN=$ALLOW_DOWN"
pass_step 1 "Topology" "${#REACH[@]} replicas, ${#DOWN[@]} down (allowed $ALLOW_DOWN)"

# Bash's own regex, NOT `printf ... | grep -qE`: under `set -o pipefail`, grep
# -q closing the pipe on a match kills printf with SIGPIPE (141) and pipefail
# promotes it — the check would FAIL BECAUSE the status matched.
status_re='"status"[[:space:]]*:[[:space:]]*"(ok|healthy|UP)"'
for u in "${REACH[@]}"; do
    b=$(curl -sS --max-time 5 "$u/health" 2>/dev/null || true)
    [[ $b =~ $status_re ]] || fail_step 2 "$u health body carries no \"status\": ok|healthy|UP"
done
pass_step 2 "Schema sanity"

declare -A OK=()
for u in "${REACH[@]}"; do
    r=$(mktemp) || undetermined "mktemp failed (TMPDIR='${TMPDIR:-}')" "#env-no-scratch"
    seq 1 "$REQS" | xargs -n1 -P "$CONC" -I{} \
        curl -sS -o /dev/null --max-time 5 -w "%{http_code}\n" "$u/health" 2>/dev/null >> "$r" || true
    okc=$(awk '$1=="200"{c++} END{print c+0}' "$r")
    tot=$(grep -c . "$r")
    rm -f "$r"
    # Executed-request floor: every request must have left a status line.
    (( tot == REQS )) ||
        undetermined "executed-request floor: $tot of $REQS requests to $u recorded — the load did not run" "#load-not-run"
    pct=$((okc * 100 / tot))
    OK[$u]=$okc
    echo "  $u -> $okc/$tot ($pct%)"
    (( pct >= MIN_PCT )) || fail_step 3 "$u answered $okc/$tot = ${pct}% (< ${MIN_PCT}%)"
done
pass_step 3 "Per-replica load"

first=$(curl -sS --max-time 5 "${REACH[0]}/health" 2>/dev/null | sed 's/"uptime[^,}]*//g; s/"timestamp[^,}]*//g')
fh=$(printf '%s' "$first" | sha256sum | awk '{print $1}')
mm=0
for u in "${REACH[@]:1}"; do
    b=$(curl -sS --max-time 5 "$u/health" 2>/dev/null | sed 's/"uptime[^,}]*//g; s/"timestamp[^,}]*//g')
    h=$(printf '%s' "$b" | sha256sum | awk '{print $1}')
    if [[ "$h" == "$fh" ]]; then echo "  MATCH: $u"; else echo "  DIFF: $u"; mm=$((mm + 1)); fi
done
(( mm == 0 )) || fail_step 4 "$mm replica(s) diverged from ${REACH[0]}"
pass_step 4 "Body-identity"

total_ok=0
for u in "${REACH[@]}"; do total_ok=$((total_ok + ${OK[$u]})); done
pass_step 5 "LB-fairness (informational)" "total_ok=$total_ok"

for u in "${REACH[@]}"; do
    c=$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$u/health" 2>/dev/null) || c="000"
    [[ "$c" == "200" ]] || fail_step 6 "$u answered HTTP $c after the load"
done
pass_step 6 "Post-scaling liveness"

(( STEPS_OK == STEPS )) ||
    fail_step "$STEPS" "executed-step floor — $STEPS_OK of $STEPS steps were judged and passed"
echo
echo "=== $NAME: PASSED ==="
echo "  evidence: ${PERMISSIVE}replicas=${#REACH[@]} down=${#DOWN[@]} allow_down=${ALLOW_DOWN} total_ok=${total_ok}"
