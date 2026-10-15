#!/usr/bin/env bash
# ddos_health_flood_challenge.sh — anti-bluff DDoS Challenge per CONST-035 +
# CONST-050(B); cascaded per CONST-051(A). Floods <PREFIX>_HEALTH_URL
# (<PREFIX> = C50_PREFIX in the module-identity block below).
#
# Five steps, each of which must pass (the executed-step floor: a step that was
# skipped or deleted is a finding, never a quiet pass):
#   [1/5] pre-flood health answers 200
#   [2/5] the health body carries "status": "ok" | "healthy" | "UP"
#   [3/5] DDOS_REQUESTS requests at DDOS_CONCURRENCY are sent and every one is
#         recorded (the executed-request floor)
#   [4/5] at least DDOS_MIN_PASS_PCT % of them answered 200
#   [5/5] post-flood health answers 200
#
# Usage:
#   bash ddos_health_flood_challenge.sh
#   bash ddos_health_flood_challenge.sh --prove-failure
#
# Exit (three-valued; a finding outranks a could-not-determine; 2 is NEVER a pass):
#   0 = the target was flooded and held the pass threshold
#   1 = a real finding — wrong health schema, the target dropped below the
#       threshold or died, or a step was skipped (executed-step floor)
#   2 = COULD NOT DETERMINE — an unknown argument, no target configured, the
#       target not answering 200 before the flood, a knob that is not a valid
#       number, fewer responses recorded than requests sent, or no curl/timeout.
#       Nothing (more) was judged; the target is NOT known to be resilient.
#
# --prove-failure (§1.1 paired proof, VSC-369; harness lib/const050b_proof.sh):
# runs copies of THIS file against loopback stub targets. A healthy stub must
# pass (0); a stub reporting a wrong status, one that fails under the flood and
# one that dies right after it must each exit 1 naming their step; no target,
# an unknown argument, a refused port and a malformed knob must exit 2; a
# deleted step (planted in a copy) trips the floor (1) and a truncated flood
# (planted in a copy) trips the request floor (2).

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
    pf_begin "$SELF" "ddos challenge" curl python3 || return $?
    local U="${C50_PREFIX}_HEALTH_URL"
    # shellcheck disable=SC2034  # *_PORT is written by pf_stub (printf -v)
    local g_from c url url_PORT closed knobs=(DDOS_REQUESTS=40 DDOS_CONCURRENCY=8)
    if c="$(pf_copy c1)" && pf_stub url ok; then
        pf_expect "C1 healthy stub target passes all 5 steps" 0 "DDoS Challenge: PASSED" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "C1" "could not start the stub target"; fi
    if c="$(pf_copy m1)" && pf_stub url bad-schema; then
        pf_expect "M1 target reporting status \"down\"" 1 "[2/5] FAIL" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "M1" "could not start the stub target"; fi
    # counted GETs before the flood: pre-check, schema probe = 2
    if c="$(pf_copy m2)" && pf_stub url fail-after 10; then
        pf_expect "M2 target failing under the flood" 1 "[4/5] FAIL" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "M2" "could not start the stub target"; fi
    if c="$(pf_copy m3)" && pf_stub url fail-after 42; then
        pf_expect "M3 target dead right after the flood" 1 "[5/5] FAIL" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "M3" "could not start the stub target"; fi
    g_from='pass_step '; g_from+='5 "Post-flood liveness"'
    if c="$(pf_copy g1)" && pf_plant "$c" "$g_from" ':' && pf_stub url ok; then
        pf_expect "G1 a deleted step trips the executed-step floor" 1 "executed-step floor" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "G1" "could not plant the defect"; fi
    g_from='seq '; g_from+='1 "$TOTAL_REQS" |'
    if c="$(pf_copy g2)" && pf_plant "$c" "$g_from" 'seq 2 "$TOTAL_REQS" |' && pf_stub url ok; then
        pf_expect "G2 a truncated flood trips the executed-request floor" 2 "executed-request floor" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "G2" "could not plant the defect"; fi
    if c="$(pf_copy u1)"; then
        pf_expect "U1 no target configured" 2 "#env-no-target" pf_run "$c"
    else pf_undet "U1" "could not copy"; fi
    if c="$(pf_copy u2)" && pf_closed_port closed; then
        pf_expect "U2 unreachable target (refused loopback port)" 2 "#env-target-down" pf_run "$c" "$U=http://127.0.0.1:$closed/health" "${knobs[@]}"
    else pf_undet "U2" "could not find a refused port"; fi
    if c="$(pf_copy u3)" && pf_stub url ok; then
        pf_expect "U3 a knob that is not a number" 2 "#bad-knob" pf_run "$c" "$U=$url/health" DDOS_REQUESTS=many
    else pf_undet "U3" "could not start the stub target"; fi
    if c="$(pf_copy u4)"; then
        pf_expect "U4 unknown argument" 2 "unknown argument" pf_run "$c" -- --no-such-flag
    else pf_undet "U4" "could not copy"; fi
    pf_end
}

if [[ "$PROVE" -eq 1 ]]; then
    prove_failure
    exit $?
fi

# ── live challenge body ──────────────────────────────────────────────────────
URL_VAR="${C50_PREFIX}_HEALTH_URL"
HEALTH_URL="${!URL_VAR:-}"
TOTAL_REQS="${DDOS_REQUESTS:-500}"
CONCURRENCY="${DDOS_CONCURRENCY:-50}"
TIMEOUT_SEC="${DDOS_TIMEOUT_SEC:-5}"
MIN_PASS_PCT="${DDOS_MIN_PASS_PCT:-95}"
STEPS=5
STEPS_OK=0
NAME="$C50_MODULE DDoS Challenge"

pass_step() { STEPS_OK=$((STEPS_OK + 1)); echo "[$1/$STEPS] $2: PASS${3:+ ($3)}"; }
fail_step() { echo "[$1/$STEPS] FAIL: $2"; echo "=== $NAME: FAILED ==="; exit 1; }
undetermined() {
    echo "COULD NOT DETERMINE: $1"
    echo "  Nothing (more) was judged. $NAME is NOT known to be resilient."
    echo "=== $NAME: COULD NOT DETERMINE ($2) ==="
    exit 2
}

echo "=== $C50_MODULE DDoS Health-Flood Challenge ==="
echo "  url=$HEALTH_URL total=$TOTAL_REQS conc=$CONCURRENCY pass>=${MIN_PASS_PCT}%"

command -v curl >/dev/null 2>&1 || undetermined "'curl' is not on PATH" "#env-no-tool"
[[ -n "$HEALTH_URL" ]] || undetermined "$URL_VAR is unset — there is no target to flood" "#env-no-target"
for knob in TOTAL_REQS CONCURRENCY TIMEOUT_SEC; do
    [[ "${!knob}" =~ ^[1-9][0-9]*$ ]] ||
        undetermined "$knob='${!knob}' (DDOS_*) is not a positive integer" "#bad-knob"
done
[[ "$MIN_PASS_PCT" =~ ^[1-9][0-9]*$ ]] && (( MIN_PASS_PCT <= 100 )) ||
    undetermined "DDOS_MIN_PASS_PCT='$MIN_PASS_PCT' is not a percentage in 1..100" "#bad-knob"

pre=$(curl -sS --max-time "$TIMEOUT_SEC" -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || pre="000"
[[ "$pre" == "200" ]] || undetermined "target did not answer 200 before the flood (HTTP $pre)" "#env-target-down"
pass_step 1 "Pre-flood"

body=$(curl -sS --max-time "$TIMEOUT_SEC" "$HEALTH_URL" 2>/dev/null || true)
# Bash's own regex, NOT `printf ... | grep -qE`: under `set -o pipefail`, grep
# -q closing the pipe on a match kills printf with SIGPIPE (141) and pipefail
# promotes it — the check would FAIL BECAUSE the status matched.
status_re='"status"[[:space:]]*:[[:space:]]*"(ok|healthy|UP)"'
[[ $body =~ $status_re ]] || fail_step 2 "health body carries no \"status\": ok|healthy|UP"
pass_step 2 "Schema sanity"

RES=$(mktemp) || undetermined "mktemp failed (TMPDIR='${TMPDIR:-}')" "#env-no-scratch"
trap 'rm -f "$RES"' EXIT
start=$(date +%s.%N)
seq 1 "$TOTAL_REQS" | xargs -n1 -P "$CONCURRENCY" -I{} \
    curl -sS -o /dev/null --max-time "$TIMEOUT_SEC" \
        -w "%{http_code} %{time_total}\n" "$HEALTH_URL" 2>/dev/null >> "$RES" || true
end=$(date +%s.%N)
wall=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b-a}')
total=$(grep -c . "$RES")
# Executed-request floor: every request must have left a status line (curl
# writes 000 for a refused or timed-out one), else the flood did not run.
(( total == TOTAL_REQS )) ||
    undetermined "executed-request floor: $total of $TOTAL_REQS requests recorded — the flood did not run" "#load-not-run"
ok=$(awk '$1=="200"{c++} END{print c+0}' "$RES")
pct=$((ok * 100 / total))
sorted=$(awk '{print $2}' "$RES" | sort -n)
p50=$(printf '%s\n' "$sorted" | awk -v n="$total" 'NR==int(n*0.5){print; exit}')
p95=$(printf '%s\n' "$sorted" | awk -v n="$total" 'NR==int(n*0.95){print; exit}')
pass_step 3 "Flood" "total=$total ok=$ok pct=${pct}% wall=${wall}s p50=${p50:-N/A}s p95=${p95:-N/A}s"

(( pct >= MIN_PASS_PCT )) || fail_step 4 "$ok/$total = ${pct}% answered 200 (< ${MIN_PASS_PCT}%)"
pass_step 4 "Threshold"

post=$(curl -sS --max-time "$TIMEOUT_SEC" -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || post="000"
[[ "$post" == "200" ]] || fail_step 5 "health answered HTTP $post after the flood"
pass_step 5 "Post-flood liveness"

(( STEPS_OK == STEPS )) ||
    fail_step "$STEPS" "executed-step floor — $STEPS_OK of $STEPS steps were judged and passed"
echo
echo "=== $NAME: PASSED ==="
echo "  evidence: reqs=$total ok=$ok pct=${pct}% wall=${wall}s p50=${p50:-N/A}s p95=${p95:-N/A}s"
