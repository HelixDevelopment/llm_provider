#!/usr/bin/env bash
# stress_sustained_load_challenge.sh — anti-bluff Stress Challenge per
# CONST-035 + CONST-050(B); cascaded per CONST-051(A). Sustained load against
# <PREFIX>_HEALTH_URL (<PREFIX> = C50_PREFIX in the module-identity block below).
#
# Six steps, each of which must pass (the executed-step floor: a step that was
# skipped or deleted is a finding, never a quiet pass):
#   [1/6] pre-stress health answers 200
#   [2/6] baseline latency: at least 5 of 10 probes answer 200; their median
#   [3/6] the health body carries "status": "ok" | "healthy" | "UP"
#   [4/6] STRESS_DURATION_SEC x STRESS_REQUESTS_PER_SEC requests at
#         STRESS_CONCURRENCY reach STRESS_MIN_PASS_PCT % 200s (every request
#         must be recorded)
#   [5/6] post-load latency: at least 5 of 10 probes answer 200 and their
#         median is within STRESS_MAX_LATENCY_DEGRADATION_PCT % of the baseline
#   [6/6] post-stress health answers 200
# A latency sample is the time of a probe that answered 200; a failed probe is
# never a sample (it used to be one, and its near-zero time could make an
# absent baseline or post-load median look fast).
#
# Usage:
#   bash stress_sustained_load_challenge.sh
#   bash stress_sustained_load_challenge.sh --prove-failure
#
# Exit (three-valued; a finding outranks a could-not-determine; 2 is NEVER a pass):
#   0 = the target sustained the load within the pass and degradation budgets
#   1 = a real finding — baseline or post-load probes failing, wrong health
#       schema, throughput or latency budget blown, the target died, or a step
#       was skipped (executed-step floor)
#   2 = COULD NOT DETERMINE — an unknown argument, no target configured, the
#       target not answering 200 before the load, a knob that is not a valid
#       number, fewer responses recorded than requests sent, or no curl.
#       Nothing (more) was judged; the target is NOT known to be healthy.
#
# --prove-failure (§1.1 paired proof, VSC-369; harness lib/const050b_proof.sh):
# runs copies of THIS file against loopback stub targets. A healthy stub must
# pass (0); a stub failing its baseline, one reporting a wrong status, one
# failing under load, one that slows down after the load and one dying right
# after it must each exit 1 naming their step; no target, an unknown argument,
# a refused port and a malformed knob must exit 2; a deleted step (planted in
# a copy) trips the floor (1) and a truncated load (planted in a copy) trips
# the request floor (2). Loopback latencies are sub-millisecond, so the proof
# runs with a 5000 % degradation budget and a 0.3 s slow-down (> 100x).

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
    pf_begin "$SELF" "stress challenge" curl python3 || return $?
    local U="${C50_PREFIX}_HEALTH_URL"
    # shellcheck disable=SC2034  # *_PORT is written by pf_stub (printf -v)
    local g_from c url url_PORT closed
    local knobs=(STRESS_DURATION_SEC=1 STRESS_REQUESTS_PER_SEC=20 STRESS_CONCURRENCY=5 STRESS_MAX_LATENCY_DEGRADATION_PCT=5000)
    if c="$(pf_copy c1)" && pf_stub url ok; then
        pf_expect "C1 healthy stub target passes all 6 steps" 0 "Stress Challenge: PASSED" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "C1" "could not start the stub target"; fi
    # counted GETs: pre-check 1, baseline 10, schema 1, load 20, post-load 10, final 1
    if c="$(pf_copy m1)" && pf_stub url fail-after 1; then
        pf_expect "M1 target failing its baseline probes" 1 "[2/6] FAIL" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "M1" "could not start the stub target"; fi
    if c="$(pf_copy m2)" && pf_stub url bad-schema; then
        pf_expect "M2 target reporting status \"down\"" 1 "[3/6] FAIL" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "M2" "could not start the stub target"; fi
    if c="$(pf_copy m3)" && pf_stub url fail-after 16; then
        pf_expect "M3 target failing under the sustained load" 1 "[4/6] FAIL" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "M3" "could not start the stub target"; fi
    if c="$(pf_copy m4)" && pf_stub url slow-after 32:0.3; then
        pf_expect "M4 target slowing down after the load" 1 "[5/6] FAIL" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "M4" "could not start the stub target"; fi
    if c="$(pf_copy m5)" && pf_stub url fail-after 42; then
        pf_expect "M5 target dead right after the load" 1 "[6/6] FAIL" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "M5" "could not start the stub target"; fi
    g_from='pass_step '; g_from+='6 "Post-stress liveness"'
    if c="$(pf_copy g1)" && pf_plant "$c" "$g_from" ':' && pf_stub url ok; then
        pf_expect "G1 a deleted step trips the executed-step floor" 1 "executed-step floor" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "G1" "could not plant the defect"; fi
    g_from='seq '; g_from+='1 "$total_target" |'
    if c="$(pf_copy g2)" && pf_plant "$c" "$g_from" 'seq 2 "$total_target" |' && pf_stub url ok; then
        pf_expect "G2 a truncated load trips the executed-request floor" 2 "executed-request floor" pf_run "$c" "$U=$url/health" "${knobs[@]}"
    else pf_undet "G2" "could not plant the defect"; fi
    if c="$(pf_copy u1)"; then
        pf_expect "U1 no target configured" 2 "#env-no-target" pf_run "$c"
    else pf_undet "U1" "could not copy"; fi
    if c="$(pf_copy u2)" && pf_closed_port closed; then
        pf_expect "U2 unreachable target (refused loopback port)" 2 "#env-target-down" pf_run "$c" "$U=http://127.0.0.1:$closed/health" "${knobs[@]}"
    else pf_undet "U2" "could not find a refused port"; fi
    if c="$(pf_copy u3)" && pf_stub url ok; then
        pf_expect "U3 a knob that is not a number" 2 "#bad-knob" pf_run "$c" "$U=$url/health" STRESS_DURATION_SEC=1.5
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
DURATION="${STRESS_DURATION_SEC:-15}"
RPS="${STRESS_REQUESTS_PER_SEC:-50}"
CONCURRENCY="${STRESS_CONCURRENCY:-20}"
TIMEOUT_SEC="${STRESS_TIMEOUT_SEC:-5}"
MIN_PASS_PCT="${STRESS_MIN_PASS_PCT:-95}"
MAX_DEG_PCT="${STRESS_MAX_LATENCY_DEGRADATION_PCT:-300}"
STEPS=6
STEPS_OK=0
NAME="$C50_MODULE Stress Challenge"

pass_step() { STEPS_OK=$((STEPS_OK + 1)); echo "[$1/$STEPS] $2: PASS${3:+ ($3)}"; }
fail_step() { echo "[$1/$STEPS] FAIL: $2"; echo "=== $NAME: FAILED ==="; exit 1; }
undetermined() {
    echo "COULD NOT DETERMINE: $1"
    echo "  Nothing (more) was judged. $NAME is NOT known to be healthy."
    echo "=== $NAME: COULD NOT DETERMINE ($2) ==="
    exit 2
}

echo "=== $C50_MODULE Stress Sustained-Load Challenge ==="
echo "  url=$HEALTH_URL dur=${DURATION}s rps=${RPS}"

command -v curl >/dev/null 2>&1 || undetermined "'curl' is not on PATH" "#env-no-tool"
[[ -n "$HEALTH_URL" ]] || undetermined "$URL_VAR is unset — there is no target to load" "#env-no-target"
for knob in DURATION RPS CONCURRENCY TIMEOUT_SEC; do
    [[ "${!knob}" =~ ^[1-9][0-9]*$ ]] ||
        undetermined "$knob='${!knob}' (STRESS_*) is not a positive integer" "#bad-knob"
done
[[ "$MIN_PASS_PCT" =~ ^[1-9][0-9]*$ ]] && (( MIN_PASS_PCT <= 100 )) ||
    undetermined "STRESS_MIN_PASS_PCT='$MIN_PASS_PCT' is not a percentage in 1..100" "#bad-knob"
[[ "$MAX_DEG_PCT" =~ ^(0|[1-9][0-9]*)$ ]] ||
    undetermined "STRESS_MAX_LATENCY_DEGRADATION_PCT='$MAX_DEG_PCT' is not a non-negative integer" "#bad-knob"

base=""
RES=""
post_base=""
trap 'rm -f -- "$base" "$RES" "$post_base"' EXIT
base=$(mktemp) && RES=$(mktemp) && post_base=$(mktemp) ||
    undetermined "mktemp failed (TMPDIR='${TMPDIR:-}')" "#env-no-scratch"

# sample FILE — 10 probes, one "<http_code> <time_total>" line each.
sample() {
    local _
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        curl -sS -o /dev/null --max-time "$TIMEOUT_SEC" -w "%{http_code} %{time_total}\n" "$HEALTH_URL" 2>/dev/null >> "$1" || true
    done
}
# median_of FILE — median time of the probes that answered 200 (empty if < 5).
median_of() {
    awk '$1=="200"{print $2}' "$1" | sort -n | awk '{v[NR]=$1} END{if (NR >= 5) print v[int((NR + 1) / 2)]}'
}

pre=$(curl -sS --max-time "$TIMEOUT_SEC" -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || pre="000"
[[ "$pre" == "200" ]] || undetermined "target did not answer 200 before the load (HTTP $pre)" "#env-target-down"
pass_step 1 "Pre-stress"

sample "$base"
base_n=$(awk '$1=="200"{c++} END{print c+0}' "$base")
base_med=$(median_of "$base")
[[ -n "$base_med" ]] || fail_step 2 "only $base_n of 10 baseline probes answered 200 — the target is failing before any load"
pass_step 2 "Baseline" "median ${base_med}s, $base_n/10 probes"

body=$(curl -sS --max-time "$TIMEOUT_SEC" "$HEALTH_URL" 2>/dev/null || true)
# Bash's own regex, NOT `printf ... | grep -qE`: under `set -o pipefail`, grep
# -q closing the pipe on a match kills printf with SIGPIPE (141) and pipefail
# promotes it — the check would FAIL BECAUSE the status matched.
status_re='"status"[[:space:]]*:[[:space:]]*"(ok|healthy|UP)"'
[[ $body =~ $status_re ]] || fail_step 3 "health body carries no \"status\": ok|healthy|UP"
pass_step 3 "Schema sanity"

start=$(date +%s.%N)
total_target=$((DURATION * RPS))
seq 1 "$total_target" | xargs -n1 -P "$CONCURRENCY" -I{} \
    curl -sS -o /dev/null --max-time "$TIMEOUT_SEC" \
        -w "%{http_code} %{time_total}\n" "$HEALTH_URL" 2>/dev/null >> "$RES" || true
finish=$(date +%s.%N)
wall=$(awk -v a="$start" -v b="$finish" 'BEGIN{printf "%.3f", b-a}')
total=$(grep -c . "$RES")
# Executed-request floor: every request must have left a status line (curl
# writes 000 for a refused or timed-out one), else the load did not run.
(( total == total_target )) ||
    undetermined "executed-request floor: $total of $total_target requests recorded — the load did not run" "#load-not-run"
ok=$(awk '$1=="200"{c++} END{print c+0}' "$RES")
pct=$((ok * 100 / total))
(( pct >= MIN_PASS_PCT )) || fail_step 4 "$ok/$total = ${pct}% answered 200 (< ${MIN_PASS_PCT}%)"
pass_step 4 "Sustained" "$ok/$total ${pct}% wall=${wall}s"

sample "$post_base"
post_n=$(awk '$1=="200"{c++} END{print c+0}' "$post_base")
post_med=$(median_of "$post_base")
[[ -n "$post_med" ]] || fail_step 5 "only $post_n of 10 post-load probes answered 200 — post-load latency unmeasurable"
deg=$(awk -v a="$base_med" -v b="$post_med" 'BEGIN{if(a<=0)a=0.0001; printf "%.0f", (b-a)*100/a}')
(( deg <= MAX_DEG_PCT )) || fail_step 5 "latency ${base_med}s -> ${post_med}s = ${deg}% (> ${MAX_DEG_PCT}%)"
pass_step 5 "Latency" "${base_med}s -> ${post_med}s, delta ${deg}%, $post_n/10 probes"

post=$(curl -sS --max-time "$TIMEOUT_SEC" -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || post="000"
[[ "$post" == "200" ]] || fail_step 6 "health answered HTTP $post after the load"
pass_step 6 "Post-stress liveness"

(( STEPS_OK == STEPS )) ||
    fail_step "$STEPS" "executed-step floor — $STEPS_OK of $STEPS steps were judged and passed"
echo
echo "=== $NAME: PASSED ==="
echo "  evidence: dur=${wall}s reqs=${total} pct=${pct}% baseline=${base_med}s deg=${deg}%"
