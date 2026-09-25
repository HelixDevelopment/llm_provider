#!/usr/bin/env bash
# stress_sustained_load_challenge.sh — anti-bluff Stress Challenge
# for LLMProvider per CONST-035 + CONST-050(B). Cascade per CONST-051(A).
#
# ── EXIT CODES — three-valued, and 2 is NEVER a pass ─────────────────────────
#   0  the challenge ran against a real target and every assertion passed
#   1  a real finding: the target was there to be tested and an assertion failed
#   2  COULD NOT DETERMINE — $LLMPROVIDER_VD_HEALTH_URL is unset, or the
#      target did not answer 200 before the load started.
#      NOTHING was measured, so the subject is NOT known to be healthy.
#
# Why the third value exists. Every branch below that could not reach its
# target used to print "PASSED (SKIP-OK)" and exit 0. That made a DNS blip, a
# restarting target, an unexported variable and a typo'd URL
# indistinguishable — BY EXIT CODE — from a genuine pass, so any caller that
# reads the exit status (the only thing most callers read) was told the subject
# had been tested and was fine when in fact nothing had been tested at all.
# "Could not test" must never read as "tested and fine".

set -uo pipefail
HEALTH_URL="${LLMPROVIDER_VD_HEALTH_URL:-}"
DURATION="${STRESS_DURATION_SEC:-15}"
RPS="${STRESS_REQUESTS_PER_SEC:-50}"
CONCURRENCY="${STRESS_CONCURRENCY:-20}"
TIMEOUT_SEC="${STRESS_TIMEOUT_SEC:-5}"
MIN_PASS_PCT="${STRESS_MIN_PASS_PCT:-95}"
MAX_DEG_PCT="${STRESS_MAX_LATENCY_DEGRADATION_PCT:-300}"

echo "=== LLMProvider Stress Sustained-Load Challenge ==="
echo "  url=$HEALTH_URL dur=${DURATION}s rps=${RPS}"

if [[ -z "$HEALTH_URL" ]]; then
    echo "[1/6] COULD NOT DETERMINE: LLMPROVIDER_VD_HEALTH_URL is unset — there is no target to load."
    echo "  Nothing was measured. LLMProvider Stress Challenge is NOT known to be healthy."
    echo "=== LLMProvider Stress Challenge: COULD NOT DETERMINE (#env-no-target) ==="
    exit 2
fi
pre=$(curl -sS --max-time "$TIMEOUT_SEC" -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || pre="000"
if [[ "$pre" != "200" ]]; then
    echo "[1/6] COULD NOT DETERMINE: target did not answer 200 before the load (HTTP $pre)."
    echo "  Nothing was measured. LLMProvider Stress Challenge is NOT known to be healthy."
    echo "=== LLMProvider Stress Challenge: COULD NOT DETERMINE (#env-target-down) ==="
    exit 2
fi
echo "[1/6] Pre-stress: PASS"

base=$(mktemp); trap "rm -f $base" EXIT
for _ in $(seq 1 10); do
    curl -sS -o /dev/null --max-time "$TIMEOUT_SEC" -w "%{time_total}\n" "$HEALTH_URL" 2>/dev/null >> "$base" || true
done
base_med=$(sort -n "$base" | awk 'NR==5{print; exit}')
echo "[2/6] Baseline median: ${base_med}s"

body=$(curl -sS --max-time "$TIMEOUT_SEC" "$HEALTH_URL" 2>/dev/null || true)
# Bash's own regex, NOT `printf ... | grep -qE`: under the `set -o pipefail`
# above, grep -q closing the pipe on a match kills printf with SIGPIPE (141)
# and pipefail promotes it — the check would FAIL BECAUSE the status matched.
status_re='"status"[[:space:]]*:[[:space:]]*"(ok|healthy|UP)"'
[[ $body =~ $status_re ]] || { echo "[3/6] FAIL"; exit 1; }
echo "[3/6] Schema sanity: PASS"

RES=$(mktemp); trap "rm -f $base $RES" EXIT
start=$(date +%s.%N)
total_target=$((DURATION * RPS))
seq 1 "$total_target" | xargs -n1 -P "$CONCURRENCY" -I{} \
    curl -sS -o /dev/null --max-time "$TIMEOUT_SEC" \
        -w "%{http_code} %{time_total}\n" "$HEALTH_URL" 2>/dev/null >> "$RES" || true
finish=$(date +%s.%N)
wall=$(awk -v a="$start" -v b="$finish" 'BEGIN{printf "%.3f", b-a}')
total=$(wc -l < "$RES" | tr -d ' '); [[ "$total" -eq 0 ]] && total=1
ok=$(awk '$1=="200"{c++} END{print c+0}' "$RES")
pct=$((ok * 100 / total))
echo "[4/6] Sustained: $ok/$total ${pct}% wall=${wall}s"
[[ "$pct" -lt "$MIN_PASS_PCT" ]] && { echo "  FAIL"; exit 1; }

post_base=$(mktemp); trap "rm -f $base $RES $post_base" EXIT
for _ in $(seq 1 10); do
    curl -sS -o /dev/null --max-time "$TIMEOUT_SEC" -w "%{time_total}\n" "$HEALTH_URL" 2>/dev/null >> "$post_base" || true
done
post_med=$(sort -n "$post_base" | awk 'NR==5{print; exit}')
deg=$(awk -v a="$base_med" -v b="$post_med" 'BEGIN{if(a<=0)a=0.0001; printf "%.0f", (b-a)*100/a}')
echo "[5/6] Latency: ${base_med}s → ${post_med}s (Δ=${deg}%)"
[[ "$deg" -gt "$MAX_DEG_PCT" ]] && { echo "  FAIL"; exit 1; }

post=$(curl -sS --max-time "$TIMEOUT_SEC" -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || post="000"
[[ "$post" != "200" ]] && { echo "[6/6] FAIL"; exit 1; }
echo "[6/6] Post-stress liveness: PASS"

echo
echo "=== LLMProvider Stress Challenge: PASSED ==="
echo "  evidence: dur=${wall}s reqs=${total} pct=${pct}% baseline=${base_med}s deg=${deg}%"
