#!/usr/bin/env bash
# ddos_health_flood_challenge.sh — anti-bluff DDoS Challenge for
# LLMProvider per CONST-035 + CONST-050(B). Submodule cascade per
# CONST-051(A). Targets $LLMPROVIDER_VD_HEALTH_URL.
#
# ── EXIT CODES — three-valued, and 2 is NEVER a pass ─────────────────────────
#   0  the challenge ran against a real target and every assertion passed
#   1  a real finding: the target was there to be tested and an assertion failed
#   2  COULD NOT DETERMINE — $LLMPROVIDER_VD_HEALTH_URL is unset, or the
#      target did not answer 200 before the flood started.
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
TOTAL_REQS="${DDOS_REQUESTS:-500}"
CONCURRENCY="${DDOS_CONCURRENCY:-50}"
TIMEOUT_SEC="${DDOS_TIMEOUT_SEC:-5}"
MIN_PASS_PCT="${DDOS_MIN_PASS_PCT:-95}"

echo "=== LLMProvider DDoS Health-Flood Challenge ==="
echo "  url=$HEALTH_URL total=$TOTAL_REQS conc=$CONCURRENCY pass≥${MIN_PASS_PCT}%"

if [[ -z "$HEALTH_URL" ]]; then
    echo "[1/5] COULD NOT DETERMINE: LLMPROVIDER_VD_HEALTH_URL is unset — there is no target to flood."
    echo "  Nothing was measured. LLMProvider DDoS Challenge is NOT known to be healthy."
    echo "=== LLMProvider DDoS Challenge: COULD NOT DETERMINE (#env-no-target) ==="
    exit 2
fi
pre=$(curl -sS --max-time "$TIMEOUT_SEC" -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || pre="000"
if [[ "$pre" != "200" ]]; then
    echo "[1/5] COULD NOT DETERMINE: target did not answer 200 before the flood (HTTP $pre)."
    echo "  Nothing was measured. LLMProvider DDoS Challenge is NOT known to be healthy."
    echo "=== LLMProvider DDoS Challenge: COULD NOT DETERMINE (#env-target-down) ==="
    exit 2
fi
echo "[1/5] Pre-flood: PASS"

body=$(curl -sS --max-time "$TIMEOUT_SEC" "$HEALTH_URL" 2>/dev/null || true)
printf '%s' "$body" | grep -qE '"status"\s*:\s*"(ok|healthy|UP)"' || { echo "[2/5] FAIL"; exit 1; }
echo "[2/5] Schema sanity: PASS"

RES=$(mktemp); trap "rm -f $RES" EXIT
start=$(date +%s.%N)
seq 1 "$TOTAL_REQS" | xargs -n1 -P "$CONCURRENCY" -I{} \
    curl -sS -o /dev/null --max-time "$TIMEOUT_SEC" \
        -w "%{http_code} %{time_total}\n" "$HEALTH_URL" 2>/dev/null >> "$RES" || true
end=$(date +%s.%N)
wall=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b-a}')
total=$(wc -l < "$RES" | tr -d ' '); [[ "$total" -eq 0 ]] && total=1
ok=$(awk '$1=="200"{c++} END{print c+0}' "$RES")
pct=$((ok * 100 / total))
sorted=$(awk '{print $2}' "$RES" | sort -n)
p50=$(printf '%s\n' "$sorted" | awk -v n="$total" 'NR==int(n*0.5){print; exit}')
p95=$(printf '%s\n' "$sorted" | awk -v n="$total" 'NR==int(n*0.95){print; exit}')

echo "[3/5] Flood: total=$total ok=$ok pct=${pct}% wall=${wall}s p50=${p50:-N/A}s p95=${p95:-N/A}s"
[[ "$pct" -lt "$MIN_PASS_PCT" ]] && { echo "[4/5] FAIL"; exit 1; }
echo "[4/5] Threshold: PASS"

post=$(curl -sS --max-time "$TIMEOUT_SEC" -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || post="000"
[[ "$post" != "200" ]] && { echo "[5/5] FAIL"; exit 1; }
echo "[5/5] Post-flood liveness: PASS"

echo
echo "=== LLMProvider DDoS Challenge: PASSED ==="
echo "  evidence: reqs=$total ok=$ok pct=${pct}% wall=${wall}s p50=${p50:-N/A}s p95=${p95:-N/A}s"
