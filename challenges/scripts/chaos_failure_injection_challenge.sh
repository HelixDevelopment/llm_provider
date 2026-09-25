#!/usr/bin/env bash
# chaos_failure_injection_challenge.sh — anti-bluff Chaos Challenge
# for LLMProvider per CONST-035 + CONST-050(B). Cascade per CONST-051(A).
#
# ── EXIT CODES — three-valued, and 2 is NEVER a pass ─────────────────────────
#   0  the challenge ran against a real target and every assertion passed
#   1  a real finding: the target was there to be tested and an assertion failed
#   2  COULD NOT DETERMINE — $LLMPROVIDER_VD_HEALTH_URL or
#      $LLMPROVIDER_VD_CHAOS_PORT is unset, or the target did not answer 200
#      before injection started.
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
CHAOS_HOST="${LLMPROVIDER_VD_CHAOS_HOST:-localhost}"
CHAOS_PORT="${LLMPROVIDER_VD_CHAOS_PORT:-}"
LEGIT_REQS="${CHAOS_LEGIT_REQUESTS:-100}"
MIN_PCT="${CHAOS_LEGIT_MIN_PASS_PCT:-95}"

echo "=== LLMProvider Chaos Failure-Injection Challenge ==="
echo "  url=$HEALTH_URL host=${CHAOS_HOST}:${CHAOS_PORT}"

if [[ -z "$HEALTH_URL" ]] || [[ -z "$CHAOS_PORT" ]]; then
    echo "[1/6] COULD NOT DETERMINE: LLMPROVIDER_VD_HEALTH_URL/_PORT unset — there is no target to injure."
    echo "  Nothing was measured. LLMProvider Chaos Challenge is NOT known to be healthy."
    echo "=== LLMProvider Chaos Challenge: COULD NOT DETERMINE (#env-no-target) ==="
    exit 2
fi
pre=$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || pre="000"
if [[ "$pre" != "200" ]]; then
    echo "[1/6] COULD NOT DETERMINE: target did not answer 200 before injection (HTTP $pre)."
    echo "  Nothing was measured. LLMProvider Chaos Challenge is NOT known to be healthy."
    echo "=== LLMProvider Chaos Challenge: COULD NOT DETERMINE (#env-target-down) ==="
    exit 2
fi
echo "[1/6] Pre-chaos: PASS"

for case in "BADVERB / HTTP/1.1" "GET / HTTP/9.9" "GET // HTTP/1.1" "INVALID"; do
    printf '%s\r\nHost: %s\r\n\r\n' "$case" "$CHAOS_HOST" 2>/dev/null \
        | timeout 3 bash -c "exec 3<>/dev/tcp/$CHAOS_HOST/$CHAOS_PORT && cat >&3 && cat <&3" >/dev/null 2>&1 || true
done
post=$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || post="000"
[[ "$post" != "200" ]] && { echo "[2/6] FAIL"; exit 1; }
echo "[2/6] Malformed-salvo: PASS"

huge=$(head -c 8192 /dev/urandom | tr -dc 'A-Za-z0-9' | head -c 8192)
oversize=$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" -H "X-Chaos-Huge: $huge" "$HEALTH_URL" 2>/dev/null) || oversize="000"
case "$oversize" in
    200|400|413|414|431|494) echo "[3/6] Oversized: PASS (HTTP $oversize)";;
    *) echo "[3/6] FAIL"; exit 1;;
esac

pids=()
for _ in $(seq 1 5); do
    (timeout 5 bash -c "
        exec 3<>/dev/tcp/$CHAOS_HOST/$CHAOS_PORT 2>/dev/null
        printf 'GET /health HTTP/1.1\r\nHost: $CHAOS_HOST\r\nX-Slow: ' >&3 2>/dev/null
        sleep 4 2>/dev/null
        printf 'done\r\n\r\n' >&3 2>/dev/null
    " >/dev/null 2>&1 || true) &
    pids+=($!)
done
echo "[4/6] Slow-loris: ${#pids[@]} spawned"

RES=$(mktemp); trap "rm -f $RES; kill ${pids[*]} 2>/dev/null" EXIT
seq 1 "$LEGIT_REQS" | xargs -n1 -P 20 -I{} \
    curl -sS -o /dev/null --max-time 5 -w "%{http_code}\n" "$HEALTH_URL" 2>/dev/null >> "$RES" || true
for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; wait 2>/dev/null || true
total=$(wc -l < "$RES" | tr -d ' '); [[ "$total" -eq 0 ]] && total=1
ok=$(awk '$1=="200"{c++} END{print c+0}' "$RES")
pct=$((ok * 100 / total))
echo "[5/6] Mixed-load: $ok/$total ${pct}%"
[[ "$pct" -lt "$MIN_PCT" ]] && { echo "  FAIL"; exit 1; }

final=$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null) || final="000"
[[ "$final" != "200" ]] && { echo "[6/6] FAIL"; exit 1; }
echo "[6/6] Post-chaos: PASS"

echo
echo "=== LLMProvider Chaos Challenge: PASSED ==="
echo "  evidence: legit=$total pct=${pct}% slow_loris=${#pids[@]} oversize=${oversize}"
