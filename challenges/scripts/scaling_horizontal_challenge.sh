#!/usr/bin/env bash
# scaling_horizontal_challenge.sh — anti-bluff Scaling Challenge for
# LLMProvider per CONST-035 + CONST-050(B). Cascade per CONST-051(A).
#
# ── EXIT CODES — three-valued, and 2 is NEVER a pass ─────────────────────────
#   0  the challenge ran against a real replica set and every assertion passed
#   1  a real finding: the replica set was there to be tested and an assertion failed
#   2  COULD NOT DETERMINE — $LLMPROVIDER_VD_SCALING_REPLICA_URLS is
#      unset, or fewer than two of the listed replicas answered 200 — a
#      horizontal-scaling claim cannot be tested against one replica.
#      NOTHING was measured, so the subject is NOT known to be healthy.
#
# Why the third value exists. Every branch below that could not reach its
# replica set used to print "PASSED (SKIP-OK)" and exit 0. That made a DNS blip, a
# restarting replica set, an unexported variable and a typo'd URL
# indistinguishable — BY EXIT CODE — from a genuine pass, so any caller that
# reads the exit status (the only thing most callers read) was told the subject
# had been tested and was fine when in fact nothing had been tested at all.
# "Could not test" must never read as "tested and fine".

set -uo pipefail
REPLICAS="${LLMPROVIDER_VD_SCALING_REPLICA_URLS:-}"
REQS="${SCALING_REQS_PER_REPLICA:-50}"
CONC="${SCALING_CONCURRENCY:-10}"
MIN_PCT="${SCALING_MIN_PASS_PCT:-95}"

echo "=== LLMProvider Scaling Challenge ==="
echo "  replicas=$REPLICAS reqs=$REQS conc=$CONC pass≥${MIN_PCT}%"

if [[ -z "$REPLICAS" ]]; then
    echo "[1/6] COULD NOT DETERMINE: LLMPROVIDER_VD_SCALING_REPLICA_URLS is unset — no replica set to test."
    echo "  Nothing was measured. LLMProvider Scaling Challenge is NOT known to be healthy."
    echo "=== LLMProvider Scaling Challenge: COULD NOT DETERMINE (#env-no-replicas) ==="
    exit 2
fi

IFS=',' read -r -a URLS <<< "$REPLICAS"
REACH=()
for u in "${URLS[@]}"; do
    u="${u// /}"; [[ -z "$u" ]] && continue
    c=$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$u/health" 2>/dev/null) || c="000"
    [[ "$c" == "200" ]] && REACH+=("$u") && echo "  reachable: $u"
done
if [[ ${#REACH[@]} -lt 2 ]]; then
    echo "[1/6] COULD NOT DETERMINE: only ${#REACH[@]} of ${#URLS[@]} replica(s) answered 200 — need at least 2."
    echo "  Nothing was measured. LLMProvider Scaling Challenge is NOT known to be healthy."
    echo "=== LLMProvider Scaling Challenge: COULD NOT DETERMINE (#env-single-replica) ==="
    exit 2
fi
echo "[1/6] Topology: ${#REACH[@]} replicas — PASS"

for u in "${REACH[@]}"; do
    b=$(curl -sS --max-time 5 "$u/health" 2>/dev/null || true)
    # Bash's own regex, NOT `printf ... | grep -qE`: under the `set -o pipefail`
    # above, grep -q closing the pipe on a match kills printf with SIGPIPE (141)
    # and pipefail promotes it — the check would FAIL BECAUSE the status matched.
    status_re='"status"[[:space:]]*:[[:space:]]*"(ok|healthy|UP)"'
    [[ $b =~ $status_re ]] || { echo "[2/6] FAIL: $u"; exit 1; }
done
echo "[2/6] Schema sanity: PASS"

declare -A OK
for u in "${REACH[@]}"; do
    r=$(mktemp)
    seq 1 "$REQS" | xargs -n1 -P "$CONC" -I{} \
        curl -sS -o /dev/null --max-time 5 -w "%{http_code}\n" "$u/health" 2>/dev/null >> "$r" || true
    okc=$(awk '$1=="200"{c++} END{print c+0}' "$r")
    tot=$(wc -l < "$r" | tr -d ' '); [[ "$tot" -eq 0 ]] && tot=1
    pct=$((okc * 100 / tot))
    OK[$u]=$okc; rm -f "$r"
    echo "  $u → $okc/$tot ($pct%)"
    [[ "$pct" -lt "$MIN_PCT" ]] && { echo "[3/6] FAIL"; exit 1; }
done
echo "[3/6] Per-replica load: PASS"

first=$(curl -sS --max-time 5 "${REACH[0]}/health" 2>/dev/null | sed 's/"uptime[^,}]*//g; s/"timestamp[^,}]*//g')
fh=$(printf '%s' "$first" | sha256sum | awk '{print $1}')
mm=0
for u in "${REACH[@]:1}"; do
    b=$(curl -sS --max-time 5 "$u/health" 2>/dev/null | sed 's/"uptime[^,}]*//g; s/"timestamp[^,}]*//g')
    h=$(printf '%s' "$b" | sha256sum | awk '{print $1}')
    [[ "$h" == "$fh" ]] && echo "  MATCH: $u" || { echo "  DIFF: $u"; mm=$((mm+1)); }
done
[[ "$mm" -gt 0 ]] && { echo "[4/6] FAIL: $mm diverged"; exit 1; }
echo "[4/6] Body-identity: PASS"

total_ok=0; for u in "${REACH[@]}"; do total_ok=$((total_ok + ${OK[$u]})); done
echo "[5/6] LB-fairness informational: total_ok=$total_ok"

for u in "${REACH[@]}"; do
    c=$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$u/health" 2>/dev/null) || c="000"
    [[ "$c" != "200" ]] && { echo "[6/6] FAIL"; exit 1; }
done
echo "[6/6] Post-scaling liveness: PASS"

echo
echo "=== LLMProvider Scaling Challenge: PASSED ==="
echo "  evidence: replicas=${#REACH[@]} total_ok=${total_ok}"
