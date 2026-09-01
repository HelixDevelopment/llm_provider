#!/usr/bin/env bash
# host_no_auto_suspend_contract_challenge.sh — CONST-033 anti-bluff gate.
#
# host_no_auto_suspend_challenge.sh asserts that the host cannot suspend itself
# mid-workload. Its failure mode is not a crash — it is returning a CONFIDENT
# verdict about a host it could not measure. Two opposite conditions used to
# collapse into the same exit 1:
#
#   * the guard was installed and has since been dismantled  -> fix the guard
#   * this host does not run systemd at all                  -> configure the
#     power manager it DOES use; this challenge inspects none of them
#
# The first is a defect report. The second is "not applicable", and answering
# it with FAIL sends an operator to repair something that was never there,
# while the host stays free to suspend by a route this challenge never looks at.
#
# This challenge drives the challenge-under-test through six host conditions —
# four of which cannot occur on the machine running it — and asserts its exit
# code against the three-valued contract declared in its own header:
#
#   0 = guard verified active
#   1 = guard genuinely broken (systemd is live, an assertion really failed)
#   2 = could not determine / not applicable — NEVER a pass
#
# Every run is sandboxed: the challenge-under-test executes inside a user+mount
# namespace with throwaway directories bind-mounted over /etc/systemd and
# /run/systemd, so the real host's power configuration is NEVER read from or
# written to, and the real journal is never consulted.
#
# Exit:
#   0 = all assertions PASS
#   1 = one or more FAIL
#   2 = invocation error, or no usable user namespace to sandbox with

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="${HERE}/host_no_auto_suspend_challenge.sh"

PASS_COUNT=0
FAIL_COUNT=0
assert_pass() { echo "PASS: $*"; PASS_COUNT=$((PASS_COUNT + 1)); }
assert_fail() { echo "FAIL: $*" >&2; FAIL_COUNT=$((FAIL_COUNT + 1)); }

echo "=== host_no_auto_suspend_contract_challenge ==="
echo

if [[ ! -f "$SUT" ]]; then
  echo "ERROR: challenge under test not found: $SUT" >&2
  exit 2
fi

# --- sandbox availability -------------------------------------------------
# Honest 2, not a silent 0: without user namespaces this gate cannot fake a
# host condition or protect the real /etc and /run, so it cannot judge.
if ! command -v unshare >/dev/null 2>&1; then
  echo "COULD NOT DETERMINE: unshare(1) not available — cannot sandbox the challenge." >&2
  exit 2
fi
if ! unshare -rm true >/dev/null 2>&1; then
  echo "COULD NOT DETERMINE: unprivileged user+mount namespaces are unavailable" >&2
  echo "  on this host, so the challenge cannot be run against a fabricated" >&2
  echo "  /etc/systemd and /run/systemd." >&2
  exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# A PATH holding only the utilities the challenge legitimately needs, so this
# gate controls precisely whether systemctl and journalctl are on it.
mkdir -p "$WORK/bin"
for b in bash mkdir cat rm ls id find grep sed tr head tail cut date stat wc printf; do
  p="$(command -v "$b" 2>/dev/null)" && ln -sf "$p" "$WORK/bin/$b"
done

# Writes a stub systemctl whose is-enabled answer this gate dictates.
# $1 = target dir, $2 = state reported for every sleep target
make_stub_systemctl() {
  local dir="$1" state="$2"
  cat > "$dir/systemctl" <<STUB
#!/usr/bin/env bash
case "\$1" in
  is-enabled) echo "$state"; exit 0 ;;
  *) echo "stub systemctl: unhandled '\$*'" >&2; exit 1 ;;
esac
STUB
  chmod +x "$dir/systemctl"
}

# A journal containing whatever lines this gate wants it to contain.
# $1 = target dir, $2 = text the journal returns
make_stub_journalctl() {
  local dir="$1" text="$2"
  cat > "$dir/journalctl" <<STUB
#!/usr/bin/env bash
printf '%s\n' "$text"
exit 0
STUB
  chmod +x "$dir/journalctl"
}

# Populates a throwaway /etc/systemd with the guard's two drop-ins.
seed_dropins() {
  local etc="$1"
  mkdir -p "$etc/sleep.conf.d" "$etc/logind.conf.d"
  printf '[Sleep]\nAllowSuspend=no\n'   > "$etc/sleep.conf.d/00-no-suspend.conf"
  printf '[Login]\nIdleAction=ignore\n' > "$etc/logind.conf.d/00-no-idle-suspend.conf"
}

# Runs the challenge with /etc/systemd bound over a throwaway dir and
# /run/systemd replaced by a tmpfs. Echoes stdout+stderr; returns its exit code.
# $1 = label, $2 = PATH to use, $3 = "booted" | "notbooted"
run_sut() {
  local label="$1" bindir="$2" booted="$3"
  local fake="$WORK/etc_$label"
  unshare -rm bash -c '
    mount --make-rprivate / 2>/dev/null
    mount -t tmpfs tmpfs /run/systemd || exit 99
    if [ "'"$booted"'" = "booted" ]; then mkdir -p /run/systemd/system; fi
    mount --bind "'"$fake"'" /etc/systemd || exit 99
    export PATH="'"$bindir"'"
    bash "'"$SUT"'"
  ' 2>&1
}

# ---------------------------------------------------------------------------
# 1. THE REGRESSION THIS GATE EXISTS FOR, part one. No systemctl anywhere:
#    the guard is INAPPLICABLE, not broken. Before the fix this returned 1.
# ---------------------------------------------------------------------------
echo "[1/6] host without systemd (no systemctl on PATH)"
mkdir -p "$WORK/etc_nosystemctl"
out="$(run_sut nosystemctl "$WORK/bin" notbooted)"; rc=$?
echo "    exit=$rc"
if [[ "$rc" -eq 2 ]]; then
  assert_pass "no systemctl -> exit 2 (could not determine / not applicable)"
else
  assert_fail "no systemctl -> exit $rc, want 2. Output: $(echo "$out" | tail -n2 | tr '\n' ' ')"
fi
if grep -q "^PASS:" <<<"$out"; then
  assert_fail "no systemctl -> still printed a PASS assertion measured with no instrument"
else
  assert_pass "no systemctl -> asserted nothing (no fabricated PASS)"
fi

# ---------------------------------------------------------------------------
# 2. THE REGRESSION THIS GATE EXISTS FOR, part two. systemctl IS present but
#    the host was not booted with systemd. `systemctl is-enabled` answers from
#    unit files on disk, so it returns a state and no error — a challenge that
#    trusts that answer believes it measured a running manager.
# ---------------------------------------------------------------------------
echo "[2/6] systemctl present but host not booted with systemd"
mkdir -p "$WORK/bin_static" && cp -a "$WORK/bin/." "$WORK/bin_static/"
make_stub_systemctl "$WORK/bin_static" static
mkdir -p "$WORK/etc_notbooted"
out="$(run_sut notbooted "$WORK/bin_static" notbooted)"; rc=$?
echo "    exit=$rc"
if [[ "$rc" -eq 2 ]]; then
  assert_pass "systemd not booted -> exit 2"
else
  assert_fail "systemd not booted -> exit $rc, want 2. Output: $(echo "$out" | tail -n2 | tr '\n' ' ')"
fi

# ---------------------------------------------------------------------------
# 3. Everything in place on a live systemd host. The contract must still be
#    able to say 0 — a fix that only ever reports 2 is also a bluff.
# ---------------------------------------------------------------------------
echo "[3/6] systemd live, guard fully in place"
mkdir -p "$WORK/bin_ok" && cp -a "$WORK/bin/." "$WORK/bin_ok/"
make_stub_systemctl "$WORK/bin_ok" masked
make_stub_journalctl "$WORK/bin_ok" "-- No entries --"
seed_dropins "$WORK/etc_allok"
out="$(run_sut allok "$WORK/bin_ok" booted)"; rc=$?
echo "    exit=$rc"
if [[ "$rc" -eq 0 ]]; then
  assert_pass "guard active on a live systemd host -> exit 0"
else
  assert_fail "guard active -> exit $rc, want 0. Output: $(echo "$out" | tail -n3 | tr '\n' ' ')"
fi

# ---------------------------------------------------------------------------
# 4. A live systemd host whose targets are NOT masked is a genuine defect and
#    must stay a 1. This is the assertion that stops the fix from degenerating
#    into "call everything undetermined".
# ---------------------------------------------------------------------------
echo "[4/6] systemd live, sleep targets unmasked (genuine defect)"
mkdir -p "$WORK/bin_broken" && cp -a "$WORK/bin/." "$WORK/bin_broken/"
make_stub_systemctl "$WORK/bin_broken" static
make_stub_journalctl "$WORK/bin_broken" "-- No entries --"
seed_dropins "$WORK/etc_broken"
out="$(run_sut broken "$WORK/bin_broken" booted)"; rc=$?
echo "    exit=$rc"
if [[ "$rc" -eq 1 ]]; then
  assert_pass "unmasked targets on a live host -> exit 1 (genuine defect)"
else
  assert_fail "unmasked targets -> exit $rc, want 1. Output: $(echo "$out" | tail -n3 | tr '\n' ' ')"
fi

# ---------------------------------------------------------------------------
# 5. A live systemd host where the guard was never installed. Determinate:
#    the marker is absent and that is directly actionable. 1, not 2.
# ---------------------------------------------------------------------------
echo "[5/6] systemd live, guard never installed (no drop-ins)"
mkdir -p "$WORK/etc_nomarker"
out="$(run_sut nomarker "$WORK/bin_ok" booted)"; rc=$?
echo "    exit=$rc"
if [[ "$rc" -eq 1 ]]; then
  assert_pass "guard never installed -> exit 1 (determinate: run the installer)"
else
  assert_fail "guard never installed -> exit $rc, want 1. Output: $(echo "$out" | tail -n3 | tr '\n' ' ')"
fi

# ---------------------------------------------------------------------------
# 6. Live systemd, drop-ins present and targets masked, but the journal cannot
#    be read. Three of four assertions pass; the fourth has no instrument. The
#    honest verdict is 2 — reporting 0 would claim a clean suspend history that
#    was never actually looked at.
# ---------------------------------------------------------------------------
echo "[6/6] systemd live, guard in place, journal unreadable"
mkdir -p "$WORK/bin_nojournal" && cp -a "$WORK/bin/." "$WORK/bin_nojournal/"
make_stub_systemctl "$WORK/bin_nojournal" masked
rm -f "$WORK/bin_nojournal/journalctl"
seed_dropins "$WORK/etc_nojournal"
out="$(run_sut nojournal "$WORK/bin_nojournal" booted)"; rc=$?
echo "    exit=$rc"
if [[ "$rc" -eq 2 ]]; then
  assert_pass "journal unreadable -> exit 2 (three passes are still not a pass)"
else
  assert_fail "journal unreadable -> exit $rc, want 2. Output: $(echo "$out" | tail -n3 | tr '\n' ' ')"
fi
if grep -q "UNDETERMINED" <<<"$out"; then
  assert_pass "journal unreadable -> said so explicitly (not silently skipped)"
else
  assert_fail "journal unreadable -> did not report the assertion as undetermined"
fi

echo
echo "=== summary: $PASS_COUNT pass, $FAIL_COUNT fail ==="
[[ $FAIL_COUNT -eq 0 ]] && exit 0 || exit 1
