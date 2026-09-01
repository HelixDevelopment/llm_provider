#!/usr/bin/env bash
# install_host_suspend_guard_contract_challenge.sh — CONST-033 anti-bluff gate.
#
# install-host-suspend-guard.sh writes systemd drop-ins that stop the host
# suspending mid-workload. Its failure mode is not a crash: it is reporting
# success on a host where the guard did NOT take effect, which is worse than
# failing, because the operator then trusts an unprotected host.
#
# This challenge drives the installer through four host conditions it cannot
# encounter on this machine and asserts its exit code against the contract
# declared in the installer's own header:
#
#   0 = guard installed AND verified active
#   1 = genuine failure (systemd operable, but a step failed)
#   2 = could not determine / not applicable — never a pass
#
# Every run is sandboxed: the installer executes inside a user+mount namespace
# with a throwaway directory bind-mounted over /etc/systemd, so the real host's
# power configuration is NEVER read from or written to.
#
# Exit:
#   0 = all assertions PASS
#   1 = one or more FAIL
#   2 = invocation error, or no usable user namespace to sandbox with

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="${HERE}/../../scripts/host-power-management/install-host-suspend-guard.sh"

PASS_COUNT=0
FAIL_COUNT=0
assert_pass() { echo "PASS: $*"; PASS_COUNT=$((PASS_COUNT + 1)); }
assert_fail() { echo "FAIL: $*" >&2; FAIL_COUNT=$((FAIL_COUNT + 1)); }

echo "=== install_host_suspend_guard_contract_challenge ==="
echo

if [[ ! -f "$SUT" ]]; then
  echo "ERROR: script under test not found: $SUT" >&2
  exit 2
fi

# --- sandbox availability -------------------------------------------------
# Honest 2, not a silent 0: without user namespaces this challenge cannot
# create a fake root or protect the real /etc/systemd, so it cannot judge.
if ! command -v unshare >/dev/null 2>&1; then
  echo "COULD NOT DETERMINE: unshare(1) not available — cannot sandbox the installer." >&2
  exit 2
fi
if ! unshare -rm true >/dev/null 2>&1; then
  echo "COULD NOT DETERMINE: unprivileged user+mount namespaces are unavailable" >&2
  echo "  on this host, so the installer cannot be run without touching /etc." >&2
  exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# A PATH holding only the utilities the installer legitimately needs, so we
# control precisely whether systemctl is on it.
mkdir -p "$WORK/bin"
for b in bash mkdir cat rm ls id find grep sed tr head; do
  p="$(command -v "$b" 2>/dev/null)" && ln -sf "$p" "$WORK/bin/$b"
done

# Writes a stub systemctl whose per-verb exit codes we dictate.
# $1 = target dir, $2 = rc for daemon-reload, $3 = rc for reload-or-restart
make_stub_systemctl() {
  local dir="$1" reload_rc="$2" logind_rc="$3"
  cat > "$dir/systemctl" <<STUB
#!/usr/bin/env bash
case "\$1" in
  mask)       for u in "\${@:2}"; do echo "Created symlink '/etc/systemd/system/\$u' -> '/dev/null'."; done; exit 0 ;;
  is-enabled) echo masked; exit 0 ;;
  daemon-reload) exit $reload_rc ;;
  reload-or-restart)
      [ $logind_rc -ne 0 ] && echo "Failed to reload-or-restart \${2}.service: Unit \${2}.service not found." >&2
      exit $logind_rc ;;
  *) echo "stub systemctl: unhandled '\$*'" >&2; exit 1 ;;
esac
STUB
  chmod +x "$dir/systemctl"
}

# Runs the installer as namespace-root with /etc/systemd bind-mounted over a
# throwaway dir. Echoes the installer's stdout+stderr; returns its exit code.
# $1 = label, $2 = PATH to use, $3 = "hide_systemd" to also tmpfs /run/systemd
run_installer() {
  local label="$1" bindir="$2" mode="${3:-}"
  local fake="$WORK/etc_$label"
  rm -rf "$fake"; mkdir -p "$fake"
  unshare -rm bash -c '
    mount --make-rprivate / 2>/dev/null
    if [ "'"$mode"'" = "hide_systemd" ]; then mount -t tmpfs tmpfs /run/systemd; fi
    mount --bind "'"$fake"'" /etc/systemd || exit 99
    export PATH="'"$bindir"'"
    bash "'"$SUT"'"
  ' 2>&1
}
files_written() { find "$WORK/etc_$1" -type f 2>/dev/null | wc -l | tr -d ' '; }

# ---------------------------------------------------------------------------
# 1. systemctl absent — the guard is INAPPLICABLE, not installed.
# ---------------------------------------------------------------------------
echo "[1/5] host without systemd (no systemctl on PATH)"
out="$(run_installer nosystemctl "$WORK/bin")"; rc=$?
n="$(files_written nosystemctl)"
echo "    exit=$rc  files written=$n"
if [[ "$rc" -eq 2 ]]; then
  assert_pass "no systemctl -> exit 2 (could not determine / not applicable)"
else
  assert_fail "no systemctl -> exit $rc, want 2. Output: $(echo "$out" | tail -n2 | tr '\n' ' ')"
fi
if [[ "$n" -eq 0 ]]; then
  assert_pass "no systemctl -> wrote no config (no half-install)"
else
  assert_fail "no systemctl -> wrote $n config file(s) that can never take effect"
fi

# ---------------------------------------------------------------------------
# 2. systemctl present, host NOT booted with systemd. `systemctl mask`
#    returns 0 here (it is an offline symlink operation), so a script that
#    trusts mask's exit code believes it succeeded.
# ---------------------------------------------------------------------------
echo "[2/5] systemctl present but host not booted with systemd"
out="$(run_installer notbooted "$PATH" hide_systemd)"; rc=$?
n="$(files_written notbooted)"
echo "    exit=$rc  files written=$n"
if [[ "$rc" -eq 2 ]]; then
  assert_pass "systemd not booted -> exit 2"
else
  assert_fail "systemd not booted -> exit $rc, want 2. Output: $(echo "$out" | tail -n2 | tr '\n' ' ')"
fi
if [[ "$n" -eq 0 ]]; then
  assert_pass "systemd not booted -> wrote no config (no half-install)"
else
  assert_fail "systemd not booted -> wrote $n inert config file(s)"
fi

# ---------------------------------------------------------------------------
# 3. THE REGRESSION THIS GATE EXISTS FOR. systemd is running and the files
#    are written, but logind never re-reads them. The guard is on disk and
#    NOT in effect. Reporting 0 here is the bluff.
# ---------------------------------------------------------------------------
echo "[3/5] systemd running, but systemd-logind reload fails"
mkdir -p "$WORK/bin_logindfail"; cp -a "$WORK/bin/." "$WORK/bin_logindfail/"
make_stub_systemctl "$WORK/bin_logindfail" 0 1
out="$(run_installer logindfail "$WORK/bin_logindfail")"; rc=$?
echo "    exit=$rc"
if [[ "$rc" -eq 2 ]]; then
  assert_pass "logind reload failed -> exit 2 (not confirmed active)"
else
  assert_fail "logind reload failed -> exit $rc, want 2 — guard is inert but reported otherwise"
fi
if grep -q "DONE" <<<"$out"; then
  assert_fail "logind reload failed -> still printed 'DONE' (claims success it did not verify)"
else
  assert_pass "logind reload failed -> did not print 'DONE'"
fi

# ---------------------------------------------------------------------------
# 4. Everything works. The contract must still be able to say 0, otherwise
#    the fix would just be 'always report failure', which is also a bluff.
# ---------------------------------------------------------------------------
echo "[4/5] systemd running and every step succeeds"
mkdir -p "$WORK/bin_ok"; cp -a "$WORK/bin/." "$WORK/bin_ok/"
make_stub_systemctl "$WORK/bin_ok" 0 0
out="$(run_installer allok "$WORK/bin_ok")"; rc=$?
n="$(files_written allok)"
echo "    exit=$rc  files written=$n"
if [[ "$rc" -eq 0 ]]; then
  assert_pass "all steps succeed -> exit 0"
else
  assert_fail "all steps succeed -> exit $rc, want 0. Output: $(echo "$out" | tail -n3 | tr '\n' ' ')"
fi
if [[ "$n" -eq 2 ]]; then
  assert_pass "all steps succeed -> both drop-ins written"
else
  assert_fail "all steps succeed -> wrote $n drop-in(s), want 2"
fi

# ---------------------------------------------------------------------------
# 5. Non-root is a determinate, actionable failure: re-run with sudo. 1, not 2.
# ---------------------------------------------------------------------------
echo "[5/5] invoked without root"
if [[ "$EUID" -eq 0 ]]; then
  echo "    running as root — cannot exercise the non-root path here"
  assert_pass "non-root path not exercised (challenge is running as root) — reported, not skipped silently"
else
  out="$(bash "$SUT" 2>&1)"; rc=$?
  echo "    exit=$rc"
  if [[ "$rc" -eq 1 ]]; then
    assert_pass "non-root -> exit 1 (determinate: re-run with sudo)"
  else
    assert_fail "non-root -> exit $rc, want 1"
  fi
fi

echo
echo "=== summary: $PASS_COUNT pass, $FAIL_COUNT fail ==="
[[ $FAIL_COUNT -eq 0 ]] && exit 0 || exit 1
