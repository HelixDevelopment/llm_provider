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
# This challenge drives the challenge-under-test through eight host conditions —
# five of which cannot occur on the machine running it — and asserts its exit
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
# Usage:
#   bash host_no_auto_suspend_contract_challenge.sh
#   bash host_no_auto_suspend_contract_challenge.sh --prove-failure
#
# Exit (three-valued; a FAIL outranks a could-not-determine):
#   0 = all EXPECTED_ASSERTIONS (11) assertions PASS
#   1 = one or more FAIL — including fewer than 11 assertions executed (the
#       executed-count floor: a deleted or skipped case is a finding)
#   2 = could not determine: an unknown argument, the challenge-under-test
#       missing, mktemp failing, no usable user namespace to sandbox with, a
#       case whose sandbox could not be built, or case 8's address-space cap
#       not applicable on this host. NEVER a pass.
#
# --prove-failure (§1.1 paired proof, VSC-370): copies this gate and the
# challenge-under-test into throw-away directories under $TMPDIR, plants one
# contract break per case and runs the copy of this gate against it. Planted in
# the challenge-under-test (each must make this gate exit 1 and name the case):
# no systemctl reported as FAIL, systemctl trusted without the sd_booted probe,
# always 0, always 2, a missing journal counted as a pass, miscounted suspend
# broadcasts, a buffered journal, a fabricated PASS. Planted in this gate's own
# copy: a deleted assertion (floor, exit 1), a sandbox mount that fails (rc 99)
# and a sandbox command that exits before the challenge without rc 99 (only the
# SANDBOX-READY marker catches it) — both exit 2. Plus: unknown argument,
# missing challenge-under-test, no unshare(1), an unshare(1) that cannot build
# a namespace, unusable TMPDIR and an inapplicable case-8 cap (exit 2), and a
# finding under that cap (exit 1). Exits 0 only when every case holds, 1 when
# any does not, 2 when the proof could not judge a case — e.g. on a host whose
# unprivileged user namespaces are restricted (Ubuntu's
# apparmor_restrict_unprivileged_userns),
# where every sandboxed case is reported UNDETERMINED, never passed.
#
# On such a host, every exit 0 of this gate and of its proof has been measured
# under rootless podman's namespace: `podman unshare bash <this file> [...]`.

set -uo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

PROVE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prove-failure) PROVE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "COULD NOT DETERMINE: unknown argument '$1' — this gate takes only --prove-failure." >&2
       exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="${HERE}/$(basename "${BASH_SOURCE[0]}")"
SUT="${HERE}/host_no_auto_suspend_challenge.sh"

# ── paired proof (--prove-failure, §1.1, VSC-370) ───────────────────────────
# Copies THIS gate and the challenge-under-test into throw-away case
# directories under $TMPDIR, plants one contract break per case (in the copy
# of the challenge-under-test, or in the copy of this gate for its own floor
# and sandbox handling) and runs the copy of this gate against it. A case
# holds only when the exit code matches AND the output names the defect.
# Cases that need the user+mount-namespace sandbox are UNDETERMINED (never
# passed) on a host that cannot build one. Exit: 0 every case held, 1 any
# case did not, 2 the proof itself could not judge one or more cases.
PF_T=""
PF_PASS=0
PF_FAIL=0
PF_UNDET=0
pf_cleanup() { [[ -n "$PF_T" && -d "$PF_T" ]] && { chmod -R u+rwx "$PF_T" 2>/dev/null; rm -rf "$PF_T"; }; }
pf_ok()    { PF_PASS=$((PF_PASS + 1)); echo "  PASS $1"; }
pf_bad()   { PF_FAIL=$((PF_FAIL + 1)); echo "  FAIL $1"; }
pf_undet() { PF_UNDET=$((PF_UNDET + 1)); echo "  UNDET $1: $2"; }
# pf_plant FILE FROM TO — literal substitution; FROM must occur exactly once.
pf_plant() {
  local f="$1" from="$2" to="$3" c rest n
  c="$(cat -- "$f" && printf x)" || return 1
  c="${c%x}"
  rest="${c//"$from"/}"
  n=$(( (${#c} - ${#rest}) / ${#from} ))
  if [[ "$n" -ne 1 ]]; then
    echo "  (plant: the text to replace occurs $n times in $(basename "$f"), need exactly 1)"
    return 1
  fi
  printf '%s' "${c/"$from"/"$to"}" > "$f"
}
# pf_expect NAME WANT NEEDLE CMD... — rc must equal WANT and, when NEEDLE is
# non-empty, the combined output must contain it.
pf_expect() {
  local name="$1" want="$2" needle="$3" out rc
  shift 3
  out="$("$@" 2>&1)"; rc=$?
  if [[ "$rc" -eq "$want" ]] && { [[ -z "$needle" ]] || grep -qF -- "$needle" <<<"$out"; }; then
    pf_ok "$name (rc $rc)"
  else
    pf_bad "$name: want rc $want${needle:+ naming \"$needle\"}, got rc $rc"
    printf '%s\n' "$out" | tail -n 6 | sed 's/^/       | /'
  fi
}
prove_failure() {
  shopt -u patsub_replacement 2>/dev/null || true
  local gname sname d sandbox=1 why=""
  if [[ ! -f "$SUT" ]]; then
    echo "prove-failure: cannot determine — challenge under test not found: $SUT"
    return 2
  fi
  PF_T="$(mktemp -d)" || { echo "prove-failure: cannot determine — mktemp -d failed"; return 2; }
  trap pf_cleanup EXIT
  mkdir -p "$PF_T/tmp" || { echo "prove-failure: cannot determine — cannot create $PF_T/tmp"; return 2; }
  gname="$(basename "$SELF")"
  sname="$(basename "$SUT")"
  # pf_case NAME — a fresh directory holding pristine copies of gate + SUT.
  pf_case() {
    local c="$PF_T/$1"
    mkdir -p "$c" && cp -- "$SELF" "$c/$gname" && cp -- "$SUT" "$c/$sname" && printf '%s\n' "$c"
  }
  pf_gate() { local c="$1"; shift; env TMPDIR="$PF_T/tmp" bash "$c/$gname" "$@"; }
  if ! command -v unshare >/dev/null 2>&1; then sandbox=0; why="unshare(1) not available"
  elif ! unshare -rm true >/dev/null 2>&1; then sandbox=0; why="unprivileged user+mount namespaces unavailable on this host"
  fi
  echo "prove-failure: contract gate $SELF"
  echo "               challenge under test $SUT"

  # --- cases that need no sandbox ------------------------------------------
  d="$(pf_case u1)" && pf_expect "U1 unknown argument is rc 2" 2 "unknown argument" pf_gate "$d" --no-such-flag
  if d="$(pf_case u2)" && rm -f -- "$d/$sname"; then
    pf_expect "U2 challenge under test missing" 2 "challenge under test not found" pf_gate "$d"
  else pf_undet "U2" "could not build the case"; fi
  pf_mkpath() { # DIR TOOL... — a PATH directory holding only the named tools
    local p="$1" b src
    shift
    mkdir -p "$p" || return 1
    for b in "$@"; do src="$(command -v "$b")" && ln -s "$src" "$p/$b" || return 1; done
  }
  if d="$(pf_case u3)" && pf_mkpath "$d/path" dirname basename mktemp rm chmod; then
    pf_expect "U3 no unshare(1) on PATH" 2 "unshare(1) not available" env PATH="$d/path" TMPDIR="$PF_T/tmp" "$BASH" "$d/$gname"
  else pf_undet "U3" "could not build a PATH without unshare"; fi
  d="$(pf_case u4)" && pf_expect "U4 TMPDIR unusable (mktemp fails)" 2 "mktemp" env TMPDIR="$PF_T/no-such-dir" bash "$d/$gname"
  # U6: an unshare(1) that exists but cannot build a namespace (a stub that
  # fails) must stop the gate at its up-front probe, before any case runs.
  if d="$(pf_case u6)" && mkdir -p "$d/stub" && printf '#!%s\nexit 1\n' "$BASH" > "$d/stub/unshare" && chmod 755 "$d/stub/unshare"; then
    pf_expect "U6 unshare present but failing -rm true" 2 "namespaces are unavailable" env PATH="$d/stub:$PATH" TMPDIR="$PF_T/tmp" "$BASH" "$d/$gname"
  else pf_undet "U6" "could not build the stub"; fi

  # --- cases that drive the sandboxed challenge -----------------------------
  local sb_cases=(C0 K1 K2 K3 K4 K5 K6 K7 K8 G1 G2 G3 U5 F1)
  if [[ "$sandbox" -eq 0 ]]; then
    local k
    for k in "${sb_cases[@]}"; do pf_undet "$k" "$why — the case cannot be run here"; done
  else
    d="$(pf_case c0)" && pf_expect "C0 clean control: all 11 assertions pass" 0 "11 pass, 0 fail" pf_gate "$d"
    # K1..K8: one planted contract break in the copy of the challenge-under-test.
    local kname kfrom kto kneedle i
    local -a K_NAME K_FROM K_TO K_NEEDLE
    K_NAME+=("K1 no systemctl reported as FAIL (the old two-valued contract)")
    K_FROM+=('if ! command -v systemctl >/dev/null 2>&1; then')
    K_TO+=('if ! command -v systemctl >/dev/null 2>&1; then exit 1; fi; if false; then')
    K_NEEDLE+=("no systemctl -> exit 1, want 2")
    K_NAME+=("K2 systemctl trusted without the sd_booted probe")
    K_FROM+=('if [[ ! -d /run/systemd/system ]]; then')
    K_TO+=('if false; then')
    K_NEEDLE+=("systemd not booted -> exit 1, want 2")
    K_NAME+=("K3 challenge always exits 0")
    K_FROM+=('PASS_COUNT=0')
    K_TO+=('exit 0')
    K_NEEDLE+=("no systemctl -> exit 0, want 2")
    K_NAME+=("K4 challenge always exits 2 (everything undetermined)")
    K_FROM+=('PASS_COUNT=0')
    K_TO+=('exit 2')
    K_NEEDLE+=("guard active -> exit 2, want 0")
    K_NAME+=("K5 a missing journal counted as a pass")
    K_FROM+=('  assert_undet "journalctl is not on PATH — cannot read the journal, so past" \')
    K_TO+=('  assert_pass "journalctl is not on PATH — cannot read the journal, so past" \')
    K_NEEDLE+=("journal unreadable -> exit 0, want 2")
    K_NAME+=("K6 suspend broadcasts miscounted")
    K_FROM+=('grep -c "The system will suspend now"')
    K_TO+=('grep -c "The system will hibernate now"')
    K_NEEDLE+=("suspend event in the journal -> exit 0, want 1")
    K_NAME+=("K7 journal buffered instead of streamed")
    K_FROM+=('  journal_rc=${PIPESTATUS[0]}')
    K_TO+=('  journal_rc=${PIPESTATUS[0]}; buffered=$(journalctl --since "$fix_iso" --no-pager 2>/dev/null); journal_rc=$?; grep -c "The system will suspend now" <<<"$buffered" > "$scratch"')
    K_NEEDLE+=("FAIL: large journal")
    K_NAME+=("K8 a PASS fabricated with no instrument")
    K_FROM+=('if ! command -v systemctl >/dev/null 2>&1; then')
    K_TO+=('echo "PASS: fabricated"; if ! command -v systemctl >/dev/null 2>&1; then')
    K_NEEDLE+=("still printed a PASS assertion")
    for i in "${!K_NAME[@]}"; do
      kname="${K_NAME[$i]}"; kfrom="${K_FROM[$i]}"; kto="${K_TO[$i]}"; kneedle="${K_NEEDLE[$i]}"
      if d="$(pf_case "k$i")" && pf_plant "$d/$sname" "$kfrom" "$kto"; then
        pf_expect "$kname" 1 "$kneedle" pf_gate "$d"
      else pf_undet "${kname%% *}" "could not plant the defect"; fi
    done
    # G1/G2: the gate's own floor and sandbox handling, planted in its copy.
    # The planted text is assembled from two halves so that it occurs only once
    # in this file (in the assertion), never in the proof arm itself.
    local g_from
    g_from='assert_pass "systemd not booted'; g_from+=' -> exit 2"'
    if d="$(pf_case g1)" && pf_plant "$d/$gname" "$g_from" ':'; then
      pf_expect "G1 a deleted assertion trips the executed-count floor" 1 "floor" pf_gate "$d"
    else pf_undet "G1" "could not plant the defect"; fi
    g_from='/etc/systemd'; g_from+=' || exit 99'
    if d="$(pf_case g2)" && pf_plant "$d/$gname" "$g_from" '/nonexistent-vsc370 || exit 99'; then
      pf_expect "G2 a sandbox that cannot be built is rc 2, not a FAIL" 2 "sandbox" pf_gate "$d"
    else pf_undet "G2" "could not plant the defect"; fi
    # G3: the sandbox command itself fails without reaching the challenge and
    # without rc 99 — only the SANDBOX-READY marker can tell that from a verdict.
    g_from='unshare -rm'; g_from+=" bash -c '"
    if d="$(pf_case g3)" && pf_plant "$d/$gname" "$g_from" "unshare -rm sh -c 'exit 7;"; then
      pf_expect "G3 a sandbox that exits before the challenge (rc 7, no marker) is rc 2" 2 "sandbox" pf_gate "$d"
    else pf_undet "G3" "could not plant the defect"; fi
    # U5: an address-space limit (soft+hard) below case 8's cap makes case 8 unjudgeable.
    d="$(pf_case u5)" && pf_expect "U5 case-8 cap not applicable is rc 2, not a FAIL" 2 "ulimit" \
      bash -c 'ulimit -v 150000 && TMPDIR="$1" exec bash "$2"' _ "$PF_T/tmp" "$d/$gname"
    # F1: a finding outranks an undetermined case.
    if d="$(pf_case f1)" && pf_plant "$d/$sname" 'PASS_COUNT=0' 'exit 0'; then
      pf_expect "F1 a finding outranks an undetermined case" 1 "no systemctl -> exit 0, want 2" \
        bash -c 'ulimit -v 150000 && TMPDIR="$1" exec bash "$2"' _ "$PF_T/tmp" "$d/$gname"
    else pf_undet "F1" "could not plant the defect"; fi
  fi
  echo "prove-failure: $PF_PASS passed, $PF_FAIL failed, $PF_UNDET undetermined"
  [[ "$PF_FAIL" -gt 0 ]] && return 1
  [[ "$PF_UNDET" -gt 0 ]] && return 2
  return 0
}

if [[ "$PROVE" -eq 1 ]]; then
  prove_failure
  exit $?
fi

# Eight cases make exactly this many assertions. Fewer executed is rc 1 even
# when none failed: a case that was deleted or skipped verified nothing.
EXPECTED_ASSERTIONS=11
# Printed inside the sandbox just before the challenge-under-test starts; a run
# whose output lacks it never reached the challenge (unshare or a mount failed).
SANDBOX_MARK="SANDBOX-READY vsc370"

PASS_COUNT=0
FAIL_COUNT=0
UNDET_COUNT=0
assert_pass()  { echo "PASS: $*"; PASS_COUNT=$((PASS_COUNT + 1)); }
assert_fail()  { echo "FAIL: $*" >&2; FAIL_COUNT=$((FAIL_COUNT + 1)); }
assert_undet() { echo "UNDETERMINED: $*" >&2; UNDET_COUNT=$((UNDET_COUNT + 1)); }

echo "=== host_no_auto_suspend_contract_challenge ==="
echo

if [[ ! -f "$SUT" ]]; then
  echo "COULD NOT DETERMINE: challenge under test not found: $SUT" >&2
  exit 2
fi

# Checked before anything is built: an empty WORK would turn every path below
# into one under / (mkdir /bin_static, ln -sf ... /bin/bash, ...).
if ! WORK="$(mktemp -d)" || [[ -z "$WORK" || ! -d "$WORK" ]]; then
  echo "COULD NOT DETERMINE: mktemp -d failed (TMPDIR='${TMPDIR:-}') — nowhere to build the sandbox." >&2
  exit 2
fi
trap 'rm -rf "$WORK"' EXIT

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
# $4 = optional `ulimit -v` cap in KiB, applied to the challenge only
run_sut() {
  local label="$1" bindir="$2" booted="$3" vcap="${4:-}"
  local fake="$WORK/etc_$label"
  unshare -rm bash -c '
    mount --make-rprivate / 2>/dev/null
    mount -t tmpfs tmpfs /run/systemd || exit 99
    if [ "'"$booted"'" = "booted" ]; then mkdir -p /run/systemd/system; fi
    mount --bind "'"$fake"'" /etc/systemd || exit 99
    export PATH="'"$bindir"'"
    if [ -n "'"$vcap"'" ]; then ulimit -v '"${vcap:-0}"' || exit 99; fi
    echo "'"$SANDBOX_MARK"'"
    bash "'"$SUT"'"
  ' 2>&1
}

# sandbox_ok RC OUT N — true when the challenge-under-test actually ran. When
# the sandbox could not be built (unshare failed, a mount failed, the cap could
# not be applied: rc 99 or no SANDBOX_MARK), the case's N assertions are
# UNDETERMINED — that exit code says nothing about the challenge-under-test.
sandbox_ok() {
  local rc="$1" out="$2" n="$3" i
  if [[ "$rc" -ne 99 ]] && grep -qxF "$SANDBOX_MARK" <<<"$out"; then
    return 0
  fi
  for ((i = 0; i < n; i++)); do
    assert_undet "sandbox could not be built (rc $rc) — case not judged. $(echo "$out" | tail -n1)"
  done
  return 1
}

# ---------------------------------------------------------------------------
# 1. THE REGRESSION THIS GATE EXISTS FOR, part one. No systemctl anywhere:
#    the guard is INAPPLICABLE, not broken. Before the fix this returned 1.
# ---------------------------------------------------------------------------
echo "[1/8] host without systemd (no systemctl on PATH)"
mkdir -p "$WORK/etc_nosystemctl"
out="$(run_sut nosystemctl "$WORK/bin" notbooted)"; rc=$?
echo "    exit=$rc"
if sandbox_ok "$rc" "$out" 2; then
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
fi

# ---------------------------------------------------------------------------
# 2. THE REGRESSION THIS GATE EXISTS FOR, part two. systemctl IS present but
#    the host was not booted with systemd. `systemctl is-enabled` answers from
#    unit files on disk, so it returns a state and no error — a challenge that
#    trusts that answer believes it measured a running manager.
# ---------------------------------------------------------------------------
echo "[2/8] systemctl present but host not booted with systemd"
mkdir -p "$WORK/bin_static" && cp -a "$WORK/bin/." "$WORK/bin_static/"
make_stub_systemctl "$WORK/bin_static" static
mkdir -p "$WORK/etc_notbooted"
out="$(run_sut notbooted "$WORK/bin_static" notbooted)"; rc=$?
echo "    exit=$rc"
if sandbox_ok "$rc" "$out" 1; then
  if [[ "$rc" -eq 2 ]]; then
    assert_pass "systemd not booted -> exit 2"
  else
    assert_fail "systemd not booted -> exit $rc, want 2. Output: $(echo "$out" | tail -n2 | tr '\n' ' ')"
  fi
fi

# ---------------------------------------------------------------------------
# 3. Everything in place on a live systemd host. The contract must still be
#    able to say 0 — a fix that only ever reports 2 is also a bluff.
# ---------------------------------------------------------------------------
echo "[3/8] systemd live, guard fully in place"
mkdir -p "$WORK/bin_ok" && cp -a "$WORK/bin/." "$WORK/bin_ok/"
make_stub_systemctl "$WORK/bin_ok" masked
make_stub_journalctl "$WORK/bin_ok" "-- No entries --"
seed_dropins "$WORK/etc_allok"
out="$(run_sut allok "$WORK/bin_ok" booted)"; rc=$?
echo "    exit=$rc"
if sandbox_ok "$rc" "$out" 1; then
  if [[ "$rc" -eq 0 ]]; then
    assert_pass "guard active on a live systemd host -> exit 0"
  else
    assert_fail "guard active -> exit $rc, want 0. Output: $(echo "$out" | tail -n3 | tr '\n' ' ')"
  fi
fi

# ---------------------------------------------------------------------------
# 4. A live systemd host whose targets are NOT masked is a genuine defect and
#    must stay a 1. This is the assertion that stops the fix from degenerating
#    into "call everything undetermined".
# ---------------------------------------------------------------------------
echo "[4/8] systemd live, sleep targets unmasked (genuine defect)"
mkdir -p "$WORK/bin_broken" && cp -a "$WORK/bin/." "$WORK/bin_broken/"
make_stub_systemctl "$WORK/bin_broken" static
make_stub_journalctl "$WORK/bin_broken" "-- No entries --"
seed_dropins "$WORK/etc_broken"
out="$(run_sut broken "$WORK/bin_broken" booted)"; rc=$?
echo "    exit=$rc"
if sandbox_ok "$rc" "$out" 1; then
  if [[ "$rc" -eq 1 ]]; then
    assert_pass "unmasked targets on a live host -> exit 1 (genuine defect)"
  else
    assert_fail "unmasked targets -> exit $rc, want 1. Output: $(echo "$out" | tail -n3 | tr '\n' ' ')"
  fi
fi

# ---------------------------------------------------------------------------
# 5. A live systemd host where the guard was never installed. Determinate:
#    the marker is absent and that is directly actionable. 1, not 2.
# ---------------------------------------------------------------------------
echo "[5/8] systemd live, guard never installed (no drop-ins)"
mkdir -p "$WORK/etc_nomarker"
out="$(run_sut nomarker "$WORK/bin_ok" booted)"; rc=$?
echo "    exit=$rc"
if sandbox_ok "$rc" "$out" 1; then
  if [[ "$rc" -eq 1 ]]; then
    assert_pass "guard never installed -> exit 1 (determinate: run the installer)"
  else
    assert_fail "guard never installed -> exit $rc, want 1. Output: $(echo "$out" | tail -n3 | tr '\n' ' ')"
  fi
fi

# ---------------------------------------------------------------------------
# 6. Live systemd, drop-ins present and targets masked, but the journal cannot
#    be read. Three of four assertions pass; the fourth has no instrument. The
#    honest verdict is 2 — reporting 0 would claim a clean suspend history that
#    was never actually looked at.
# ---------------------------------------------------------------------------
echo "[6/8] systemd live, guard in place, journal unreadable"
mkdir -p "$WORK/bin_nojournal" && cp -a "$WORK/bin/." "$WORK/bin_nojournal/"
make_stub_systemctl "$WORK/bin_nojournal" masked
rm -f "$WORK/bin_nojournal/journalctl"
seed_dropins "$WORK/etc_nojournal"
out="$(run_sut nojournal "$WORK/bin_nojournal" booted)"; rc=$?
echo "    exit=$rc"
if sandbox_ok "$rc" "$out" 2; then
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
fi

# ---------------------------------------------------------------------------
# 7. The journal actually CONTAINS a suspend broadcast. Everything else is in
#    place, so this isolates the counting itself: a challenge that reads the
#    journal but miscounts would pass cases 1-6 unchanged and still miss the
#    one event the whole assertion exists to find.
# ---------------------------------------------------------------------------
echo "[7/8] systemd live, guard in place, but the journal shows a suspend"
mkdir -p "$WORK/bin_suspended" && cp -a "$WORK/bin/." "$WORK/bin_suspended/"
make_stub_systemctl "$WORK/bin_suspended" masked
make_stub_journalctl "$WORK/bin_suspended" \
  "May 01 03:14:07 host systemd-logind[1183]: The system will suspend now!"
seed_dropins "$WORK/etc_suspended"
out="$(run_sut suspended "$WORK/bin_suspended" booted)"; rc=$?
echo "    exit=$rc"
if sandbox_ok "$rc" "$out" 1; then
  if [[ "$rc" -eq 1 ]]; then
    assert_pass "suspend event in the journal -> exit 1 (masking didn't take)"
  else
    assert_fail "suspend event in the journal -> exit $rc, want 1. Output: $(echo "$out" | tail -n3 | tr '\n' ' ')"
  fi
fi

# ---------------------------------------------------------------------------
# 8. THE REGRESSION THIS CASE EXISTS FOR. The journal window is open-ended — it
#    starts when the guard was installed, which can be months back — so the
#    journal must be STREAMED, never buffered. A first draft of this very fix
#    captured it with `$(journalctl ...)`; measured on the development host that
#    window was already past 190 MB after 100 seconds and still going, and the
#    challenge died with rc=139 (SIGSEGV) on a run that had previously exited 0.
#
#    This case does NOT try to reproduce that by brute-forcing a journal big
#    enough to exhaust the machine — that would be slow, unbounded, and unkind
#    to whatever else the host is running. It tests the PROPERTY instead: a
#    streaming implementation uses memory independent of journal size, so it
#    survives inside a small address space, while a buffering one needs memory
#    proportional to the journal and cannot.
#
#    Calibrated on this host: with `ulimit -v 200000` (~200 MB) and a 64 MB
#    journal, streaming exits 0 while buffering dies with
#    "xrealloc: cannot allocate ... bytes" and rc=2.
# ---------------------------------------------------------------------------
echo "[8/8] systemd live, large journal in a small address space (must stream)"
VCAP_KIB=200000
JOURNAL_BYTES=67108864
mkdir -p "$WORK/bin_bigjournal" && cp -a "$WORK/bin/." "$WORK/bin_bigjournal/"
make_stub_systemctl "$WORK/bin_bigjournal" masked
for b in yes head; do
  p="$(command -v "$b" 2>/dev/null)" && ln -sf "$p" "$WORK/bin_bigjournal/$b"
done
cat > "$WORK/bin_bigjournal/journalctl" <<STUB
#!/usr/bin/env bash
# A journal-shaped stream of ${JOURNAL_BYTES} bytes, none of it matching.
yes "Sep 01 12:00:00 host kernel: an ordinary uninteresting log line for padding" \
  | head -c ${JOURNAL_BYTES}
exit 0
STUB
chmod +x "$WORK/bin_bigjournal/journalctl"
seed_dropins "$WORK/etc_bigjournal"

# Sanity: the cap must actually be applicable here, or this case proves nothing.
# An inapplicable cap is could-not-determine, not a defect of the challenge.
if ! ( ulimit -v "$VCAP_KIB" && true ) 2>/dev/null; then
  assert_undet "cannot apply 'ulimit -v $VCAP_KIB' on this host — case 8 (streaming) not judged"
  assert_undet "cannot apply 'ulimit -v $VCAP_KIB' on this host — case 8 (allocation) not judged"
else
  out="$(run_sut bigjournal "$WORK/bin_bigjournal" booted "$VCAP_KIB")"; rc=$?
  echo "    exit=$rc  (journal $((JOURNAL_BYTES / 1048576)) MB, ulimit -v ${VCAP_KIB} KiB)"
  if sandbox_ok "$rc" "$out" 2; then
    if [[ "$rc" -eq 0 ]]; then
      assert_pass "large journal in a small address space -> exit 0 (streamed)"
    else
      assert_fail "large journal -> exit $rc, want 0 — the journal is being buffered, not streamed. Output: $(echo "$out" | tail -n2 | tr '\n' ' ')"
    fi
    if grep -qE 'xrealloc|cannot allocate|Killed' <<<"$out"; then
      assert_fail "large journal -> allocation failure in the challenge: $(grep -oE '(xrealloc|cannot allocate)[^\n]*' <<<"$out" | head -1)"
    else
      assert_pass "large journal -> no allocation failure (memory use is independent of journal size)"
    fi
  fi
fi

# --- executed-count floor -------------------------------------------------
judged=$((PASS_COUNT + FAIL_COUNT + UNDET_COUNT))
if [[ "$judged" -lt "$EXPECTED_ASSERTIONS" ]]; then
  assert_fail "executed $judged of $EXPECTED_ASSERTIONS assertions — a case was skipped or deleted (executed-count floor)"
fi

echo
echo "=== summary: $PASS_COUNT pass, $FAIL_COUNT fail, $UNDET_COUNT undetermined ==="
if [[ $FAIL_COUNT -gt 0 ]]; then
  exit 1
fi
if [[ $UNDET_COUNT -gt 0 ]]; then
  echo "COULD NOT DETERMINE: $UNDET_COUNT assertion(s) could not be judged — THIS IS NOT A PASS." >&2
  exit 2
fi
exit 0
