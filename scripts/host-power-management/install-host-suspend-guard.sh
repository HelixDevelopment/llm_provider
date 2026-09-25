#!/usr/bin/env bash
# install-host-suspend-guard.sh
#
# MANUAL PREREQUISITE — run ONCE per host, with sudo, BEFORE running
# any project test / challenge / build that boots containers or
# spawns long-running CLI agents.
#
# Background (CONST-032 / CONST-033): on 2026-04-26 18:23:43 the host
# suspended mid-session, killing the parent project + 41 services + the user's
# SSH session. journalctl showed:
#   systemd-logind[1183]: The system will suspend now!
# Root cause: the GDM greeter session at the local console has its own
# power policy; SSH sessions don't count as activity. User-level
# `sleep-inactive-ac-type=nothing` is necessary but not sufficient.
#
# This script applies defence in depth so neither the greeter, nor any
# DE, nor any user with logind privileges, can suspend the host while
# it's running mission-critical workloads.
#
# Verification (re-run the challenge after this script):
#   bash challenges/scripts/host_no_auto_suspend_challenge.sh
# All 4 assertions must PASS.
#
# Exit:
#   0 = guard installed AND verified active on this host
#   1 = genuine failure — systemd is operable here, but a step failed
#       (not root, mask refused, /etc not writable)
#   2 = COULD NOT DETERMINE / not applicable — this host is not running
#       systemd, or the drop-ins were written but systemd did not confirm
#       it re-read them. 2 is NEVER a pass: after a 2 the host is not
#       known to be protected.
#
# Why the distinction is drawn where it is: writing a drop-in file is not
# the same as the drop-in taking effect, and this script's whole purpose is
# the effect. Every step below therefore either proves its result or
# downgrades the verdict — nothing is assumed from a command's silence.
#
# Contract guarded by:
#   bash challenges/scripts/install_host_suspend_guard_contract_challenge.sh

set -uo pipefail

err() { printf '%s\n' "$*" >&2; }

if [[ "$EUID" -ne 0 ]]; then
    err "ERROR: must be run as root (sudo)."
    # 1, not 2: this is fully determined and directly actionable —
    # nothing about the host is unknown, the invocation was simply wrong.
    exit 1
fi

# --- capability detection -------------------------------------------------
# A host with no systemd has not FAILED to install this guard, and it has not
# succeeded either. The guard is a set of systemd drop-ins; on such a host
# there is nothing here to configure (so 1 would be wrong), yet the host can
# still suspend by other means this script does not address — elogind, acpid,
# a desktop environment's own policy (so 0 would be a lie). The honest verdict
# is 2, and 2 is never a pass.
if ! command -v systemctl >/dev/null 2>&1; then
    err "COULD NOT DETERMINE: systemctl is not on PATH."
    err "  This host does not appear to run systemd, so the systemd drop-ins"
    err "  this script installs cannot apply to it. NOTHING has been written."
    err "  The host is NOT known to be protected against suspend — configure"
    err "  whatever power manager it does use (elogind / acpid / DE policy)."
    exit 2
fi

# systemctl can be present on a host that is not BOOTED with systemd, such as
# a container or a chroot. `systemctl mask` returns 0 there, because masking is
# an offline symlink operation that needs no running manager — so neither the
# binary's presence nor a 0 from mask is evidence that anything took effect.
# /run/systemd/system is the canonical sd_booted(3) probe for a live manager,
# and unlike `systemctl is-system-running` it does not report failure merely
# because the system is `degraded`.
if [[ ! -d /run/systemd/system ]]; then
    err "COULD NOT DETERMINE: systemctl is present, but this host is not booted"
    err "  with systemd (/run/systemd/system does not exist). Masking would"
    err "  write symlinks that no running manager would ever read."
    err "  NOTHING has been written. The host is NOT known to be protected."
    exit 2
fi

echo "[1/4] Masking sleep / suspend / hibernate / hybrid-sleep targets..."
if ! systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target; then
    err "FAILED: could not mask the sleep targets. The host is unprotected."
    exit 1
fi

echo "[2/4] Setting AllowSuspend=no in /etc/systemd/sleep.conf.d/..."
if ! mkdir -p /etc/systemd/sleep.conf.d; then
    err "FAILED: could not create /etc/systemd/sleep.conf.d."
    exit 1
fi
if ! cat > /etc/systemd/sleep.conf.d/00-no-suspend.conf <<'EOF'
# CONST-033: host runs mission-critical parallel CLI-agent + container
# workloads; auto-suspend is unsafe. Defence in depth — see also the
# masked targets above and the logind drop-in below.
[Sleep]
AllowSuspend=no
AllowHibernation=no
AllowSuspendThenHibernate=no
AllowHybridSleep=no
EOF
then
    err "FAILED: could not write /etc/systemd/sleep.conf.d/00-no-suspend.conf."
    exit 1
fi

echo "[3/4] Setting logind IdleAction=ignore + HandleLidSwitch=ignore..."
if ! mkdir -p /etc/systemd/logind.conf.d; then
    err "FAILED: could not create /etc/systemd/logind.conf.d."
    exit 1
fi
if ! cat > /etc/systemd/logind.conf.d/00-no-idle-suspend.conf <<'EOF'
# CONST-033: do not suspend the host on idle (SSH sessions don't count
# as activity; the GDM greeter's idle policy was the historical
# trigger).
[Login]
IdleAction=ignore
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
EOF
then
    err "FAILED: could not write /etc/systemd/logind.conf.d/00-no-idle-suspend.conf."
    exit 1
fi

echo "[4/4] Reloading systemd and verifying the guard is active..."

# These two reloads are what turn the files above into policy. A failure here
# used to be swallowed by `|| true`, which left the host unprotected while the
# script printed DONE and exited 0. It now downgrades the verdict to 2.
undetermined=0
if ! systemctl daemon-reload; then
    err "WARNING: 'systemctl daemon-reload' failed."
    undetermined=1
fi
if ! systemctl reload-or-restart systemd-logind; then
    err "WARNING: could not reload systemd-logind — the logind drop-in is on"
    err "  disk but logind has NOT re-read it, so IdleAction/HandleLidSwitch"
    err "  are not yet in force."
    undetermined=1
fi

# Verify the masking actually took, rather than trusting mask's exit code.
unmasked=()
for tgt in sleep.target suspend.target hibernate.target hybrid-sleep.target; do
    state=$( { systemctl is-enabled "$tgt" 2>/dev/null || true; } | head -n1 | tr -d '[:space:]')
    [[ "$state" != "masked" ]] && unmasked+=( "$tgt(${state:-unknown})" )
done
if [[ ${#unmasked[@]} -gt 0 ]]; then
    err "FAILED: these targets are still not masked after masking them:"
    err "  ${unmasked[*]}"
    exit 1
fi

# A drop-in that is missing or empty is not a drop-in.
for f in /etc/systemd/sleep.conf.d/00-no-suspend.conf \
         /etc/systemd/logind.conf.d/00-no-idle-suspend.conf; do
    if [[ ! -s "$f" ]]; then
        err "FAILED: $f is missing or empty after being written."
        exit 1
    fi
done

if [[ "$undetermined" -ne 0 ]]; then
    err ""
    err "COULD NOT DETERMINE: the drop-ins are written and the sleep targets"
    err "  are masked, but systemd did not confirm it re-read the drop-ins, so"
    err "  the idle / lid policy may not be live. THIS IS NOT A PASS."
    err "  Resolve the warning above and re-run, or reboot, then confirm with:"
    err "    bash challenges/scripts/host_no_auto_suspend_challenge.sh"
    exit 2
fi

echo
echo "DONE — guard installed, and masking verified active."
echo "Confirm independently with:"
echo "  bash challenges/scripts/host_no_auto_suspend_challenge.sh"
echo "All 4 assertions must PASS."
exit 0
