# const050b_proof.sh — the shared paired-proof harness of the CONST-050(B)
# challenge family (VSC-369). SOURCED by a family challenge's --prove-failure
# arm, and executable on its own only as `bash const050b_proof.sh --selftest`;
# never read by a live run, so a live challenge keeps working when this file is
# absent (its proof then exits 2). Byte-identical in every repository that
# carries the family.
#
# ── THE ONE PROOF CONVENTION ─────────────────────────────────────────────────
# Every family challenge takes exactly one argument, --prove-failure; anything
# else is "COULD NOT DETERMINE" (exit 2). Its live exit contract is
# three-valued: 0 the subject was measured and every step passed, 1 a real
# finding (a finding outranks could-not-determine), 2 COULD NOT DETERMINE —
# never a pass. --prove-failure runs a COPY of the challenge in a hermetic
# environment (env -i: only PATH, LANG/LC_ALL, HOME and TMPDIR = throw-away
# directories, and the case's own variables) and, per case, requires the exit
# code AND a named defect in the output:
#
#   clean arm        a healthy loopback stub target (or fake binary / clean
#                    planted tree)                                  -> rc 0
#   stub-target arms a deliberately broken loopback stub target, one defect
#                    per case, each aimed at one step               -> rc 1
#   unreachable arm  a refused loopback port / absent binary / missing
#                    scanner                                        -> rc 2
#   unset target, malformed knob, unknown argument                  -> rc 2
#   planted copy     a defect planted in the COPY of the challenge itself
#                    (a deleted step trips the executed-step floor -> rc 1;
#                    a truncated load trips the executed-request floor -> rc 2)
#
# Stub targets bind 127.0.0.1 only, and that is MEASURED, not trusted: every
# stub reports the address it bound and pf_stub also reads the kernel's
# listening-socket table (/proc/net/tcp, tcp6); a stub listening anywhere but
# 127.0.0.1 is killed and recorded as a FAIL. No case ever touches a real
# service, the network beyond loopback, or a credential (CONST-042). A proof
# exits 0 only when every case held, 1 when any case did not, 2 when it could
# not judge one.
#
# Every proof first runs the harness SELF-TEST (pf_selftest; also
# `bash const050b_proof.sh --selftest`): verdict precedence, needle and
# forbidden-text enforcement, hermeticity (no caller variable and no caller ~/.curlrc reaches
# a case) and the loopback-only check, each against a planted defect. A failed
# self-test fails the proof before any case runs.
# host_no_auto_suspend_challenge.sh is the one member whose proof is a sibling:
# its --prove-failure runs host_no_auto_suspend_contract_challenge.sh, which
# drives it through planted host conditions inside a user+mount namespace.
#
# Requires: bash 4.3+, mktemp, timeout, and per proof curl / python3 (stub
# targets). A missing tool makes the proof exit 2, never 0.

# shellcheck shell=bash
[[ -n "${C50_PROOF_LIB_LOADED:-}" ]] && return 0
C50_PROOF_LIB_LOADED=1

PF_T=""
PF_SELF=""
PF_PASS=0
PF_FAIL=0
PF_UNDET=0
PF_N=0
PF_PIDS=()
PF_SELFTEST_HELD=0
PF_CASE_TIMEOUT="${PF_CASE_TIMEOUT:-180}"
# The address every stub binds. A shell variable, set here unconditionally, so
# no caller environment can widen it; pf_stub refuses any stub that did not end
# up listening on 127.0.0.1 alone (the self-test plants 127.0.0.2 to prove it).
PF_STUB_ADDR=127.0.0.1

pf_cleanup() {
  local p
  for p in "${PF_PIDS[@]}"; do kill "$p" 2>/dev/null; done
  for p in "${PF_PIDS[@]}"; do wait "$p" 2>/dev/null; done
  if [[ -n "$PF_T" && -d "$PF_T" ]]; then
    chmod -R u+rwx "$PF_T" 2>/dev/null
    rm -rf "$PF_T"
  fi
}
pf_ok()    { PF_PASS=$((PF_PASS + 1)); echo "  PASS $1"; }
pf_bad()   { PF_FAIL=$((PF_FAIL + 1)); echo "  FAIL $1"; }
pf_undet() { PF_UNDET=$((PF_UNDET + 1)); echo "  UNDET $1: $2"; }

# pf_begin SELF LABEL [TOOL...] — scratch directory, cleanup trap, tool check.
pf_begin() {
  PF_SELF="$1"
  local label="$2" t
  shift 2
  shopt -u patsub_replacement 2>/dev/null || true
  for t in mktemp timeout env "$@"; do
    if ! command -v "$t" >/dev/null 2>&1; then
      echo "prove-failure: cannot determine — '$t' is not on PATH"
      return 2
    fi
  done
  PF_T="$(mktemp -d)" || { echo "prove-failure: cannot determine — mktemp -d failed (TMPDIR='${TMPDIR:-}')"; return 2; }
  trap pf_cleanup EXIT
  mkdir -p "$PF_T/home" || { echo "prove-failure: cannot determine — cannot create $PF_T/home"; return 2; }
  echo "prove-failure: $label — $PF_SELF"
  local rc=0
  pf_selftest || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "prove-failure: the harness self-test did not pass (rc $rc) — no case of this proof can be trusted"
    return "$rc"
  fi
}

# pf_end — the verdict: 1 when any case failed, else 2 when any was undetermined.
pf_end() {
  echo "prove-failure: $PF_PASS passed, $PF_FAIL failed, $PF_UNDET undetermined"
  [[ "$PF_FAIL" -gt 0 ]] && return 1
  [[ "$PF_UNDET" -gt 0 ]] && return 2
  [[ "$PF_PASS" -gt 0 ]] || { echo "prove-failure: no case was judged"; return 2; }
  [[ "$PF_SELFTEST_HELD" -eq 1 ]] || { echo "prove-failure: the harness self-test did not run — no case can be trusted"; return 2; }
  return 0
}

# pf_copy CASE — a fresh copy of the challenge under test; prints its path.
pf_copy() {
  local d="$PF_T/case.$1"
  mkdir -p "$d" && cp -- "$PF_SELF" "$d/$(basename "$PF_SELF")" && printf '%s\n' "$d/$(basename "$PF_SELF")"
}

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

# pf_run COPY [VAR=VALUE...] [-- ARG...] — run a copy hermetically: nothing of
# the caller's environment reaches it except PATH, so a real target configured
# in the caller's shell can never be measured by a proof case. HOME is a fresh
# empty directory, so the caller's ~/.curlrc (a proxy, a resolve rule, a
# default header) cannot change what curl measures either.
pf_run() {
  local copy="$1"
  shift
  local -a envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  [[ "${1-}" == "--" ]] && shift
  timeout "$PF_CASE_TIMEOUT" env -i PATH="$PATH" HOME="$PF_T/home" LANG=C LC_ALL=C \
    TMPDIR="$PF_T" "${envs[@]}" bash "$copy" "$@"
}

# pf_expect NAME WANT NEEDLE CMD... — rc must equal WANT and, when NEEDLE is
# non-empty, the combined output must contain it.
pf_expect() {
  local name="$1" want="$2" needle="$3"
  shift 3
  _pf_judge "$name" "$want" "$needle" "" "$@"
}

# pf_expect_without NAME WANT NEEDLE ABSENT CMD... — as pf_expect, and the
# combined output must NOT contain ABSENT (e.g. a run exactly at a pass-rate
# floor must not call itself permissive).
pf_expect_without() {
  local name="$1" want="$2" needle="$3" absent="$4"
  shift 4
  _pf_judge "$name" "$want" "$needle" "$absent" "$@"
}

_pf_judge() {
  local name="$1" want="$2" needle="$3" absent="$4" out rc
  shift 4
  out="$("$@" 2>&1)"; rc=$?
  if [[ "$rc" -eq "$want" ]] && { [[ -z "$needle" ]] || grep -qF -- "$needle" <<<"$out"; } &&
     { [[ -z "$absent" ]] || ! grep -qF -- "$absent" <<<"$out"; }; then
    pf_ok "$name (rc $rc)"
  else
    pf_bad "$name: want rc $want${needle:+ naming \"$needle\"}${absent:+ and never \"$absent\"}, got rc $rc"
    printf '%s\n' "$out" | tail -n 6 | sed 's/^/       | /'
  fi
}

# The stub target: a loopback HTTP server. MODE (with ARG):
#   ok                  200 {"status":"ok"} on every GET
#   bad-schema          200 {"status":"down"}
#   fail-after N        200 for the first N counted GETs, then 503
#   break-on-malformed  200 until the first malformed request, then 503
#   reject-oversize     drops (no response) any request with a header > 4096 B
#   variant X           200 {"status":"ok","replica":"X"}
#   slow-after N:SECS   GETs after the first N counted ones sleep SECS
#   fail-window A:B     503 for counted GETs A+1..B, 200 before and after
#                       (a target that degrades for a while and recovers)
# A GET carrying an X-Slow header (the chaos slow-loris) is not counted.
# The stub exits when the proof that started it exits (parent watch, 600 s cap).
PF_STUB_PY='
import json, os, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
bind, portfile, mode = sys.argv[1], sys.argv[2], sys.argv[3]
arg = sys.argv[4] if len(sys.argv) > 4 else "0"
ppid = os.getppid()
st = {"n": 0, "broken": False}
lk = threading.Lock()
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    def log_message(self, *a):
        pass
    def send_error(self, *a, **k):
        st["broken"] = True
        try:
            super().send_error(*a, **k)
        except OSError:
            pass
    def do_GET(self):
        with lk:
            if self.headers.get("X-Slow") is None:
                st["n"] += 1
            n = st["n"]
        code, body = 200, {"status": "ok"}
        if mode == "bad-schema":
            body = {"status": "down"}
        elif mode == "fail-after" and n > int(arg):
            code, body = 503, {"status": "unavailable"}
        elif mode == "break-on-malformed" and st["broken"]:
            code, body = 503, {"status": "unavailable"}
        elif mode == "reject-oversize" and any(len(v) > 4096 for v in self.headers.values()):
            self.close_connection = True
            return
        elif mode == "variant":
            body = {"status": "ok", "replica": arg}
        elif mode == "slow-after" and n > int(arg.split(":")[0]):
            time.sleep(float(arg.split(":")[1]))
        elif mode == "fail-window" and int(arg.split(":")[0]) < n <= int(arg.split(":")[1]):
            code, body = 503, {"status": "unavailable"}
        data = json.dumps(body).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass
class S(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512
    def handle_error(self, request, client_address):
        pass
def watch():
    t0 = time.time()
    while time.time() - t0 < 600 and os.getppid() == ppid:
        time.sleep(0.5)
    os._exit(0)
srv = S((bind, 0), H)
threading.Thread(target=watch, daemon=True).start()
with open(portfile + ".tmp", "w") as f:
    f.write("%s %d" % srv.server_address[:2])
os.replace(portfile + ".tmp", portfile)
srv.serve_forever()
'

# pf_loopback_only PORT — read the kernel's listening sockets for PORT: 0 when
# every listener on it is 127.0.0.1 (and there is at least one), 1 when any
# listens on another address (0.0.0.0, 127.0.0.2, any IPv6 address), 2 when
# /proc/net/tcp cannot be read. Independent of what the stub says about itself.
pf_loopback_only() {
  local hex n=0 wide=0 f _sl laddr _raddr st _rest
  printf -v hex '%04X' "$1"
  [[ -r /proc/net/tcp ]] || return 2
  for f in /proc/net/tcp /proc/net/tcp6; do
    [[ -r "$f" ]] || continue
    while read -r _sl laddr _raddr st _rest; do
      [[ "$st" == 0A && "${laddr##*:}" == "$hex" ]] || continue
      n=$((n + 1))
      [[ "$f" == /proc/net/tcp && "${laddr%%:*}" == 0100007F ]] || wide=1
    done < "$f"
  done
  [[ "$wide" -eq 1 ]] && return 1
  [[ "$n" -ge 1 ]] || return 2
  return 0
}

# pf_stub VAR MODE [ARG] — start a stub target; VAR receives its base URL
# (http://127.0.0.1:PORT) and VAR_PORT its port. rc 1 when it did not start, or
# when it is listening anywhere but 127.0.0.1 (then it is killed and the case
# is recorded as a FAIL: a proof must never open a port beyond loopback).
pf_stub() {
  local var="$1" pf pid i rep addr port lrc=0
  shift
  PF_N=$((PF_N + 1))
  pf="$PF_T/stub.$PF_N.$BASHPID.port"
  rm -f -- "$pf"
  python3 -I -c "$PF_STUB_PY" "$PF_STUB_ADDR" "$pf" "$@" >/dev/null 2>&1 &
  pid=$!
  PF_PIDS+=("$pid")
  for ((i = 0; i < 200; i++)); do
    [[ -s "$pf" ]] && break
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 0.05
  done
  rep="$(cat "$pf" 2>/dev/null)"
  addr="${rep% *}"
  port="${rep##* }"
  [[ "$port" =~ ^[0-9]+$ ]] || { kill "$pid" 2>/dev/null; return 1; }
  pf_loopback_only "$port" || lrc=$?
  if [[ "$addr" != 127.0.0.1 || "$lrc" -eq 1 ]]; then
    kill "$pid" 2>/dev/null
    pf_bad "stub target bound '$addr' port $port, not 127.0.0.1 alone (kernel listening table rc $lrc) — killed"
    return 1
  fi
  if [[ "$lrc" -ne 0 ]]; then
    kill "$pid" 2>/dev/null
    pf_undet "stub target" "cannot read /proc/net/tcp to confirm it listens on 127.0.0.1 alone"
    return 1
  fi
  printf -v "$var" 'http://127.0.0.1:%s' "$port"
  printf -v "${var}_PORT" '%s' "$port"
}

# pf_closed_port VAR — a loopback port that was just bound and released, so a
# connection to it is refused: the "unreachable target" of the convention.
pf_closed_port() {
  local p
  p="$(python3 -I -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()' 2>/dev/null)" || return 1
  [[ "$p" =~ ^[0-9]+$ ]] || return 1
  printf -v "$1" '%s' "$p"
}

# pf_bin VAR KIND — a fake command-line binary; VAR receives its path. KIND:
#   clean              --help / --version answer, an unknown flag exits 2
#   panic-help         --help prints a Go panic and a goroutine trace
#   big-panic-help     --help prints a panic FIRST, then > 64 KiB of text
#   empty-help         --help and -h print nothing
#   panic-version      --version prints "fatal error:"
#   panic-bogus        an unknown flag prints a Go panic
#   hang-bogus         an unknown flag never returns
#   break-after-bogus  after one unknown flag, --help and -h print nothing
pf_bin() {
  local f
  PF_N=$((PF_N + 1))
  f="$PF_T/bin.$PF_N"
  {
    printf '#!/usr/bin/env bash\nkind=%q\n' "$2"
    cat <<'EOF'
usage() { echo "usage: proofbin [--help] [--version]"; }
case "${1-}" in
  --help|-h)
    case "$kind" in
      panic-help) echo "panic: runtime error: index out of range [3] with length 3"
                  echo "goroutine 1 [running]:"; exit 2 ;;
      big-panic-help) echo "panic: runtime error: invalid memory address"
                      head -c 70000 /dev/zero | tr '\0' 'x'; echo; exit 0 ;;
      empty-help) exit 0 ;;
      break-after-bogus) [[ -e "$0.state" ]] && exit 0; usage ;;
      *) usage ;;
    esac ;;
  --version|-v)
    [[ "$kind" == panic-version ]] && { echo "fatal error: all goroutines are asleep - deadlock!"; exit 2; }
    echo "proofbin 1.0.0" ;;
  *)
    [[ "$kind" == hang-bogus ]] && exec sleep 600
    [[ "$kind" == panic-bogus ]] && { echo "panic: flag provided but not defined"; echo "goroutine 1 [running]:"; exit 2; }
    [[ "$kind" == break-after-bogus ]] && : > "$0.state"
    echo "proofbin: unknown flag '${1-}'" >&2; exit 2 ;;
esac
EOF
  } > "$f" && chmod 755 "$f" || return 1
  printf -v "$1" '%s' "$f"
}

# pf_selftest — the harness proves itself against planted defects before any
# case runs (called by pf_begin; prints "  self-test ..." lines). 0 every check
# held, 1 one did not, 2 one could not be judged (a check whose tool is absent
# is skipped and said so; the proof that needs that tool already requires it).
pf_selftest() {
  # shellcheck disable=SC2034  # url_PORT is written by pf_stub (printf -v)
  local sfail=0 sundet=0 ok rc out url url_PORT closed lpid lport i
  _pf_st() { if [[ "$2" -eq 0 ]]; then echo "  self-test PASS $1"; else echo "  self-test FAIL $1"; sfail=1; fi; }
  # verdict precedence (pf_end)
  ( PF_PASS=1; PF_FAIL=1; PF_UNDET=1; pf_end ) >/dev/null; rc=$?
  ok=1; [[ "$rc" -eq 1 ]] && ok=0; _pf_st "T1 a failed case outranks an undetermined one (pf_end rc $rc)" "$ok"
  ( PF_PASS=1; PF_FAIL=0; PF_UNDET=1; pf_end ) >/dev/null; rc=$?
  ok=1; [[ "$rc" -eq 2 ]] && ok=0; _pf_st "T2 an undetermined case is rc 2, never 0 (pf_end rc $rc)" "$ok"
  ( PF_PASS=1; PF_FAIL=0; PF_UNDET=0; PF_SELFTEST_HELD=1; pf_end ) >/dev/null; rc=$?
  ok=1; [[ "$rc" -eq 0 ]] && ok=0; _pf_st "T3 every case held is rc 0 (pf_end rc $rc)" "$ok"
  ( PF_PASS=1; PF_FAIL=0; PF_UNDET=0; PF_SELFTEST_HELD=0; pf_end ) >/dev/null; rc=$?
  ok=1; [[ "$rc" -eq 2 ]] && ok=0; _pf_st "T3b every case held but no self-test ran is rc 2 (pf_end rc $rc)" "$ok"
  ( PF_PASS=0; PF_FAIL=0; PF_UNDET=0; pf_end ) >/dev/null; rc=$?
  ok=1; [[ "$rc" -eq 2 ]] && ok=0; _pf_st "T4 no case judged is rc 2 (pf_end rc $rc)" "$ok"
  # needle and rc enforcement (pf_expect)
  out="$(PF_PASS=0; PF_FAIL=0; pf_expect st 0 "a-defect-name-that-is-absent" true >/dev/null; echo "$PF_PASS/$PF_FAIL")"
  ok=1; [[ "$out" == 0/1 ]] && ok=0; _pf_st "T5 the right rc without the named defect is a FAIL ($out)" "$ok"
  out="$(PF_PASS=0; PF_FAIL=0; pf_expect st 1 "" true >/dev/null; echo "$PF_PASS/$PF_FAIL")"
  ok=1; [[ "$out" == 0/1 ]] && ok=0; _pf_st "T6 a wrong rc is a FAIL ($out)" "$ok"
  out="$(PF_PASS=0; PF_FAIL=0; pf_expect st 3 "named" bash -c 'echo named; exit 3' >/dev/null; echo "$PF_PASS/$PF_FAIL")"
  ok=1; [[ "$out" == 1/0 ]] && ok=0; _pf_st "T7 the right rc naming the defect is a PASS ($out)" "$ok"
  out="$(PF_PASS=0; PF_FAIL=0; pf_expect_without st 0 "named" "forbidden" bash -c 'echo named; echo forbidden' >/dev/null; echo "$PF_PASS/$PF_FAIL")"
  ok=1; [[ "$out" == 0/1 ]] && ok=0; _pf_st "T13 the right rc and needle but a forbidden text present is a FAIL ($out)" "$ok"
  out="$(PF_PASS=0; PF_FAIL=0; pf_expect_without st 0 "named" "forbidden" bash -c 'echo named' >/dev/null; echo "$PF_PASS/$PF_FAIL")"
  ok=1; [[ "$out" == 1/0 ]] && ok=0; _pf_st "T14 the right rc and needle with the forbidden text absent is a PASS ($out)" "$ok"
  # hermeticity (pf_run): a caller variable and the caller's HOME never reach a case
  printf '%s\n' 'echo "home=$HOME canary=${PF_SELFTEST_CANARY-unset}"' > "$PF_T/st-env.sh"
  out="$(PF_SELFTEST_CANARY=leaked HOME="$PF_T/callerhome" pf_run "$PF_T/st-env.sh" 2>&1)"
  ok=1; [[ "$out" == "home=$PF_T/home canary=unset" ]] && ok=0; _pf_st "T8 a case sees a fresh HOME and no caller variable ($out)" "$ok"
  if command -v python3 >/dev/null 2>&1; then
    # loopback-only binding: a stub forced onto 127.0.0.2 must be refused ...
    out="$(PF_STUB_ADDR=127.0.0.2; pf_stub url ok; echo "rc=$?")"
    ok=1; [[ "$out" == *"not 127.0.0.1 alone (kernel listening table rc 1)"* && "$out" == *rc=1 ]] && ok=0; _pf_st "T9 a stub listening beyond 127.0.0.1 is refused and killed, by its own report and by the kernel table" "$ok"
    # ... and the kernel-table check alone must see a 127.0.0.2 listener as wide.
    python3 -I -c 'import socket, sys, time
s = socket.socket(); s.bind(("127.0.0.2", 0)); s.listen(1)
open(sys.argv[1] + ".tmp", "w").write(str(s.getsockname()[1]))
import os; os.replace(sys.argv[1] + ".tmp", sys.argv[1]); time.sleep(30)' "$PF_T/st-listener.port" >/dev/null 2>&1 &
    lpid=$!
    PF_PIDS+=("$lpid")
    for ((i = 0; i < 200; i++)); do [[ -s "$PF_T/st-listener.port" ]] && break; sleep 0.05; done
    lport="$(cat "$PF_T/st-listener.port" 2>/dev/null)"
    if [[ "$lport" =~ ^[0-9]+$ ]]; then
      rc=0; pf_loopback_only "$lport" || rc=$?
      ok=1; [[ "$rc" -eq 1 ]] && ok=0; _pf_st "T10 the kernel listening table shows a 127.0.0.2 listener as wide (rc $rc)" "$ok"
    else
      echo "  self-test UNDET T10 the 127.0.0.2 listener did not start"; sundet=1
    fi
    kill "$lpid" 2>/dev/null
    # a stub on 127.0.0.1 passes the same checks
    local f0="$PF_FAIL"
    if pf_stub url ok; then
      _pf_st "T11 a stub on 127.0.0.1 is accepted" 0
      if command -v curl >/dev/null 2>&1 && pf_closed_port closed; then
        # the caller's ~/.curlrc (here: a proxy on a refused port) must not reach a case
        mkdir -p "$PF_T/callerhome" && printf 'proxy = "http://127.0.0.1:%s"\n' "$closed" > "$PF_T/callerhome/.curlrc"
        printf '%s\n' 'curl -sS -o /dev/null -w "%{http_code}" "$1"' > "$PF_T/st-curl.sh"
        out="$(HOME="$PF_T/callerhome" pf_run "$PF_T/st-curl.sh" -- "$url/health" 2>&1)"
        ok=1; [[ "$out" == 200 ]] && ok=0; _pf_st "T12 a caller ~/.curlrc proxy line does not reach a case (curl answered '$out')" "$ok"
      else
        echo "  self-test SKIP T12 (no curl in this proof)"
      fi
    elif [[ "$PF_FAIL" -gt "$f0" ]]; then
      _pf_st "T11 a stub asked for 127.0.0.1 was refused: it listened beyond loopback" 1
    else
      echo "  self-test UNDET T11 a stub on 127.0.0.1 did not start"; sundet=1
    fi
  else
    echo "  self-test SKIP T9-T12 (no python3 in this proof)"
  fi
  [[ "$sfail" -eq 0 ]] || return 1
  [[ "$sundet" -eq 0 ]] || return 2
  PF_SELFTEST_HELD=1
  return 0
}

# Executed on its own: `bash const050b_proof.sh --selftest`. Anything else is 2.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1-}" in
    --selftest)
      pf_begin "${BASH_SOURCE[0]}" "harness self-test" || exit $?
      echo "prove-failure: harness self-test held"
      exit 0 ;;
    *)
      echo "COULD NOT DETERMINE: this file is the CONST-050(B) proof harness; run it only as --selftest (got '${1-}')." >&2
      exit 2 ;;
  esac
fi
