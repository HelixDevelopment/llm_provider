# const050b_proof.sh — the shared paired-proof harness of the CONST-050(B)
# challenge family (VSC-369). SOURCED by a family challenge's --prove-failure
# arm only; never executed on its own and never read by a live run, so a live
# challenge keeps working when this file is absent (its proof then exits 2).
# Byte-identical in every repository that carries the family.
#
# ── THE ONE PROOF CONVENTION ─────────────────────────────────────────────────
# Every family challenge takes exactly one argument, --prove-failure; anything
# else is "COULD NOT DETERMINE" (exit 2). Its live exit contract is
# three-valued: 0 the subject was measured and every step passed, 1 a real
# finding (a finding outranks could-not-determine), 2 COULD NOT DETERMINE —
# never a pass. --prove-failure runs a COPY of the challenge in a hermetic
# environment (env -i: only PATH, HOME, LANG/LC_ALL, TMPDIR = a throw-away
# directory, and the case's own variables) and, per case, requires the exit
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
# Stub targets bind 127.0.0.1 only; no case ever touches a real service, the
# network beyond loopback, or a credential (CONST-042). A proof exits 0 only
# when every case held, 1 when any case did not, 2 when it could not judge one.
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
PF_CASE_TIMEOUT="${PF_CASE_TIMEOUT:-180}"

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
  echo "prove-failure: $label — $PF_SELF"
}

# pf_end — the verdict: 1 when any case failed, else 2 when any was undetermined.
pf_end() {
  echo "prove-failure: $PF_PASS passed, $PF_FAIL failed, $PF_UNDET undetermined"
  [[ "$PF_FAIL" -gt 0 ]] && return 1
  [[ "$PF_UNDET" -gt 0 ]] && return 2
  [[ "$PF_PASS" -gt 0 ]] || { echo "prove-failure: no case was judged"; return 2; }
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
# the caller's environment reaches it except PATH and HOME, so a real target
# configured in the caller's shell can never be measured by a proof case.
pf_run() {
  local copy="$1"
  shift
  local -a envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  [[ "${1-}" == "--" ]] && shift
  timeout "$PF_CASE_TIMEOUT" env -i PATH="$PATH" HOME="${HOME:-/nonexistent}" LANG=C LC_ALL=C \
    TMPDIR="$PF_T" "${envs[@]}" bash "$copy" "$@"
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

# The stub target: a loopback HTTP server. MODE (with ARG):
#   ok                  200 {"status":"ok"} on every GET
#   bad-schema          200 {"status":"down"}
#   fail-after N        200 for the first N counted GETs, then 503
#   break-on-malformed  200 until the first malformed request, then 503
#   reject-oversize     drops (no response) any request with a header > 4096 B
#   variant X           200 {"status":"ok","replica":"X"}
#   slow-after N:SECS   GETs after the first N counted ones sleep SECS
# A GET carrying an X-Slow header (the chaos slow-loris) is not counted.
# The stub exits when the proof that started it exits (parent watch, 600 s cap).
PF_STUB_PY='
import json, os, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
portfile, mode = sys.argv[1], sys.argv[2]
arg = sys.argv[3] if len(sys.argv) > 3 else "0"
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
srv = S(("127.0.0.1", 0), H)
threading.Thread(target=watch, daemon=True).start()
with open(portfile + ".tmp", "w") as f:
    f.write(str(srv.server_address[1]))
os.replace(portfile + ".tmp", portfile)
srv.serve_forever()
'

# pf_stub VAR MODE [ARG] — start a stub target; VAR receives its base URL
# (http://127.0.0.1:PORT) and VAR_PORT its port. rc 1 when it did not start.
pf_stub() {
  local var="$1" pf pid i port
  shift
  PF_N=$((PF_N + 1))
  pf="$PF_T/stub.$PF_N.port"
  python3 -I -c "$PF_STUB_PY" "$pf" "$@" >/dev/null 2>&1 &
  pid=$!
  PF_PIDS+=("$pid")
  for ((i = 0; i < 200; i++)); do
    [[ -s "$pf" ]] && break
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 0.05
  done
  port="$(cat "$pf" 2>/dev/null)"
  [[ "$port" =~ ^[0-9]+$ ]] || return 1
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
