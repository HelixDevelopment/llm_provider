#!/usr/bin/env bash
# apikeys_live_discovery_challenge.sh
#
# CONST-035 anti-bluff challenge: prove the LLMProvider pkg/apikeys credential
# wiring reaches a real provider endpoint — not a mock, not a stub, not a
# structural check.
#
# What is exercised, stated precisely (§11.4.6): pkg/apikeys.Scan() resolves
# the probe provider's credential from the environment, and a one-shot Go probe
# sends it as a bearer token in a direct GET of HuggingFace's public model list.
# pkg/discovery is NOT exercised — this header used to claim "apikeys +
# discovery wiring", but the probe never imported pkg/discovery; that claim is
# withdrawn, not restated.
#
# Steps:
#   1. Source $API_KEYS_FILE (default ~/api_keys.sh, the operator's credential
#      file) and require ApiKey_<PROBE_PROVIDER> to be set AND exported (an
#      unexported variable never reaches the probe's environment).
#   2. Build the probe against this module (`go build`; the import resolves
#      via this module's go.mod).
#   3. Run it. It prints "OK: discovered N models ..." only when the endpoint
#      returned a non-empty list whose first id is non-empty; an exit 0 without
#      that evidence line is a finding, not a pass.
#
# Credential handling (CONST-042): the credential file is sourced into this
# process with its stdout discarded (a file that echoes its keys prints
# nothing here; its stderr is kept); ONLY ApiKey_<PROBE_PROVIDER> is sent
# anywhere, and only to the probe's endpoint, which is HuggingFace's.
# PROBE_PROVIDER is therefore limited to HuggingFace — the previous body
# accepted any provider name and sent that provider's key to huggingface.co.
# No key value is ever printed.
#
# Honest boundary: the model list is a public endpoint, so whether HuggingFace
# rejects a WRONG bearer token was not measured here (a live run reads real
# credentials); a live 0 therefore proves the key is resolved and sent, not
# that its value is right. The --prove-failure fake endpoint does compare the
# token, which is how a tampered key value is proven to be caught.
#
# Usage (from anywhere; the module root is derived from this file's location):
#   bash challenges/scripts/apikeys_live_discovery_challenge.sh
#   bash challenges/scripts/apikeys_live_discovery_challenge.sh --prove-failure
#
# ── EXIT CODES — three-valued, and 2 is NEVER a pass ─────────────────────────
#   0  the probe resolved the key, reached the endpoint and printed its
#      evidence line: a non-empty model list with a non-empty first id
#   1  a real finding: the key was set and exported and the wiring failed —
#      pkg/apikeys did not resolve it, the probe does not build against this
#      module, the endpoint answered 4xx (e.g. a rejected token), the body was
#      not a model list, the list or its first id was empty, or the probe
#      exited 0 without its evidence line
#   2  COULD NOT DETERMINE — an unknown argument, a PROBE_PROVIDER with no wired
#      endpoint, no readable credential file, no key or a key that is not
#      exported, a key value containing a line break, no usable go toolchain,
#      a module that cannot be loaded,
#      mktemp failing, the endpoint unreachable or answering 5xx, or an
#      unrecognised probe exit. NOTHING about the wiring was established.
#
# The header used to promise "never PASS-on-absence", and then both absence
# branches printed SKIP-OK and `exit 0`; they exit 2. It also ran the probe with
# `go run`, which collapses every probe exit status to 1 — an unreachable
# network was reported as a FAIL. The probe is now built and run directly so
# its exit status survives.
#
# --prove-failure (§1.1 paired proof, VSC-370): NEVER reads real credentials.
# Runs copies of this module under $TMPDIR with `env -i`, a scratch HOME,
# fixture credential files holding synthetic keys, GOPROXY=off, and the probe
# endpoint rewritten to a fake model-list server on 127.0.0.1 that checks the
# bearer token and logs every request in full; every fixture also exports a
# synthetic ApiKey_Nvidia, and in EVERY case neither synthetic key may appear
# in the output nor the Nvidia key anywhere in a request (line, headers, body).
# An empty list, an empty id, a 404, a non-JSON body, a Scan() that loses the
# key, a Get() that tampers with it, a probe without its evidence line and
# wiring that does not compile must exit 1; a non-HuggingFace PROBE_PROVIDER
# must exit 2, and a credential file that re-points it must not take effect; a
# credential file that echoes its keys must still exit 0 with nothing printed;
# an unknown argument, no credential file, no key, an unexported key, an
# unreachable or 503 endpoint, no go, an unusable TMPDIR, an unreadable
# credential file, an unloadable module and a key value with a line break must
# exit 2. Exits 0 only when every case holds, 1 when any does not, 2 when the
# proof could not judge a case.

set -o pipefail
# Deliberately NOT using -u (or -e) here: ~/api_keys.sh re-exports some env
# vars that reference other unset env vars (e.g., Google_Vertex_AI). The user's
# file is the source of truth and must be sourced as-is; every step below
# checks its own status instead of relying on -e.

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

PROVE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prove-failure) PROVE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "COULD NOT DETERMINE: unknown argument '$1' — this challenge takes only --prove-failure."
       exit 2 ;;
  esac
done

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── paired proof (--prove-failure, §1.1, VSC-370) ───────────────────────────
# NEVER touches real credentials (CONST-042). Every case runs a copy of THIS
# module (no .git) under $TMPDIR with `env -i`: HOME is a scratch directory,
# API_KEYS_FILE names a fixture holding SYNTHETIC keys only, GOPROXY=off and
# GOTOOLCHAIN=local keep the build off the network, and the copy's probe
# endpoint is rewritten (exactly once) to a fake model-list server on 127.0.0.1
# that compares the bearer token and logs EVERY request in full (request line,
# all headers, body). Every fixture also exports a synthetic ApiKey_Nvidia. A
# case holds only when the exit code matches AND the output names the defect
# AND no synthetic key appears in the output AND no request made during the
# case carries the Nvidia key anywhere (only ApiKey_HuggingFace may be sent).
# Exit: 0 every case held, 1 any case did not, 2 the proof could not judge one.
PF_T=""
PF_SRV=""
PF_PASS=0
PF_FAIL=0
PF_UNDET=0
PF_SECRETS=()      # values that must never appear in a case's output
PF_REQLOG=""       # the fake endpoint's full request log
PF_NOSEND=""       # a value that must never appear in any request
PF_LAST_OUT=""     # the output of the last pf_expect run
pf_cleanup() {
  if [[ -n "$PF_SRV" ]]; then kill "$PF_SRV" 2>/dev/null; wait "$PF_SRV" 2>/dev/null; fi
  [[ -n "$PF_T" && -d "$PF_T" ]] && { chmod -R u+rwx "$PF_T" 2>/dev/null; rm -rf "$PF_T"; }
}
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
# Also: no PF_SECRETS value in the output, and no request logged during the
# case carries PF_NOSEND.
pf_expect() {
  local name="$1" want="$2" needle="$3" out rc pre=0 sec leak=""
  shift 3
  [[ -n "$PF_REQLOG" ]] && pre="$(wc -l < "$PF_REQLOG")"
  out="$("$@" 2>&1)"; rc=$?
  PF_LAST_OUT="$out"
  for sec in "${PF_SECRETS[@]}"; do
    grep -qF -- "$sec" <<<"$out" && leak+=" a synthetic key was PRINTED;"
  done
  if [[ -n "$PF_REQLOG" && -n "$PF_NOSEND" ]] && tail -n +"$((pre + 1))" "$PF_REQLOG" | grep -qF -- "$PF_NOSEND"; then
    leak+=" the Nvidia key reached the endpoint;"
  fi
  if [[ -z "$leak" && "$rc" -eq "$want" ]] && { [[ -z "$needle" ]] || grep -qF -- "$needle" <<<"$out"; }; then
    pf_ok "$name (rc $rc)"
  else
    pf_bad "$name: want rc $want${needle:+ naming \"$needle\"}, got rc $rc${leak:+ —$leak}"
    for sec in "${PF_SECRETS[@]}"; do out="${out//"$sec"/<synthetic-key>}"; done
    printf '%s\n' "$out" | tail -n 6 | sed 's/^/       | /'
  fi
}
prove_failure() {
  shopt -u patsub_replacement 2>/dev/null || true
  local mod gname gocache gomodcache port i d ep_from ok_from k
  local key='synthetic-hf-key-vsc370-not-a-secret' nkey='synthetic-nvidia-key-vsc370-not-a-secret'
  mod="$(cd "$(dirname "$SELF")/../.." && pwd)"
  gname="$(basename "$SELF")"
  for k in go python3 tar; do
    command -v "$k" >/dev/null 2>&1 || { echo "prove-failure: cannot determine — $k is not on PATH"; return 2; }
  done
  [[ -f "$mod/go.mod" ]] || { echo "prove-failure: cannot determine — no go.mod at $mod"; return 2; }
  gocache="$(go env GOCACHE 2>/dev/null)"; gomodcache="$(go env GOMODCACHE 2>/dev/null)"
  [[ -n "$gocache" && -n "$gomodcache" ]] || { echo "prove-failure: cannot determine — go env GOCACHE/GOMODCACHE empty"; return 2; }
  PF_T="$(mktemp -d)" || { echo "prove-failure: cannot determine — mktemp -d failed"; return 2; }
  trap pf_cleanup EXIT
  mkdir -p "$PF_T/home" "$PF_T/tmp" || { echo "prove-failure: cannot determine — scratch dirs"; return 2; }
  # Fixture credential files — synthetic values only.
  # Every fixture also exports the synthetic Nvidia key, so a probe that sends
  # any other key is caught in every case, not only in K1/K2.
  local nv
  nv="$(printf "export ApiKey_Nvidia='%s'" "$nkey")"
  printf "export ApiKey_HuggingFace='%s'\n%s\n" "$key" "$nv" > "$PF_T/keys_ok.sh"
  printf "export ApiKey_SomeOtherProvider='synthetic-other'\n%s\n" "$nv" > "$PF_T/keys_nokey.sh"
  printf "ApiKey_HuggingFace='%s'\n%s\n" "$key" "$nv" > "$PF_T/keys_unexported.sh"
  # A credential file that prints its keys while it is sourced.
  printf "export ApiKey_HuggingFace='%s'\n%s\necho \"loaded \$ApiKey_HuggingFace \$ApiKey_Nvidia\"\n" "$key" "$nv" > "$PF_T/keys_echo.sh"
  # A key value with a line break in it (Go refuses it as a header value).
  printf "export ApiKey_HuggingFace=\$'%s-line1\\n%s-line2'\n%s\n" "$key" "$key" "$nv" > "$PF_T/keys_newline.sh"
  printf "export ApiKey_HuggingFace='%s'\nexport ApiKey_Nvidia='%s'\n" "$key" "$nkey" > "$PF_T/keys_nvidia.sh"
  printf "export ApiKey_HuggingFace='%s'\nexport ApiKey_Nvidia='%s'\nPROBE_PROVIDER=Nvidia\n" "$key" "$nkey" > "$PF_T/keys_repoint.sh"
  printf "export ApiKey_HuggingFace='%s'\n%s\n" "$key" "$nv" > "$PF_T/keys_unreadable.sh"
  chmod 000 "$PF_T/keys_unreadable.sh"
  cat > "$PF_T/fake_endpoint.py" <<'PY'
import http.server, os, sys
mode_file, log_file, port_file, expected = sys.argv[1:5]
BODIES = {"ok": (200, b'[{"id":"proof/model-a"},{"id":"proof/model-b"}]'),
          "empty": (200, b"[]"), "emptyid": (200, b'[{"id":""}]'),
          "badjson": (200, b"this is not json"), "404": (404, b'{"error":"not found"}'),
          "503": (503, b"upstream down")}
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        auth = self.headers.get("Authorization", "")
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n).decode("utf-8", "replace") if n > 0 else ""
        with open(log_file, "a") as f:
            f.write(self.requestline + "\n")
            for k, v in self.headers.items():
                f.write(k + ": " + v + "\n")
            f.write("body: " + body.replace("\n", "\\n") + "\n")
        mode = open(mode_file).read().strip()
        if not self.path.startswith("/api/models"):
            code, body = 404, b'{"error":"no such path"}'
        elif auth != "Bearer " + expected:
            code, body = 401, b'{"error":"invalid token"}'
        else:
            code, body = BODIES.get(mode, (500, b"bad mode"))
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
srv = http.server.HTTPServer(("127.0.0.1", 0), H)
with open(port_file + ".tmp", "w") as f:
    f.write(str(srv.server_address[1]))
os.rename(port_file + ".tmp", port_file)
srv.serve_forever()
PY
  printf 'ok\n' > "$PF_T/mode"
  : > "$PF_T/requests.log"
  python3 -I "$PF_T/fake_endpoint.py" "$PF_T/mode" "$PF_T/requests.log" "$PF_T/port" "$key" &
  PF_SRV=$!
  for ((i = 0; i < 100; i++)); do [[ -s "$PF_T/port" ]] && break; sleep 0.1; done
  port="$(cat "$PF_T/port" 2>/dev/null)"
  [[ "$port" =~ ^[0-9]+$ ]] || { echo "prove-failure: cannot determine — the fake endpoint did not start"; return 2; }
  echo "prove-failure: apikeys live-discovery challenge $SELF"
  echo "               fake endpoint 127.0.0.1:$port, synthetic keys only, HOME=$PF_T/home"
  # The endpoint literal is assembled from halves so it occurs once in this
  # file (in the probe), never in the proof arm itself.
  ep_from='"https://huggingface'; ep_from+='.co/api/models?limit=5"'
  # pf_case NAME [ENDPOINT] — a module copy whose probe targets ENDPOINT
  # (default: the fake server); prints its path.
  pf_case() {
    local c="$PF_T/$1" ep="${2:-http://127.0.0.1:$port/api/models?limit=5}"
    mkdir -p "$c" && tar -C "$mod" --exclude=./.git -cf - . | tar -C "$c" -xf - &&
      pf_plant "$c/challenges/scripts/$gname" "$ep_from" "\"$ep\"" >/dev/null && printf '%s\n' "$c"
  }
  # pf_run DIR KEYFILE [VAR=VAL...] [-- ARGS...] — the copy under env -i.
  pf_run() {
    local c="$1" keys="$2" envs=()
    shift 2
    while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
    [[ "${1:-}" == "--" ]] && shift
    env -i PATH="$PATH" HOME="$PF_T/home" TMPDIR="$PF_T/tmp" GOCACHE="$gocache" \
      GOMODCACHE="$gomodcache" GOPROXY=off GOTOOLCHAIN=local GOFLAGS=-mod=readonly \
      API_KEYS_FILE="$keys" "${envs[@]}" "$BASH" "$c/challenges/scripts/$gname" "$@"
  }
  pf_mode() { printf '%s\n' "$1" > "$PF_T/mode"; }
  local K="$PF_T/keys_ok.sh"
  PF_SECRETS=("$key" "$nkey")
  PF_REQLOG="$PF_T/requests.log"
  PF_NOSEND="$nkey"

  pf_mode ok
  if d="$(pf_case c0)"; then pf_expect "C0 clean control: key resolved, list fetched" 0 "OK: discovered 2 models" pf_run "$d" "$K"
  else pf_undet "C0" "could not build the case"; fi
  # M1..M4: the endpoint's answer is wrong
  local -a M_MODE=(empty emptyid 404 badjson) M_NEEDLE=("empty model list" "empty id" "status 404" "json decode")
  local -a M_NAME=("M1 empty model list" "M2 first model has an empty id" "M3 endpoint answers 404" "M4 body is not JSON")
  for i in "${!M_MODE[@]}"; do
    pf_mode "${M_MODE[$i]}"
    if d="$(pf_case "m$i")"; then pf_expect "${M_NAME[$i]}" 1 "${M_NEEDLE[$i]}" pf_run "$d" "$K"
    else pf_undet "${M_NAME[$i]%% *}" "could not build the case"; fi
  done
  pf_mode ok
  # M5/M6/M8: a fake key source / broken wiring, planted in the copy's pkg/apikeys
  if d="$(pf_case m5)" && pf_plant "$d/pkg/apikeys/apikeys.go" '		out[provider] = val' '		_ = val'; then
    pf_expect "M5 Scan() loses the exported key" 1 "snapshot did not find" pf_run "$d" "$K"
  else pf_undet "M5" "could not plant the defect"; fi
  if d="$(pf_case m6)" && pf_plant "$d/pkg/apikeys/apikeys.go" 'return s[provider]' 'return s[provider] + "-tampered"'; then
    pf_expect "M6 Get() returns a different key (endpoint rejects it)" 1 "status 401" pf_run "$d" "$K"
  else pf_undet "M6" "could not plant the defect"; fi
  ok_from='fmt.Printf("OK: discovered'; ok_from+=' %d models from %s, first=%q\n", len(models), provider, firstID)'
  if d="$(pf_case m7)" && pf_plant "$d/challenges/scripts/$gname" "$ok_from" '_ = firstID'; then
    pf_expect "M7 probe exits 0 without its evidence line" 1 "evidence line" pf_run "$d" "$K"
  else pf_undet "M7" "could not plant the defect"; fi
  if d="$(pf_case m8)" && pf_plant "$d/pkg/apikeys/apikeys.go" 'func Scan() Snapshot {' 'func Scan() Snapshot { undefinedSymbolVSC370()'; then
    pf_expect "M8 the apikeys wiring does not compile" 1 "undefinedSymbolVSC370" pf_run "$d" "$K"
  else pf_undet "M8" "could not plant the defect"; fi
  # K1: a PROBE_PROVIDER other than HuggingFace must never send its key anywhere
  if d="$(pf_case k1)"; then
    pf_expect "K1 PROBE_PROVIDER=Nvidia is refused (rc 2)" 2 "PROBE_PROVIDER" pf_run "$d" "$PF_T/keys_nvidia.sh" PROBE_PROVIDER=Nvidia
  else pf_undet "K1" "could not build the case"; fi
  # K2: the credential file itself re-points PROBE_PROVIDER after the check
  if d="$(pf_case k2)"; then
    pf_expect "K2 a credential file cannot re-point PROBE_PROVIDER" 0 "OK: discovered 2 models from HuggingFace" pf_run "$d" "$PF_T/keys_repoint.sh"
  else pf_undet "K2" "could not build the case"; fi
  # S1: a credential file that prints its keys while sourced must not leak them
  if d="$(pf_case s1)"; then
    pf_expect "S1 a credential file's own output is not printed" 0 "OK: discovered 2 models" pf_run "$d" "$PF_T/keys_echo.sh"
  else pf_undet "S1" "could not build the case"; fi
  # U1..U10: could not determine
  if d="$(pf_case u1)"; then pf_expect "U1 unknown argument" 2 "unknown argument" pf_run "$d" "$K" -- --no-such-flag
  else pf_undet "U1" "could not build the case"; fi
  if d="$(pf_case u2)"; then pf_expect "U2 no credential file" 2 "not readable" pf_run "$d" "$PF_T/no-such-file.sh"
  else pf_undet "U2" "could not build the case"; fi
  if d="$(pf_case u3)"; then pf_expect "U3 no key for the probe provider" 2 "ApiKey_HuggingFace" pf_run "$d" "$PF_T/keys_nokey.sh"
  else pf_undet "U3" "could not build the case"; fi
  if d="$(pf_case u4)"; then pf_expect "U4 key set but not exported" 2 "not exported" pf_run "$d" "$PF_T/keys_unexported.sh"
  else pf_undet "U4" "could not build the case"; fi
  if d="$(pf_case u5 'http://127.0.0.1:1/api/models?limit=5')"; then
    pf_expect "U5 endpoint unreachable is rc 2, not a FAIL" 2 "unreachable" pf_run "$d" "$K"
  else pf_undet "U5" "could not build the case"; fi
  pf_mode 503
  if d="$(pf_case u6)"; then pf_expect "U6 endpoint answers 503 is rc 2, not a FAIL" 2 "upstream unavailable" pf_run "$d" "$K"
  else pf_undet "U6" "could not build the case"; fi
  pf_mode ok
  if d="$(pf_case u7)" && mkdir -p "$d/nogo" && for k in dirname basename mktemp rm cat; do
       ln -s "$(command -v "$k")" "$d/nogo/$k" || break; done && [[ -e "$d/nogo/cat" ]]; then
    pf_expect "U7 no go toolchain" 2 "go toolchain" pf_run "$d" "$K" PATH="$d/nogo"
  else pf_undet "U7" "could not build a PATH without go"; fi
  if d="$(pf_case u8)"; then pf_expect "U8 TMPDIR unusable (mktemp fails)" 2 "mktemp" pf_run "$d" "$K" TMPDIR="$PF_T/no-such-dir"
  else pf_undet "U8" "could not build the case"; fi
  if [[ -r "$PF_T/keys_unreadable.sh" ]]; then pf_undet "U9" "chmod 000 does not deny reads here (running as root?)"
  elif d="$(pf_case u9)"; then pf_expect "U9 unreadable credential file" 2 "not readable" pf_run "$d" "$PF_T/keys_unreadable.sh"
  else pf_undet "U9" "could not build the case"; fi
  if d="$(pf_case u10)" && rm -f "$d/go.mod"; then
    pf_expect "U10 module cannot be loaded is rc 2, not a FAIL" 2 "cannot load the module" pf_run "$d" "$K"
  else pf_undet "U10" "could not build the case"; fi
  if d="$(pf_case u11)"; then
    pf_expect "U11 key value with a line break is rc 2, not 'unreachable'" 2 "line break" pf_run "$d" "$PF_T/keys_newline.sh"
    if grep -qF -- "probe exit" <<<"$PF_LAST_OUT"; then pf_bad "U11b the probe still ran with a line-break key (its verdict decided the rc)"
    else pf_ok "U11b the probe was not run with a line-break key"; fi
  else pf_undet "U11" "could not build the case"; fi
  # N1: over the WHOLE request log of every case, the Nvidia key never left.
  if [[ ! -s "$PF_REQLOG" ]]; then pf_undet "N1" "the fake endpoint logged no request at all"
  elif grep -qF -- "$nkey" "$PF_REQLOG"; then pf_bad "N1 the Nvidia key appears in a logged request"
  else pf_ok "N1 no logged request (line, headers, body) carries the Nvidia key"; fi
  echo "prove-failure: $PF_PASS passed, $PF_FAIL failed, $PF_UNDET undetermined"
  [[ "$PF_FAIL" -gt 0 ]] && return 1
  [[ "$PF_UNDET" -gt 0 ]] && return 2
  return 0
}

if [[ "$PROVE" -eq 1 ]]; then
  prove_failure
  exit $?
fi

cd "$REPO_ROOT" || { echo "COULD NOT DETERMINE: cannot enter the module root $REPO_ROOT"; exit 2; }

# The probe's endpoint is HuggingFace's, so the only key it may carry is the
# HuggingFace key. Checked BEFORE the credential file is sourced.
PROBE_PROVIDER="${PROBE_PROVIDER:-HuggingFace}"
if [[ "$PROBE_PROVIDER" != "HuggingFace" ]]; then
  echo "COULD NOT DETERMINE: PROBE_PROVIDER='$PROBE_PROVIDER' has no wired endpoint — the probe"
  echo "  queries huggingface.co, and sending ApiKey_${PROBE_PROVIDER} there would hand that"
  echo "  provider's credential to a third party. NOTHING was measured."
  exit 2
fi
# Frozen before the credential file is sourced, so the file cannot re-point
# the probe at a different provider's key after the check above.
readonly PROBE_PROVIDER

API_KEYS_FILE="${API_KEYS_FILE:-$HOME/api_keys.sh}"
if [[ ! -r "$API_KEYS_FILE" ]]; then
  echo "COULD NOT DETERMINE: #env-no-api-keys-file — $API_KEYS_FILE not readable;"
  echo "  cannot exercise the live probe without operator credentials. NOTHING was"
  echo "  measured: the apikeys wiring is NOT known to work."
  exit 2
fi

# Its stdout is discarded: a credential file that echoes its keys while it is
# sourced would otherwise print them here. Its stderr is kept (e.g. the
# readonly notice when it tries to re-point PROBE_PROVIDER).
# shellcheck source=/dev/null
source "$API_KEYS_FILE" >/dev/null

PROBE_VAR="ApiKey_${PROBE_PROVIDER}"
PROBE_VALUE="${!PROBE_VAR-}"
if [[ -z "$PROBE_VALUE" ]]; then
  echo "COULD NOT DETERMINE: #env-no-${PROBE_PROVIDER}-key — operator has no"
  echo "  ${PROBE_VAR} exported in ${API_KEYS_FILE}. NOTHING was measured: the"
  echo "  apikeys wiring is NOT known to work."
  exit 2
fi
# Only the attribute flags are inspected; the declare -p output (which holds
# the value) is never printed.
if ! [[ "$(declare -p "$PROBE_VAR" 2>/dev/null)" =~ ^declare\ -[A-Za-z]*x ]]; then
  echo "COULD NOT DETERMINE: ${PROBE_VAR} is set in ${API_KEYS_FILE} but not exported,"
  echo "  so it never reaches the probe's environment. NOTHING was measured."
  exit 2
fi
# Go refuses a header value with CR/LF in it as a transport error, which would
# be reported as "endpoint unreachable". Caught here, by name, without
# printing the value.
if [[ "$PROBE_VALUE" == *$'\n'* || "$PROBE_VALUE" == *$'\r'* ]]; then
  echo "COULD NOT DETERMINE: the ${PROBE_VAR} credential value contains a line break — it cannot"
  echo "  be sent as a bearer token. Fix the credential file. NOTHING was measured."
  exit 2
fi
unset PROBE_VALUE

if ! command -v go >/dev/null 2>&1 || ! [[ "$(go version 2>/dev/null)" == "go version "* ]]; then
  echo "COULD NOT DETERMINE: no usable go toolchain on PATH — the probe cannot be built."
  exit 2
fi
# Loading the module is environment (this module's go.mod in use, toolchain
# new enough); compiling the probe against it is the wiring under test.
# `go list -m` alone is not enough: outside any module it still exits 0.
gomod="$(go env GOMOD 2>/dev/null)"
if [[ -z "$gomod" || ! "$gomod" -ef "$REPO_ROOT/go.mod" ]] || ! go list -m >/dev/null 2>&1; then
  echo "COULD NOT DETERMINE: go cannot load the module at $REPO_ROOT (GOMOD='$gomod'):"
  go list -m 2>&1 | head -n 5 | sed 's/^/  /'
  exit 2
fi

PROBE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/llmprovider-apikeys-challenge.XXXXXX")" || PROBE_DIR=""
if [[ -z "$PROBE_DIR" || ! -d "$PROBE_DIR" ]]; then
  echo "COULD NOT DETERMINE: mktemp failed (TMPDIR='${TMPDIR:-}') — nowhere to build the probe."
  exit 2
fi
trap 'rm -rf "$PROBE_DIR"' EXIT

# The probe's exit status is its verdict: 0 evidence printed; 10-15 a finding;
# 20-22 the endpoint could not be measured. (2 is avoided: a Go panic exits 2.)
cat > "$PROBE_DIR/main.go" <<'GO'
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"digital.vasic.llmprovider/pkg/apikeys"
)

func main() {
	provider := os.Args[1]
	snap := apikeys.Scan()
	if !snap.Has(provider) {
		fmt.Printf("FAIL: snapshot did not find ApiKey_%s although it is exported\n", provider)
		os.Exit(10)
	}
	// HuggingFace public models endpoint — no auth required, but we send
	// the token anyway to prove credentials wiring works.
	req, err := http.NewRequest("GET", "https://huggingface.co/api/models?limit=5", nil)
	if err != nil {
		fmt.Printf("FAIL: build request: %v\n", err)
		os.Exit(11)
	}
	req.Header.Set("Authorization", "Bearer "+snap.Get(provider))
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("UNREACHABLE: endpoint unreachable: %v\n", err)
		os.Exit(20)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("UNREACHABLE: read body: %v\n", err)
		os.Exit(21)
	}
	if resp.StatusCode >= 500 {
		fmt.Printf("UNAVAILABLE: endpoint answered status %d — upstream unavailable\n", resp.StatusCode)
		os.Exit(22)
	}
	if resp.StatusCode != 200 {
		fmt.Printf("FAIL: status %d body=%s\n", resp.StatusCode, string(body)[:min(200, len(body))])
		os.Exit(12)
	}
	var models []map[string]any
	if err := json.Unmarshal(body, &models); err != nil {
		fmt.Printf("FAIL: json decode: %v\n", err)
		os.Exit(13)
	}
	if len(models) == 0 {
		fmt.Printf("FAIL: HuggingFace returned empty model list — the key was sent but the upstream list is empty\n")
		os.Exit(14)
	}
	firstID, _ := models[0]["id"].(string)
	if firstID == "" {
		fmt.Printf("FAIL: first model has empty id field — schema may have changed\n")
		os.Exit(15)
	}
	fmt.Printf("OK: discovered %d models from %s, first=%q\n", len(models), provider, firstID)
}
GO

# Built, not `go run`: `go run` turns every non-zero probe exit into 1, which
# erased the difference between a finding and an unreachable endpoint.
if ! build_out="$(go build -o "$PROBE_DIR/probe" "$PROBE_DIR/main.go" 2>&1)"; then
  echo "FAIL: the probe does not build against this module's pkg/apikeys:"
  printf '%s\n' "$build_out" | head -n 20 | sed 's/^/  /'
  exit 1
fi

probe_out="$("$PROBE_DIR/probe" "$PROBE_PROVIDER" 2>&1)"; probe_rc=$?
printf '%s\n' "$probe_out"
case "$probe_rc" in
  0)
    if grep -qE '^OK: discovered [1-9][0-9]* models' <<<"$probe_out"; then
      echo "PASS: apikeys_live_discovery_challenge"
      exit 0
    fi
    echo "FAIL: the probe exited 0 without its evidence line ('OK: discovered N models') —"
    echo "  an exit status alone is not evidence that anything was discovered."
    exit 1 ;;
  10|11|12|13|14|15)
    echo "FAIL: apikeys_live_discovery_challenge (probe exit $probe_rc)"
    exit 1 ;;
  20|21|22)
    echo "COULD NOT DETERMINE: the endpoint could not be measured (probe exit $probe_rc) —"
    echo "  NOTHING about the apikeys wiring was established."
    exit 2 ;;
  *)
    echo "COULD NOT DETERMINE: unrecognised probe exit $probe_rc (a Go panic exits 2, a signal"
    echo "  128+N) — NOTHING about the apikeys wiring was established."
    exit 2 ;;
esac
