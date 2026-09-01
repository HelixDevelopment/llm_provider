#!/usr/bin/env bash
# prove-offline-discovery.sh — the §1.1 paired-mutation prover for the provider
# capability tests that used to be availability probes.
#
# WHAT IT PROVES
#
# Six adapters (novita, sarvam, nvidia, sambanova, openrouter, venice) each
# constructed a provider on the PRODUCTION base URL and asserted
# `NotEmpty(SupportedModels)` against LIVE discovery. That is not a unit test:
# it asserts a third party is up and reachable from whatever machine runs it.
# They were green only on a host with egress; with HTTP(S)_PROXY pointed at a
# closed port all six failed.
#
# The assertion also demanded the OPPOSITE of the module's documented contract.
# pkg/discovery returns nil when live discovery is unreachable (CONST-036 — no
# hardcoded fallback; LLMsVerifier is the single source of truth for the model
# catalogue), so a CORRECT implementation could only ever fail it.
#
# The tests now run against httptest fixtures. This prover exists because a test
# that has never been observed FAILING is not known to test anything: each
# mutation below seeds a real defect into real source, requires the suite to go
# RED, restores the source byte-identically, and requires GREEN again.
#
# THE RUN IS DELIBERATELY OFFLINE. Every go-test invocation here is made with
# HTTP(S)_PROXY pointed at a closed port. That is not incidental: it is the
# environment in which the original defect reproduced, and it is the only
# environment in which these mutations are DETERMINISTIC — with egress, a
# re-pinned endpoint would reach the real provider and might answer, so a
# surviving mutation could not be distinguished from a lucky one. Loopback is
# never proxied by Go's http.ProxyFromEnvironment, so the fixtures still serve.
#
# THREE-VALUED EXIT, and rc=2 is never a pass:
#   0  every mutation went red then green — the tests are proven
#   1  a real problem: a mutation the suite did not catch, or a suite that
#      stayed red after the source was restored
#   2  could not determine: no toolchain, a mutant that will not compile, a
#      stale anchor, a source file that moved, an interrupted run, or zero
#      mutations executed
#
# Usage:
#   bash scripts/prove-offline-discovery.sh                 # every mutation
#   bash scripts/prove-offline-discovery.sh pin-novita      # one
#   bash scripts/prove-offline-discovery.sh --list
#   bash scripts/prove-offline-discovery.sh --check-anchors  # preflight only

set -uo pipefail

MODULE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MODULE" || { echo "FATAL: cannot enter $MODULE" >&2; exit 2; }

ANCHOR_LIB="$MODULE/scripts/lib/anchor.py"

# The closed port every offline invocation proxies through. Port 9 (discard) is
# conventionally not listening; the assertion that it is closed is made below
# rather than assumed.
readonly DEAD_PROXY="http://127.0.0.1:9"

OK=0; PROBLEM=0; UNDET=0
BACKUP_DIR=""

log()   { printf '%s\n' "$*"; }
red()   { printf '  \033[31m%s\033[0m\n' "$*"; }
green() { printf '  \033[32m%s\033[0m\n' "$*"; }

restore() {
    [ -n "$BACKUP_DIR" ] && [ -d "$BACKUP_DIR" ] || return 0
    ( cd "$BACKUP_DIR" && find . -type f -print0 ) | while IFS= read -r -d '' f; do
        cp -p "$BACKUP_DIR/$f" "$MODULE/$f"
    done
}
cleanup() {
    restore
    [ -n "$BACKUP_DIR" ] && rm -rf "$BACKUP_DIR"
}
trap cleanup EXIT
trap 'echo "INTERRUPTED — source restored" >&2; exit 2' INT TERM

# ---------------------------------------------------------------------------
# The mutation set.
#
# Each row is  id | file | test-regexp | package | description
# with the anchor and replacement supplied by mutation_anchor / mutation_repl.
#
# Anchors are matched through scripts/lib/anchor.py, which ignores the RUN
# LENGTH of horizontal whitespace. That matters here because several of these
# anchors sit inside gofmt-aligned struct literals: `ModelsEndpoint:` is padded
# relative to its siblings, and adding any longer field name to the literal
# makes gofmt re-pad the whole block. An anchor that quoted that padding would
# be disarmed by a purely cosmetic reformat, and the prover would then report
# UNDETERMINED — a verdict that reads as neither pass nor fail.
# ---------------------------------------------------------------------------

mutation_ids() {
    printf '%s\n' \
        pin-novita pin-sarvam pin-nvidia pin-sambanova pin-openrouter pin-venice \
        tier3-fallback probe-novita probe-sarvam
}

mutation_file() {
    case "$1" in
        pin-novita)     echo pkg/providers/novita/novita.go ;;
        pin-sarvam)     echo pkg/providers/sarvam/sarvam.go ;;
        pin-nvidia)     echo pkg/providers/nvidia/nvidia.go ;;
        pin-sambanova)  echo pkg/providers/sambanova/sambanova.go ;;
        pin-openrouter) echo pkg/providers/openrouter/openrouter.go ;;
        pin-venice)     echo pkg/providers/venice/venice.go ;;
        tier3-fallback) echo pkg/discovery/discovery.go ;;
        probe-novita)   echo pkg/providers/novita/novita_test.go ;;
        probe-sarvam)   echo pkg/providers/sarvam/sarvam_test.go ;;
    esac
}

mutation_pkg() {
    case "$1" in
        pin-novita|probe-novita) echo ./pkg/providers/novita/ ;;
        pin-sarvam|probe-sarvam) echo ./pkg/providers/sarvam/ ;;
        pin-nvidia)              echo ./pkg/providers/nvidia/ ;;
        pin-sambanova)           echo ./pkg/providers/sambanova/ ;;
        pin-openrouter)          echo ./pkg/providers/openrouter/ ;;
        pin-venice)              echo ./pkg/providers/venice/ ;;
        # The CONST-036 mutation is a change to shared discovery, so it is
        # answered by every adapter's honest-unavailability test at once.
        tier3-fallback)          echo ./pkg/providers/novita/ ;;
    esac
}

mutation_test() {
    case "$1" in
        pin-openrouter) echo '^TestSimpleOpenRouterProvider_GetCapabilities$' ;;
        tier3-fallback) echo '^TestGetCapabilitiesDiscoveryUnavailable$' ;;
        *)              echo '^TestGetCapabilities$' ;;
    esac
}

mutation_desc() {
    case "$1" in
        pin-*)          echo 're-pin the discovery endpoint to the production constant, so no fixture can reach it' ;;
        tier3-fallback) echo 'serve the deprecated FallbackModels catalogue when live discovery is unreachable (the CONST-036 defect)' ;;
        probe-*)        echo 'restore the availability probe: construct on the production base URL and assert NotEmpty' ;;
    esac
}

mutation_anchor() {
    case "$1" in
        pin-novita)     printf '%s' 'ModelsEndpoint: p.modelsURL(),
		ModelsDevID:    "novita",' ;;
        pin-sarvam)     printf '%s' 'ModelsEndpoint: p.modelsURL(),
		ModelsDevID:    "sarvam",' ;;
        pin-nvidia)     printf '%s' 'ModelsEndpoint: p.modelsURL(),
		ModelsDevID:    "nvidia",' ;;
        pin-sambanova)  printf '%s' 'ModelsEndpoint: p.modelsURL(),
		ModelsDevID:    "sambanova",' ;;
        pin-openrouter) printf '%s' 'ModelsEndpoint: baseURL + "/models",' ;;
        pin-venice)     printf '%s' 'ModelsEndpoint: modelsURL,
		ModelsDevID:    "venice",' ;;
        tier3-fallback) printf '%s' 'return nil
}

// GetCachedModels returns the currently cached models without triggering discovery.' ;;
        probe-novita)  printf '%s' 'srv := modelsFixture(t,
		"meta-llama/llama-3-8b-instruct",
		"mistralai/mistral-7b-instruct",
		"baai/bge-m3-embedding",
	)
	provider := NewNovitaProvider("test-key", srv.URL+"/v3/openai/chat/completions", "")' ;;
        probe-sarvam)  printf '%s' 'srv := modelsFixture(t, "sarvam-m", "sarvam-2b", "sarvam-text-embedding")
	provider := NewSarvamProvider("test-key", srv.URL+"/v1/chat/completions", "")' ;;
    esac
}

mutation_repl() {
    case "$1" in
        pin-novita)     printf '%s' 'ModelsEndpoint: NovitaModelsURL, // MUTANT
		ModelsDevID:    "novita",' ;;
        pin-sarvam)     printf '%s' 'ModelsEndpoint: SarvamModelsURL, // MUTANT
		ModelsDevID:    "sarvam",' ;;
        pin-nvidia)     printf '%s' 'ModelsEndpoint: NvidiaModelsURL, // MUTANT
		ModelsDevID:    "nvidia",' ;;
        pin-sambanova)  printf '%s' 'ModelsEndpoint: SambaNovaModelsURL, // MUTANT
		ModelsDevID:    "sambanova",' ;;
        pin-openrouter) printf '%s' 'ModelsEndpoint: defaultBaseURL + "/models", // MUTANT' ;;
        pin-venice)     printf '%s' 'ModelsEndpoint: VeniceModelsURL, // MUTANT
		ModelsDevID:    "venice",' ;;
        tier3-fallback) printf '%s' 'return d.config.FallbackModels // MUTANT
}

// GetCachedModels returns the currently cached models without triggering discovery.' ;;
        probe-novita)  printf '%s' 'provider := NewNovitaProvider("test-key", "", "") // MUTANT' ;;
        probe-sarvam)  printf '%s' 'provider := NewSarvamProvider("test-key", "", "") // MUTANT' ;;
    esac
}

# ---------------------------------------------------------------------------
# Anchor preflight — run BEFORE any verdict is claimed.
# ---------------------------------------------------------------------------

anchor_manifest() {
    local id file
    printf '['
    local first=1
    while IFS= read -r id; do
        file="$(mutation_file "$id")"
        [ "$first" -eq 1 ] || printf ','
        first=0
        python3 -c 'import json,sys; print(json.dumps([sys.argv[1], sys.argv[2]]), end="")' \
            "$file" "$(mutation_anchor "$id")"
    done < <(mutation_ids)
    printf ']'
}

check_anchors() {
    local out rc
    out="$(anchor_manifest | python3 "$ANCHOR_LIB" --check 2>&1)"
    rc=$?
    printf '%s\n' "$out"
    return $rc
}

# ---------------------------------------------------------------------------

apply_mutation() {
    local id="$1" file="$2"
    ANCHOR_LIB="$ANCHOR_LIB" python3 - "$file" "$(mutation_anchor "$id")" "$(mutation_repl "$id")" <<'PY'
import os, sys, importlib.util
spec = importlib.util.spec_from_file_location("anchor", os.environ["ANCHOR_LIB"])
anchor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(anchor)

path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path, encoding="utf-8").read()
try:
    out = anchor.apply(src, old, new)
except anchor.AnchorError as exc:
    sys.stderr.write("%s\n" % exc)
    sys.exit(2)
open(path, "w", encoding="utf-8").write(out)
PY
}

offline_go() {
    HTTP_PROXY="$DEAD_PROXY" HTTPS_PROXY="$DEAD_PROXY" \
    http_proxy="$DEAD_PROXY" https_proxy="$DEAD_PROXY" \
        "$@"
}

compiles() { go build ./... >/dev/null 2>&1; }

run_target() {
    local id="$1"
    offline_go go test "$(mutation_pkg "$id")" -run "$(mutation_test "$id")" \
        -count=1 >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Environment preconditions. Each is MEASURED; none is assumed.
# ---------------------------------------------------------------------------

command -v go       >/dev/null 2>&1 || { echo "UNDETERMINED: no go toolchain" >&2; exit 2; }
command -v python3  >/dev/null 2>&1 || { echo "UNDETERMINED: no python3" >&2; exit 2; }
[ -f "$ANCHOR_LIB" ] || { echo "UNDETERMINED: missing $ANCHOR_LIB" >&2; exit 2; }

# The offline half of this proof is worthless if the "dead" proxy is actually
# alive, so prove it is refused rather than trusting the port number.
if (exec 3<>/dev/tcp/127.0.0.1/9) 2>/dev/null; then
    echo "UNDETERMINED: something is LISTENING on 127.0.0.1:9, so the offline" >&2
    echo "              half of this proof would not actually be offline." >&2
    exit 2
fi

case "${1:-}" in
    --list)
        while IFS= read -r id; do
            printf '%-16s %-44s %s\n' "$id" "$(mutation_file "$id")" "$(mutation_desc "$id")"
        done < <(mutation_ids)
        exit 0
        ;;
    --check-anchors)
        # The preflight on its own: cheap, mutates nothing, and answers the one
        # question a stale prover cannot answer about itself.
        if check_anchors; then
            exit 0
        fi
        exit 2
        ;;
esac

SELECT="${1:-}"

log "prove-offline-discovery — §1.1 paired mutation, run with egress blocked"
log ""

# PREFLIGHT. A prover must not claim a verdict on a mutation set it cannot even
# locate. Without this, a rotted anchor surfaces only as a per-mutation
# UNDETERMINED buried in the run — which reads as neither pass nor fail.
log "preflight: resolving every anchor against the current source"
if ! check_anchors; then
    log ""
    red "ABORT (rc=2): the mutation set is STALE. Repair the anchors before"
    red "              treating any verdict from this prover as meaningful."
    exit 2
fi
log ""

BACKUP_DIR="$(mktemp -d)" || { echo "UNDETERMINED: mktemp failed" >&2; exit 2; }
while IFS= read -r id; do
    f="$(mutation_file "$id")"
    [ -f "$MODULE/$f" ] || { echo "UNDETERMINED: missing source $f" >&2; exit 2; }
    mkdir -p "$BACKUP_DIR/$(dirname "$f")"
    cp -p "$MODULE/$f" "$BACKUP_DIR/$f"
done < <(mutation_ids)

# CONTROL. If the suite is not green BEFORE any mutation, every RED below is
# unattributable and the run proves nothing.
log "control: the target tests must be GREEN offline before anything is seeded"
control_failed=0
while IFS= read -r id; do
    [ -n "$SELECT" ] && [ "$SELECT" != "$id" ] && continue
    if ! run_target "$id"; then
        red "control FAILED for $id — the suite is already red offline"
        control_failed=1
    fi
done < <(mutation_ids)
if [ "$control_failed" -ne 0 ]; then
    red "ABORT (rc=2): control is not green, so no mutation result is attributable"
    exit 2
fi
green "control green"
log ""

ran=0
while IFS= read -r id; do
    [ -n "$SELECT" ] && [ "$SELECT" != "$id" ] && continue
    ran=$((ran + 1))
    file="$(mutation_file "$id")"
    printf '%-16s %s\n' "$id" "$(mutation_desc "$id")"

    if ! apply_mutation "$id" "$file"; then
        red "UNDETERMINED: the mutation could not be applied to $file"
        restore
        UNDET=$((UNDET + 1))
        continue
    fi

    # A mutant that does not compile reddens on a build error, not on the defect
    # it was meant to seed. That is a vacuous proof, so it is UNDETERMINED.
    if ! compiles; then
        red "UNDETERMINED: the mutant does not compile — it would redden on a"
        red "              build error rather than on the seeded defect"
        restore
        UNDET=$((UNDET + 1))
        continue
    fi

    if run_target "$id"; then
        red "SURVIVED: the suite stayed GREEN with the defect in place."
        red "          That is a finding about the TEST, not about the mutation."
        PROBLEM=$((PROBLEM + 1))
    else
        green "RED with the defect seeded"
        OK=$((OK + 1))
    fi

    restore

    # Byte-identical restoration, proved rather than assumed.
    if ! cmp -s "$MODULE/$file" "$BACKUP_DIR/$file"; then
        red "UNDETERMINED: $file was NOT restored byte-identically"
        UNDET=$((UNDET + 1))
        continue
    fi

    if ! run_target "$id"; then
        red "STAYED RED after restore — the tree is not back to its pristine state"
        PROBLEM=$((PROBLEM + 1))
    else
        green "GREEN again after restore (checksum-identical)"
    fi
done < <(mutation_ids)

log ""
log "proven: $OK   problems: $PROBLEM   undetermined: $UNDET"

if [ "$ran" -eq 0 ]; then
    red "no mutation was executed — that is not a pass"
    exit 2
fi
[ "$PROBLEM" -gt 0 ] && exit 1
[ "$UNDET"   -gt 0 ] && exit 2
exit 0
