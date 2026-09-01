#!/usr/bin/env bash
#
# §1.1 PAIRED-MUTATION PROOF for the pkg/settings override layer.
#
# A gate that has never been seen to fail is not evidence that the defect is
# absent; it is evidence of nothing at all. This script seeds each defect back
# into the tree one at a time, requires the guarding test to come back RED,
# then restores the file and requires GREEN — with the restoration proved by
# checksum, not by intention.
#
# THREE-VALUED, and 2 is never a pass:
#   0  every mutant was caught and every restore was byte-identical
#   1  at least one mutant was NOT caught -- a real finding
#   2  COULD NOT DETERMINE (no toolchain, dirty tree, failed restore)
#
# WHY EACH MUTANT IS SHAPED THE WAY IT IS. A mutant that fails to COMPILE
# proves nothing: the test goes red on a build error rather than on the defect,
# and the gate would look effective while being blind. So every mutant below is
# checked with `go build` BEFORE its test is run, and each one keeps the
# pkg/settings import in use so the compiler cannot redden on an unused import
# instead of on the behaviour.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || { echo "cannot reach module root" >&2; exit 2; }
ROOT=$PWD

RED=0
UNDET=0

log()  { printf '%s\n' "$*"; }
fail() { printf 'FINDING  %s\n' "$*"; RED=1; }
undet(){ printf 'UNDET    %s\n' "$*"; UNDET=1; }

command -v go >/dev/null 2>&1 || { undet "no go toolchain on PATH"; exit 2; }
command -v sha256sum >/dev/null 2>&1 && SUM=sha256sum || {
    command -v shasum >/dev/null 2>&1 && SUM="shasum -a 256"
}
[ -n "${SUM:-}" ] || { undet "no sha256sum or shasum -- restoration cannot be proved"; exit 2; }

digest() { $SUM "$1" | awk '{print $1}'; }

# The precondition is NOT "the working tree is clean" -- this script has to be
# runnable while the change it proves is still uncommitted, and a git-clean gate
# would make it unrunnable exactly when it matters. What must hold instead is
# that every file this script touches ends the run byte-for-byte as it started,
# whatever state that was. Record the manifest up front and re-check it at the
# end; a mismatch is UNDETERMINED, never a pass.
TARGETS=(
    pkg/settings/settings.go
    pkg/providers/xai/xai.go
    pkg/providers/gemini/gemini.go
    pkg/providers/cerebras/cerebras.go
    pkg/providers/generic/generic.go
    pkg/providers/claude/claude.go
    pkg/providers/qwen/qwen.go
)
MANIFEST=$(mktemp) || { undet "mktemp failed"; exit 2; }
for f in "${TARGETS[@]}"; do
    [ -f "$f" ] || { undet "target $f does not exist -- this proof is blind"; exit 2; }
    printf '%s  %s\n' "$(digest "$f")" "$f" >> "$MANIFEST"
done

# seed <file> <search> <replace> <test-package> <test-name> <label>
seed() {
    local file=$1 search=$2 replace=$3 pkg=$4 test=$5 label=$6
    local before after backup
    backup=$(mktemp) || { undet "$label: mktemp failed"; return; }
    cp "$file" "$backup"
    before=$(digest "$file")

    if ! python3 - "$file" "$search" "$replace" <<'PY'
import sys
path, search, replace = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
if s.count(search) != 1:
    sys.exit("anchor appears %d times, expected exactly 1" % s.count(search))
open(path, 'w').write(s.replace(search, replace, 1))
PY
    then
        undet "$label: could not seed the mutant"
        cp "$backup" "$file"; rm -f "$backup"
        return
    fi

    if ! go build ./... >/dev/null 2>&1; then
        undet "$label: the MUTANT DOES NOT COMPILE -- a red test would be a build error, not a catch"
        cp "$backup" "$file"; rm -f "$backup"
        return
    fi

    if go test "$pkg" -run "$test" -count=1 >/dev/null 2>&1; then
        fail "$label: mutant NOT caught -- $test passed with the defect seeded"
    else
        log "RED      $label -- $test caught it"
    fi

    cp "$backup" "$file"; rm -f "$backup"
    after=$(digest "$file")
    if [ "$before" != "$after" ]; then
        undet "$label: restore was NOT byte-identical ($before -> $after)"
        return
    fi

    if go test "$pkg" -run "$test" -count=1 >/dev/null 2>&1; then
        log "GREEN    $label -- restored byte-identically ($before), $test passes"
    else
        undet "$label: file restored byte-identically but $test still fails"
    fi
}

log "=== M1  the env-var PREFIX is renamed ==="
log "    A pin table that recomputed its expectation with settings.Key() would"
log "    move with the subject and stay green through this."
seed pkg/settings/settings.go \
    'const Prefix = "LLMPROVIDER_"' \
    'const Prefix = "LLM_"' \
    ./pkg/settings/ TestKeyProducesTheLiteralOperatorFacingName \
    "settings prefix rename"

log
log "=== M2  xAI's region dispatch hands its endpoint on POSITIONALLY ==="
log "    The original defect: a non-empty argument makes the callee's own"
log "    empty-check unreachable, so the override works everywhere else."
seed pkg/providers/xai/xai.go \
    'return NewProviderWithRetry(apiKey, settings.BaseURL("xai", baseURL), model, region, DefaultRetryConfig())' \
    'return NewProviderWithRetry(apiKey, baseURL, model, region, DefaultRetryConfig())' \
    ./pkg/providers/xai/ TestRegionSelectedEndpointIsStillOverridable \
    "xai positional endpoint bypass"

log
log "=== M3  Gemini's compat constructor writes the timeout into the literal ==="
seed pkg/providers/gemini/gemini.go \
    '	config := GeminiUnifiedConfig{
		APIKey: apiKey,
		// Timeout deliberately left zero — see NewGeminiProvider.
		BaseURL:         baseURL,' \
    '	config := GeminiUnifiedConfig{
		APIKey:          apiKey,
		Timeout:         DefaultUnifiedTimeout,
		BaseURL:         baseURL,' \
    ./pkg/providers/gemini/ TestBackwardCompatibleConstructorsDoNotBypassTheOverride \
    "gemini positional timeout bypass"

log
log "=== M4  a plain adapter's model default is re-frozen ==="
log "    cerebras keeps using settings for BASE_URL and TIMEOUT, so the import"
log "    stays in use and the compiler cannot redden instead of the test."
seed pkg/providers/cerebras/cerebras.go \
    'model = settings.Model("cerebras", CerebrasModel)' \
    'model = CerebrasModel' \
    ./pkg/providers/cerebras/ TestDefaultModelIsOverridableFromTheEnvironment \
    "cerebras frozen model"

log
log "=== M5  the generic shim collapses to ONE key for every backend ==="
log "    This is the regression that looks like a simplification."
seed pkg/providers/generic/generic.go \
    'Timeout: settings.Timeout(name, DefaultTimeout),' \
    'Timeout: settings.Timeout("generic", DefaultTimeout),' \
    ./pkg/providers/generic/ TestTimeoutIsKeyedByTheCallersProviderName \
    "generic per-backend key collapsed"

log
log "=== M6  Claude's two auth paths are merged onto one MODEL key ==="
seed pkg/providers/claude/claude.go \
    'model = settings.Model(SettingsProviderOAuth, ClaudeOAuthModel)' \
    'model = settings.Model(SettingsProvider, ClaudeOAuthModel)' \
    ./pkg/providers/claude/ TestTheTwoAuthPathsHaveSeparateModelKeys \
    "claude oauth model key merged"

log
log "=== M7  Qwen's two auth paths are merged onto one BASE_URL key ==="
seed pkg/providers/qwen/qwen.go \
    'baseURL = settings.BaseURL(SettingsProviderOAuth, QwenOAuthAPIURL)' \
    'baseURL = settings.BaseURL(SettingsProvider, QwenOAuthAPIURL)' \
    ./pkg/providers/qwen/ TestTheTwoAuthPathsHaveSeparateEndpointKeys \
    "qwen oauth endpoint key merged"

log
while read -r want f; do
    got=$(digest "$f")
    [ "$want" = "$got" ] || undet "$f did NOT come back byte-identical ($want -> $got)"
done < "$MANIFEST"
rm -f "$MANIFEST"

log
if [ "$UNDET" -eq 1 ]; then
    log "RESULT: COULD NOT DETERMINE (2) -- not a pass"
    exit 2
fi
if [ "$RED" -eq 1 ]; then
    log "RESULT: FINDING (1) -- a seeded defect went uncaught"
    exit 1
fi
log "RESULT: all mutants caught, all restores byte-identical (0)"
exit 0
