#!/usr/bin/env bash
# upstreams_recipe_origin_challenge.sh — wrong-destination push guard.
#
# WHAT THIS CATCHES, and why it is not cosmetic. Each `upstreams/<name>.sh`
# exports UPSTREAMABLE_REPOSITORY. The shared toolkit's push_all.sh turns every
# such file into a git remote named after the FILE (lowercased) and then runs
# `git push <name>` against it. A recipe therefore does not describe a
# repository — it SELECTS the repository this tree's commits are published to.
#
# HISTORY. On 2026-09-01 this repository's `upstreams/github.sh` named
#   git@github.com:HelixDevelopment/LLMProvider.git
# while a checkout's `origin` named
#   git@github.com:vasic-digital/LLMProvider.git
# and the first version of this guard treated that as a typo, demanding that
# EVERY recipe match origin's organisation. That premise was wrong for this
# repository: it is deliberately published to MORE THAN ONE organisation. On
# 2026-09-25 the operator declared HelixDevelopment/LLMProvider the CANONICAL
# lineage, with vasic-digital GitHub + GitLab as mirrors, and every consuming
# project's .gitmodules names the canonical URL. A single-org assertion would
# therefore FAIL in every honest checkout — a §11.4.201(1) false-positive
# refusal — so the expectation is re-seated (§11.4.120) onto what the checkout
# itself declares: its configured remotes.
#
# Nothing here is hardcoded to this repository. Every expectation is DERIVED
# at run time from `git remote -v` and `git remote get-url origin`, so this
# file is a drop-in for any project's challenges/scripts/ directory
# (§11.4.28(B)).
#
# Assertions, per recipe (2-5 run on every readable recipe that exports a
# parsable URL — exactly four; fewer is a finding: the executed-count floor):
#   1. it exports a non-empty UPSTREAMABLE_REPOSITORY
#   2. the repository it names is one this checkout already has as a configured
#      remote — the fetch or push URL of ANY remote, compared on host + org
#      exactly and repo name case-insensitively. A recipe naming a repository no
#      remote knows would publish this tree somewhere this checkout does not
#      track. Organisation case is not forgiven: two orgs differing only by case
#      are two different accounts on both major forges.
#   3. its REPOSITORY NAME equals origin's, case-insensitively — every mirror is
#      the SAME repository. A forge may hold the canonical name as `Containers`
#      while a recipe writes `containers`; measured on GitHub, both resolve to
#      one repository, so a case difference is a NOTE, not a failure.
#   4. it is executable, like its siblings — a recipe that push_all.sh sources
#      rather than executes still reads as a script to every human and tool
#      that meets it.
#   5. it ends with a newline. `cat` of a recipe without one runs its last line
#      into the next file's shebang, which is exactly how the original wrong
#      line escaped notice.
# And once, for the whole set:
#   6. origin's own URL is named by at least one recipe — the canonical
#      destination is always among the publish targets.
#
# A credential embedded in a remote or recipe URL (https://user:token@host/...)
# is never printed: every URL is shown with its userinfo replaced by '***'.
#
# Usage:
#   bash upstreams_recipe_origin_challenge.sh
#   bash upstreams_recipe_origin_challenge.sh --prove-failure
#
# Exit (three-valued; a FAIL outranks a could-not-determine):
#   0 = every recipe names a repository this checkout tracks, and origin is covered
#   1 = at least one recipe would publish this tree somewhere else, or origin
#       is not covered, or a recipe was judged on fewer than four assertions
#   2 = COULD NOT DETERMINE — an unknown argument, not inside a git repository,
#       no upstreams/ directory or no recipe in it, no origin remote, no
#       parsable remote URLs, or a recipe that cannot be read. NEVER a pass.
#
# --prove-failure (§1.1 paired proof, VSC-370): builds throw-away git
# repositories under $TMPDIR with planted remotes and recipes and runs a copy
# of THIS file in each: a foreign-organisation recipe, a tracked remote of a
# different repository, a recipe exporting nothing, a non-executable recipe, a
# recipe without a trailing newline, origin covered by no recipe, an
# organisation differing only in case, an untracked host and an unparsable URL
# must each exit 1 and name the recipe; clean sets (including a repo-name case
# difference and a URL-style recipe) exit 0; a credential in a URL is never
# printed; a deleted assertion trips the floor (exit 1); no upstreams/, no
# origin, no repository, no recipe, an unknown argument, an unparsable origin
# and an unreadable recipe exit 2, and a finding outranks an unreadable recipe.
# Exits 0 only when every case holds, 1 when any does not, 2 when the proof
# could not judge a case (e.g. as root, where chmod 000 does not deny a read).

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

# ── paired proof (--prove-failure, §1.1, VSC-370) ───────────────────────────
# Builds throw-away git repositories under $TMPDIR, each holding a copy of THIS
# gate at challenges/scripts/ plus planted remotes and upstreams/ recipes, and
# runs that copy. A case holds only when the exit code matches AND the output
# names the defect. Every git call runs with the caller's GIT_* repository
# variables removed, global/system git config ignored (an insteadOf rewrite
# must not change what the gate sees) and GIT_CEILING_DIRECTORIES at the
# scratch root, so a planted repository can never resolve to an enclosing one.
# Exit: 0 every case held, 1 any case did not, 2 the proof could not judge one.
PF_T=""
PF_PASS=0
PF_FAIL=0
PF_UNDET=0
PF_GIT=()
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
# pf_repo NAME ORIGIN — a planted repository (ORIGIN "" = no origin remote)
# with an empty upstreams/ and a copy of this gate; prints its path.
pf_repo() {
  local d="$PF_T/$1"
  mkdir -p "$d/upstreams" "$d/challenges/scripts" || return 1
  "${PF_GIT[@]}" git init -q "$d" || return 1
  if [[ -n "$2" ]]; then "${PF_GIT[@]}" git -C "$d" remote add origin "$2" || return 1; fi
  cp -- "$SELF" "$d/challenges/scripts/$(basename "$SELF")" || return 1
  printf '%s\n' "$d"
}
pf_remote() { "${PF_GIT[@]}" git -C "$1" remote add "$2" "$3"; }
# pf_recipe REPO FILE URL — an executable recipe ending in a newline.
pf_recipe() {
  printf '#!/usr/bin/env bash\nexport UPSTREAMABLE_REPOSITORY=%s\n' "$3" > "$1/upstreams/$2" &&
    chmod 755 "$1/upstreams/$2"
}
pf_gate() { local d="$1"; shift; "${PF_GIT[@]}" bash "$d/challenges/scripts/$(basename "$SELF")" "$@"; }
pf_begin() {
  shopt -u patsub_replacement 2>/dev/null || true
  if ! command -v git >/dev/null 2>&1; then
    echo "prove-failure: cannot determine — git is not on PATH"
    return 2
  fi
  PF_T="$(mktemp -d)" || { echo "prove-failure: cannot determine — mktemp -d failed"; return 2; }
  trap pf_cleanup EXIT
  PF_GIT=(env -u GIT_DIR -u GIT_WORK_TREE -u GIT_INDEX_FILE -u GIT_OBJECT_DIRECTORY
          -u GIT_ALTERNATE_OBJECT_DIRECTORIES -u GIT_COMMON_DIR -u GIT_NAMESPACE -u GIT_PREFIX
          GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_CEILING_DIRECTORIES="$PF_T")
  echo "prove-failure: upstreams recipe gate $SELF"
}
pf_end() {
  echo "prove-failure: $PF_PASS passed, $PF_FAIL failed, $PF_UNDET undetermined"
  [[ "$PF_FAIL" -gt 0 ]] && return 1
  [[ "$PF_UNDET" -gt 0 ]] && return 2
  return 0
}
prove_failure() {
  pf_begin || return $?
  local O='git@github.com:proof-org/Proofrepo.git' M='git@gitlab.com:proof-org/Proofrepo.git'
  local d g_from out rc
  # pf_std NAME — origin + one tracked mirror, one recipe for each: a clean set.
  pf_std() {
    local r
    r="$(pf_repo "$1" "$O")" && pf_remote "$r" mirror "$M" &&
      pf_recipe "$r" github.sh "$O" && pf_recipe "$r" gitlab.sh "$M" && printf '%s\n' "$r"
  }
  # C0 / A1 / A2: clean sets
  if d="$(pf_std c0)"; then pf_expect "C0 clean control: 2 recipes, origin covered" 0 "9 pass, 0 fail" pf_gate "$d"
  else pf_undet "C0" "could not build the repository"; fi
  if d="$(pf_std a1)" && pf_recipe "$d" gitlab.sh 'git@gitlab.com:proof-org/proofrepo.git'; then
    pf_expect "A1 repo-name case difference is a NOTE, not a finding" 0 "differs from origin's 'Proofrepo' only in case" pf_gate "$d"
  else pf_undet "A1" "could not build the repository"; fi
  if d="$(pf_repo a2 "$O")" && pf_recipe "$d" github.sh 'https://github.com/proof-org/Proofrepo.git'; then
    pf_expect "A2 URL-style recipe equals the scp-style origin" 0 "5 pass, 0 fail" pf_gate "$d"
  else pf_undet "A2" "could not build the repository"; fi
  # M1..M9: one planted wrong destination / defect each
  if d="$(pf_std m1)" && pf_recipe "$d" evil.sh 'git@github.com:evil-org/Proofrepo.git'; then
    pf_expect "M1 foreign-organisation recipe" 1 "evil.sh: 'github.com/evil-org/proofrepo' is NOT a configured remote" pf_gate "$d"
  else pf_undet "M1" "could not build the repository"; fi
  if d="$(pf_std m2)" && pf_remote "$d" other 'git@github.com:proof-org/Otherrepo.git' &&
     pf_recipe "$d" other.sh 'git@github.com:proof-org/Otherrepo.git'; then
    pf_expect "M2 tracked remote, different repository" 1 "other.sh: repository 'Otherrepo' != origin's 'Proofrepo'" pf_gate "$d"
  else pf_undet "M2" "could not build the repository"; fi
  if d="$(pf_std m3)" && printf '#!/usr/bin/env bash\n# no export\n' > "$d/upstreams/empty.sh" && chmod 755 "$d/upstreams/empty.sh"; then
    pf_expect "M3 recipe exports nothing" 1 "empty.sh exports no UPSTREAMABLE_REPOSITORY" pf_gate "$d"
  else pf_undet "M3" "could not build the repository"; fi
  if d="$(pf_std m4)" && chmod 644 "$d/upstreams/gitlab.sh"; then
    pf_expect "M4 recipe not executable" 1 "gitlab.sh: not executable" pf_gate "$d"
  else pf_undet "M4" "could not build the repository"; fi
  if d="$(pf_std m5)" && printf '#!/usr/bin/env bash\nexport UPSTREAMABLE_REPOSITORY=%s' "$M" > "$d/upstreams/gitlab.sh"; then
    pf_expect "M5 recipe without a trailing newline" 1 "gitlab.sh: no trailing newline" pf_gate "$d"
  else pf_undet "M5" "could not build the repository"; fi
  if d="$(pf_repo m6 "$O")" && pf_remote "$d" mirror "$M" && pf_recipe "$d" gitlab.sh "$M"; then
    pf_expect "M6 origin named by no recipe" 1 "is named by NO recipe" pf_gate "$d"
  else pf_undet "M6" "could not build the repository"; fi
  if d="$(pf_std m7)" && pf_recipe "$d" caseorg.sh 'git@github.com:Proof-Org/Proofrepo.git'; then
    pf_expect "M7 organisation differing only in case" 1 "caseorg.sh: 'github.com/Proof-Org/proofrepo' is NOT" pf_gate "$d"
  else pf_undet "M7" "could not build the repository"; fi
  if d="$(pf_std m8)" && pf_recipe "$d" host.sh 'git@bitbucket.org:proof-org/Proofrepo.git'; then
    pf_expect "M8 same org/repo on an untracked host" 1 "host.sh: 'bitbucket.org/proof-org/proofrepo' is NOT" pf_gate "$d"
  else pf_undet "M8" "could not build the repository"; fi
  if d="$(pf_std m9)" && pf_recipe "$d" junk.sh 'not-a-url'; then
    pf_expect "M9 unparsable recipe URL" 1 "junk.sh: cannot parse" pf_gate "$d"
  else pf_undet "M9" "could not build the repository"; fi
  # R1: a credential embedded in a remote URL must never be printed (CONST-042)
  local tok='SYNTH-TOKEN-VSC370'
  if d="$(pf_repo r1 "https://proof-user:${tok}@github.com/proof-org/Proofrepo.git")" &&
     pf_recipe "$d" github.sh "https://proof-user:${tok}@github.com/proof-org/Proofrepo.git"; then
    out="$(pf_gate "$d" 2>&1)"; rc=$?
    if [[ "$rc" -eq 0 ]] && ! grep -qF -- "$tok" <<<"$out"; then pf_ok "R1 credential in a remote URL is not printed (rc 0)"
    else pf_bad "R1 credential in a remote URL: want rc 0 and the token absent, got rc $rc, token printed $(grep -cF -- "$tok" <<<"$out") time(s)"; fi
  else pf_undet "R1" "could not build the repository"; fi
  # G1: the gate's own per-recipe executed-count floor, planted in its copy.
  g_from='assert_pass "$name: '; g_from+='executable"'
  if d="$(pf_std g1)" && pf_plant "$d/challenges/scripts/$(basename "$SELF")" "$g_from" ':'; then
    pf_expect "G1 a deleted assertion trips the executed-count floor" 1 "floor" pf_gate "$d"
  else pf_undet "G1" "could not plant the defect"; fi
  # U1..U8: could not determine (and a finding outranks it)
  if d="$(pf_repo u1 "$O")" && rm -rf "$d/upstreams"; then
    pf_expect "U1 no upstreams/ directory" 2 "neither upstreams/ nor Upstreams/" pf_gate "$d"
  else pf_undet "U1" "could not build the repository"; fi
  if d="$(pf_repo u2 "")" && pf_recipe "$d" github.sh "$O"; then
    pf_expect "U2 no origin remote" 2 "no 'origin' remote" pf_gate "$d"
  else pf_undet "U2" "could not build the repository"; fi
  d="$PF_T/u3"
  if mkdir -p "$d/challenges/scripts" "$d/upstreams" && cp -- "$SELF" "$d/challenges/scripts/" && pf_recipe "$d" github.sh "$O"; then
    pf_expect "U3 not inside a git repository" 2 "not inside a git repository" pf_gate "$d"
  else pf_undet "U3" "could not build the directory"; fi
  if d="$(pf_repo u4 "$O")"; then
    pf_expect "U4 upstreams/ holds no recipes" 2 "holds no *.sh recipes" pf_gate "$d"
  else pf_undet "U4" "could not build the repository"; fi
  if d="$(pf_std u5)"; then pf_expect "U5 unknown argument" 2 "unknown argument" pf_gate "$d" --no-such-flag
  else pf_undet "U5" "could not build the repository"; fi
  if d="$(pf_repo u6 'weird-origin')" && pf_recipe "$d" github.sh "$O"; then
    pf_expect "U6 unparsable origin URL" 2 "cannot parse" pf_gate "$d"
  else pf_undet "U6" "could not build the repository"; fi
  if d="$(pf_std u7)" && chmod 000 "$d/upstreams/gitlab.sh"; then
    if [[ -r "$d/upstreams/gitlab.sh" ]]; then
      pf_undet "U7/U8" "chmod 000 does not deny reads here (running as root?)"
    else
      pf_expect "U7 unreadable recipe in a clean set is rc 2, not a FAIL" 2 "gitlab.sh: unreadable" pf_gate "$d"
      if d="$(pf_std u8)" && chmod 000 "$d/upstreams/gitlab.sh" && pf_recipe "$d" evil.sh 'git@github.com:evil-org/Proofrepo.git'; then
        pf_expect "U8 a finding outranks an unreadable recipe" 1 "evil.sh: 'github.com/evil-org/proofrepo' is NOT" pf_gate "$d"
      else pf_undet "U8" "could not build the repository"; fi
      if d="$(pf_std u9)" && chmod 000 "$d/upstreams/github.sh"; then
        pf_expect "U9 origin's only recipe unreadable: coverage is rc 2, not a FAIL" 2 "no READABLE recipe" pf_gate "$d"
      else pf_undet "U9" "could not build the repository"; fi
    fi
  else pf_undet "U7/U8" "could not build the repository"; fi
  pf_end
}

if [[ "$PROVE" -eq 1 ]]; then
  prove_failure
  exit $?
fi

ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)"

PASS_COUNT=0
FAIL_COUNT=0
NOTE_COUNT=0
UNDET_COUNT=0
assert_pass()  { echo "PASS: $*"; PASS_COUNT=$((PASS_COUNT + 1)); }
assert_fail()  { echo "FAIL: $*" >&2; FAIL_COUNT=$((FAIL_COUNT + 1)); }
assert_undet() { echo "UNDETERMINED: $*" >&2; UNDET_COUNT=$((UNDET_COUNT + 1)); }
note()         { echo "NOTE: $*"; NOTE_COUNT=$((NOTE_COUNT + 1)); }

# Shows a URL with any userinfo of a URL-style remote replaced by '***', so a
# token embedded as https://user:token@host/... is never printed. scp-style
# `user@host:path` carries no password and is shown as is.
redact_url() {
  local u="$1" scheme rest auth
  if [[ "$u" == *://* ]]; then
    scheme="${u%%://*}"; rest="${u#*://}"; auth="${rest%%/*}"
    if [[ "$auth" == *@* ]]; then
      printf '%s://***@%s%s' "$scheme" "${auth##*@}" "${rest#"$auth"}"
      return 0
    fi
  fi
  printf '%s' "$u"
}

echo "=== upstreams_recipe_origin_challenge ==="
echo

if [[ -z "$ROOT" ]]; then
  echo "COULD NOT DETERMINE: not inside a git repository." >&2
  exit 2
fi

UPSTREAMS_DIR=""
for cand in "$ROOT/upstreams" "$ROOT/Upstreams"; do
  [[ -d "$cand" ]] && { UPSTREAMS_DIR="$cand"; break; }
done
if [[ -z "$UPSTREAMS_DIR" ]]; then
  echo "COULD NOT DETERMINE: neither upstreams/ nor Upstreams/ exists in $ROOT." >&2
  echo "  With no recipes, push_all.sh walks UP the parent chain looking for" >&2
  echo "  someone else's — which is its own defect, but not one this challenge" >&2
  echo "  can judge from here." >&2
  exit 2
fi

ORIGIN_URL="$(git -C "$ROOT" remote get-url origin 2>/dev/null)"
if [[ -z "$ORIGIN_URL" ]]; then
  echo "COULD NOT DETERMINE: this repository has no 'origin' remote, so there" >&2
  echo "  is nothing to compare the recipes against." >&2
  exit 2
fi

# Parses <host>, <org> and <repo> out of scp-style (git@host:org/repo.git) and
# URL-style (https://host/org/repo.git, ssh://git@host/org/repo.git) remotes
# alike. Echoes "<host>\t<org>\t<repo>"; returns 1 when the shape is not
# recognised.
parse_host_org_repo() {
  local url="$1" host path
  case "$url" in
    *://*)  path="${url#*://}"; host="${path%%/*}"; host="${host##*@}"; path="${path#*/}" ;;
    *:*)    host="${url%%:*}"; host="${host##*@}"; path="${url#*:}" ;;
    *)      return 1 ;;
  esac
  path="${path%.git}"
  path="${path#/}"
  local org="${path%/*}" repo="${path##*/}"
  org="${org##*/}"                       # tolerate nested groups (GitLab)
  [[ -z "$host" || -z "$org" || -z "$repo" || "$org" == "$path" ]] && return 1
  printf '%s\t%s\t%s\n' "$host" "$org" "$repo"
}

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

# Canonical key for "the same repository": host and org exact, repo lowercased.
repo_key() {
  local parts host org repo
  parts="$(parse_host_org_repo "$1")" || return 1
  host="${parts%%$'\t'*}"; parts="${parts#*$'\t'}"
  org="${parts%%$'\t'*}"; repo="${parts##*$'\t'}"
  printf '%s/%s/%s\n' "$(lower "$host")" "$org" "$(lower "$repo")"
}

if ! origin_key="$(repo_key "$ORIGIN_URL")"; then
  echo "COULD NOT DETERMINE: cannot parse a host/org/repo out of origin '$(redact_url "$ORIGIN_URL")'." >&2
  exit 2
fi
ORIGIN_REPO="$(parse_host_org_repo "$ORIGIN_URL" | cut -f3)"

# The set of repositories this checkout tracks: every fetch and push URL of
# every configured remote. This is the ground truth a recipe is judged against.
declare -A KNOWN=()
while read -r _name url _dir; do
  [[ -z "$url" ]] && continue
  key="$(repo_key "$url")" || continue
  KNOWN["$key"]="$url"
done < <(git -C "$ROOT" remote -v 2>/dev/null)
if [[ ${#KNOWN[@]} -eq 0 ]]; then
  echo "COULD NOT DETERMINE: no parsable remote URLs in 'git remote -v'." >&2
  exit 2
fi

echo "origin:   $(redact_url "$ORIGIN_URL")"
echo "expected: every recipe names one of the ${#KNOWN[@]} repositories this checkout tracks;"
echo "          repo name '$ORIGIN_REPO' (case-insensitive); origin covered by >=1 recipe"
for k in "${!KNOWN[@]}"; do echo "          tracked: $k"; done | sort
echo

shopt -s nullglob
recipes=( "$UPSTREAMS_DIR"/*.sh )
shopt -u nullglob
if [[ ${#recipes[@]} -eq 0 ]]; then
  echo "COULD NOT DETERMINE: $UPSTREAMS_DIR holds no *.sh recipes." >&2
  exit 2
fi

ORIGIN_COVERED=0
for recipe in "${recipes[@]}"; do
  name="$(basename "$recipe")"
  echo "[$name]"

  # An unreadable recipe would source as empty and read as "exports nothing" —
  # a FAIL about a file nobody could look at. It is could-not-determine.
  if [[ ! -r "$recipe" || ! -f "$recipe" ]]; then
    assert_undet "$name: unreadable (or not a regular file) — cannot determine where it would push"
    continue
  fi

  # Sourced in a subshell so one recipe cannot leak into the next.
  url="$(bash -c 'unset UPSTREAMABLE_REPOSITORY; . "$1" >/dev/null 2>&1; printf "%s" "${UPSTREAMABLE_REPOSITORY:-}"' _ "$recipe")"

  if [[ -z "$url" ]]; then
    assert_fail "$name exports no UPSTREAMABLE_REPOSITORY"
    continue
  fi
  echo "    -> $(redact_url "$url")"

  if ! key="$(repo_key "$url")"; then
    assert_fail "$name: cannot parse a host/org/repo out of '$(redact_url "$url")'"
    continue
  fi
  repo="$(parse_host_org_repo "$url" | cut -f3)"
  judged_before=$((PASS_COUNT + FAIL_COUNT))

  if [[ -n "${KNOWN[$key]:-}" ]]; then
    assert_pass "$name: names a repository this checkout tracks ($key)"
  else
    assert_fail "$name: '$key' is NOT a configured remote of this checkout — a push through this recipe would publish this tree somewhere this checkout does not track"
  fi

  if [[ "$(lower "$repo")" == "$(lower "$ORIGIN_REPO")" ]]; then
    assert_pass "$name: repository '$repo' matches origin"
    [[ "$repo" != "$ORIGIN_REPO" ]] && \
      note "$name: '$repo' differs from origin's '$ORIGIN_REPO' only in case"
  else
    assert_fail "$name: repository '$repo' != origin's '$ORIGIN_REPO'"
  fi

  [[ "$key" == "$origin_key" ]] && ORIGIN_COVERED=1

  if [[ -x "$recipe" ]]; then
    assert_pass "$name: executable"
  else
    assert_fail "$name: not executable, unlike its siblings"
  fi

  if [[ "$(tail -c1 "$recipe" | od -An -tx1 | tr -d ' \n')" == "0a" ]]; then
    assert_pass "$name: ends with a newline"
  else
    assert_fail "$name: no trailing newline — its last line runs into whatever is concatenated after it"
  fi

  judged=$((PASS_COUNT + FAIL_COUNT - judged_before))
  if [[ "$judged" -ne 4 ]]; then
    assert_fail "$name: judged on $judged of 4 assertions — an assertion was skipped or deleted (executed-count floor)"
  fi
done

echo
echo "[set]"
if [[ $ORIGIN_COVERED -eq 1 ]]; then
  assert_pass "origin ($origin_key) is named by at least one recipe"
elif [[ $UNDET_COUNT -gt 0 ]]; then
  assert_undet "origin ($origin_key) is named by no READABLE recipe — an unreadable one may name it"
else
  assert_fail "origin ($origin_key) is named by NO recipe — the canonical destination would never be pushed by push_all.sh"
fi

echo
echo "=== summary: $PASS_COUNT pass, $FAIL_COUNT fail, $NOTE_COUNT note, $UNDET_COUNT undetermined ==="
if [[ $FAIL_COUNT -gt 0 ]]; then
  exit 1
fi
if [[ $UNDET_COUNT -gt 0 ]]; then
  echo "COULD NOT DETERMINE: $UNDET_COUNT recipe(s) could not be judged — THIS IS NOT A PASS." >&2
  exit 2
fi
exit 0
