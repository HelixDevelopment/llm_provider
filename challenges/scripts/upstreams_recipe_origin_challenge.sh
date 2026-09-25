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
# Assertions, per recipe:
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
# Exit:
#   0 = every recipe names a repository this checkout tracks, and origin is covered
#   1 = at least one recipe would publish this tree somewhere else, or origin
#       is not covered
#   2 = COULD NOT DETERMINE — no upstreams/ directory, no origin remote, no
#       parsable remote URLs. NEVER a pass.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)"

PASS_COUNT=0
FAIL_COUNT=0
NOTE_COUNT=0
assert_pass() { echo "PASS: $*"; PASS_COUNT=$((PASS_COUNT + 1)); }
assert_fail() { echo "FAIL: $*" >&2; FAIL_COUNT=$((FAIL_COUNT + 1)); }
note()        { echo "NOTE: $*"; NOTE_COUNT=$((NOTE_COUNT + 1)); }

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
  echo "COULD NOT DETERMINE: cannot parse a host/org/repo out of origin '$ORIGIN_URL'." >&2
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

echo "origin:   $ORIGIN_URL"
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

  # Sourced in a subshell so one recipe cannot leak into the next.
  url="$(bash -c 'unset UPSTREAMABLE_REPOSITORY; . "$1" >/dev/null 2>&1; printf "%s" "${UPSTREAMABLE_REPOSITORY:-}"' _ "$recipe")"

  if [[ -z "$url" ]]; then
    assert_fail "$name exports no UPSTREAMABLE_REPOSITORY"
    continue
  fi
  echo "    -> $url"

  if ! key="$(repo_key "$url")"; then
    assert_fail "$name: cannot parse a host/org/repo out of '$url'"
    continue
  fi
  repo="$(parse_host_org_repo "$url" | cut -f3)"

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
done

echo
echo "[set]"
if [[ $ORIGIN_COVERED -eq 1 ]]; then
  assert_pass "origin ($origin_key) is named by at least one recipe"
else
  assert_fail "origin ($origin_key) is named by NO recipe — the canonical destination would never be pushed by push_all.sh"
fi

echo
echo "=== summary: $PASS_COUNT pass, $FAIL_COUNT fail, $NOTE_COUNT note ==="
[[ $FAIL_COUNT -eq 0 ]] && exit 0 || exit 1
