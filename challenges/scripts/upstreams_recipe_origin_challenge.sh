#!/usr/bin/env bash
# upstreams_recipe_origin_challenge.sh — wrong-organisation push guard.
#
# WHAT THIS CATCHES, and why it is not cosmetic. Each `upstreams/<name>.sh`
# exports UPSTREAMABLE_REPOSITORY. The shared toolkit's push_all.sh turns every
# such file into a git remote named after the FILE (lowercased) and then runs
# `git push <name>` against it. A recipe therefore does not describe a
# repository — it SELECTS the repository this tree's commits are published to.
#
# On 2026-09-01 this repository's `upstreams/github.sh` named
#   git@github.com:HelixDevelopment/LLMProvider.git
# while its `origin`, and the superproject's .gitmodules entry, both named
#   git@github.com:vasic-digital/LLMProvider.git
# Both repositories exist and they are NOT the same repository. Measured with
# `git ls-remote --symref` on 2026-09-01, both default to `master` and their
# masters differ (60e6d18 vs 0446b8f), while their `main` refs happen to agree.
# A wrapper-driven push would therefore have aimed this tree's history at a
# divergent twin. Re-derive:
#   git ls-remote --symref git@github.com:vasic-digital/LLMProvider.git
#   git ls-remote --symref git@github.com:HelixDevelopment/LLMProvider.git
#
# Nothing here is hardcoded to this repository. Every expectation is DERIVED
# from `git remote get-url origin` at run time, so this file is a drop-in for
# any project's challenges/scripts/ directory (§11.4.28(B)).
#
# Assertions, per recipe:
#   1. it exports a non-empty UPSTREAMABLE_REPOSITORY
#   2. its ORGANISATION equals origin's organisation, exactly. This is the
#      wrong-org defect above; case is not forgiven here because two orgs
#      differing only by case are two different accounts on both major forges.
#   3. its REPOSITORY NAME equals origin's, case-insensitively. A forge may
#      hold the canonical name as `Containers` while a sibling recipe writes
#      `containers`; measured on GitHub, both resolve to one repository, so a
#      case difference is reported as a NOTE and is not a failure.
#   4. it is executable, like its siblings — a recipe that push_all.sh sources
#      rather than executes still reads as a script to every human and tool
#      that meets it.
#   5. it ends with a newline. `cat` of a recipe without one runs its last line
#      into the next file's shebang, which is exactly how the wrong-org line
#      escaped notice.
#
# Exit:
#   0 = every recipe names this repository
#   1 = at least one recipe would publish this tree somewhere else
#   2 = COULD NOT DETERMINE — no upstreams/ directory, no origin remote, or
#       an origin URL this challenge cannot parse. NEVER a pass.

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

# Parses <org> and <repo> out of scp-style (git@host:org/repo.git) and URL-style
# (https://host/org/repo.git, ssh://git@host/org/repo.git) remotes alike.
# Echoes "<org>\t<repo>"; returns 1 when the shape is not recognised.
parse_org_repo() {
  local url="$1" path
  case "$url" in
    *://*)  path="${url#*://}"; path="${path#*/}" ;;
    *:*)    path="${url#*:}" ;;
    *)      return 1 ;;
  esac
  path="${path%.git}"
  path="${path#/}"
  local org="${path%/*}" repo="${path##*/}"
  org="${org##*/}"                       # tolerate nested groups (GitLab)
  [[ -z "$org" || -z "$repo" || "$org" == "$path" ]] && return 1
  printf '%s\t%s\n' "$org" "$repo"
}

if ! origin_parts="$(parse_org_repo "$ORIGIN_URL")"; then
  echo "COULD NOT DETERMINE: cannot parse an org/repo out of origin '$ORIGIN_URL'." >&2
  exit 2
fi
ORIGIN_ORG="${origin_parts%%$'\t'*}"
ORIGIN_REPO="${origin_parts##*$'\t'}"

echo "origin:   $ORIGIN_URL"
echo "expected: org='$ORIGIN_ORG'  repo='$ORIGIN_REPO' (repo compared case-insensitively)"
echo

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

shopt -s nullglob
recipes=( "$UPSTREAMS_DIR"/*.sh )
shopt -u nullglob
if [[ ${#recipes[@]} -eq 0 ]]; then
  echo "COULD NOT DETERMINE: $UPSTREAMS_DIR holds no *.sh recipes." >&2
  exit 2
fi

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

  if ! parts="$(parse_org_repo "$url")"; then
    assert_fail "$name: cannot parse an org/repo out of '$url'"
    continue
  fi
  org="${parts%%$'\t'*}"
  repo="${parts##*$'\t'}"

  if [[ "$org" == "$ORIGIN_ORG" ]]; then
    assert_pass "$name: organisation '$org' matches origin"
  else
    assert_fail "$name: organisation '$org' != origin's '$ORIGIN_ORG' — a push through this recipe would publish this tree to a DIFFERENT account"
  fi

  if [[ "$(lower "$repo")" == "$(lower "$ORIGIN_REPO")" ]]; then
    assert_pass "$name: repository '$repo' matches origin"
    [[ "$repo" != "$ORIGIN_REPO" ]] && \
      note "$name: '$repo' differs from origin's '$ORIGIN_REPO' only in case"
  else
    assert_fail "$name: repository '$repo' != origin's '$ORIGIN_REPO'"
  fi

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
echo "=== summary: $PASS_COUNT pass, $FAIL_COUNT fail, $NOTE_COUNT note ==="
[[ $FAIL_COUNT -eq 0 ]] && exit 0 || exit 1
