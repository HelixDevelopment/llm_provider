#!/usr/bin/env bash
# llmprovider_describe_challenge.sh — round-292 anti-bluff
# wrapper around the in-process Challenge runner
# (challenges/runner/main.go). Mirrors the HelixDevelopment twin's
# round-276 wrapper.
#
# Two-mode behaviour (CONST-050(A) paired-mutation; §1.1):
#
#   normal:    exits 0 only when the runner exits 0 (all 23
#              invariants pass). Any deviation FAILS.
#
# Exit codes — three-valued, and 2 is NEVER a pass:
#   0  the runner ran and every invariant passed
#   1  a real finding (or, in mutate mode, an undetected mutation)
#   2  COULD NOT DETERMINE — there is no Go toolchain on PATH, so the runner
#      could not be executed at all. This used to print "PASSED (SKIP-OK)" and
#      exit 0: a machine with no Go was reported as a machine whose 23
#      invariants all held.
#
#   mutate:    sets LLMPROVIDER_MUTATE_RUNNER=1 which inverts
#              invariant 3 (circuit.opens_after_failures.*)
#              inside the runner. The runner MUST then exit
#              non-zero; this wrapper rewrites that non-zero
#              exit to 99 (paired-mutation success). If the
#              runner exits 0 under mutation, this wrapper
#              FAILS — proving the runner actually checks what
#              it claims to check, not a metadata-only PASS.
#
# Verbatim 2026-05-19 operator mandate (preserved per
# CONST-049 §11.4.17):
#   "all existing tests and Challenges do work in anti-bluff
#    manner - they MUST confirm that all tested codebase really
#    works as expected! We had been in position that all tests
#    do execute with success and all Challenges as well, but
#    in reality the most of the features does not work and
#    can't be used! This MUST NOT be the case and execution
#    of tests and Challenges MUST guarantee the quality, the
#    completition and full usability by end users of the
#    product!"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-normal}"
echo "=== LLMProvider Describe Challenge (round-292) ==="
echo "  mode=${MODE}"
echo "  module=${MODULE_DIR}"

if ! command -v go >/dev/null 2>&1; then
    echo "COULD NOT DETERMINE: #env-no-go-toolchain — 'go' is not on PATH, so"
    echo "  the in-process Challenge runner could not be executed. NOTHING was"
    echo "  measured: the 23 invariants are NOT known to hold."
    echo "=== Describe Challenge: COULD NOT DETERMINE ==="
    exit 2
fi

cd "${MODULE_DIR}"

case "${MODE}" in
    normal)
        unset LLMPROVIDER_MUTATE_RUNNER
        out="$(go run ./challenges/runner/ 2>&1)"
        rc=$?
        echo "${out}" | tail -30
        if [[ "${rc}" -ne 0 ]]; then
            echo "=== Describe Challenge: FAILED (runner rc=${rc}) ==="
            exit 1
        fi
        # Belt-and-braces: assert the summary line carries
        # FAIL=0 — defends against an accidental exit-0 with
        # buried FAILs.
        if ! echo "${out}" | grep -q "FAIL=0"; then
            echo "=== Describe Challenge: FAILED (no FAIL=0 line) ==="
            exit 1
        fi
        echo "=== Describe Challenge: PASSED ==="
        exit 0
        ;;
    mutate)
        export LLMPROVIDER_MUTATE_RUNNER=1
        out="$(go run ./challenges/runner/ 2>&1)"
        rc=$?
        echo "${out}" | tail -15
        if [[ "${rc}" -eq 0 ]]; then
            echo "=== Describe Challenge: FAILED " \
                 "(mutation undetected — runner exited 0) ==="
            exit 1
        fi
        echo "=== Describe Challenge: MUTATION DETECTED " \
             "(runner rc=${rc} → exit 99) ==="
        exit 99
        ;;
    *)
        echo "usage: $0 [normal|mutate]"
        exit 2
        ;;
esac
