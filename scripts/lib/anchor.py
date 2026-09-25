"""anchor.py — whitespace-robust source anchoring for mutation provers.

WHY THIS EXISTS

A mutation prover seeds a real defect into real source, requires the gate to go
RED, restores the source, and requires GREEN again. It locates the seeding site
with an ANCHOR: a literal snippet of the source it expects to find.

The failure mode this module removes is ANCHOR ROT. When an anchor quotes
horizontal whitespace -- gofmt's struct-field column alignment, an indentation
level, the padding inside a composite literal -- then a purely COSMETIC
reformat, triggered by an edit somewhere else entirely, moves the anchor. The
prover can no longer find its own seeding site. It then reports UNDETERMINED,
which is neither a pass nor a failure and is easy to skim past, so a gate that
has quietly stopped proving anything looks approximately fine.

That is not hypothetical. In this tree a gofmt realignment of one struct block
-- caused by an unrelated field gaining a doc comment, which ends the preceding
alignment group -- moved a prover's anchor and took it from rc=1 (RED then
GREEN, working) to rc=2 (0 proven / 1 undetermined).

WHAT THIS MODULE DOES

`resolve()` matches an anchor against a view of the source in which every run of
spaces and tabs is collapsed to a single space. Alignment padding and
indentation therefore cannot break a match, while every non-whitespace
character still must agree exactly. The span it returns is in ORIGINAL
coordinates, so `apply()` splices the real bytes and reformats nothing.

Two properties are deliberate:

  * A match MUST be unique. A collapsed anchor that occurs twice is ambiguous,
    and silently taking the first occurrence would let a prover seed its defect
    somewhere other than where its author meant. Ambiguity is UNDETERMINED.

  * Resolution is separable from mutation. `resolve()` never writes. A prover
    can therefore verify that every one of its anchors still resolves BEFORE it
    claims any verdict -- see `check_anchors()` -- which turns anchor rot from
    something noticed only when someone runs the prover and reads past a 2 into
    something a cheap, fast preflight reports on its own.

EXIT CONVENTION for the CLI below, matching the provers that use it:
    0  every anchor resolved
    2  at least one anchor did not resolve, or resolved ambiguously
       (UNDETERMINED -- never a pass)
"""

from __future__ import annotations

import json
import sys

_HSPACE = " \t"


class AnchorError(Exception):
    """An anchor could not be resolved. Always UNDETERMINED, never a failure."""


def _collapse(text: str) -> tuple[str, list[int]]:
    """Collapse runs of horizontal whitespace, keeping a map back to `text`.

    Returns (collapsed, offsets) where offsets[i] is the index in `text` at
    which collapsed[i] starts, and offsets[len(collapsed)] == len(text). The
    map is what lets a match found in collapsed space be spliced in original
    space without reformatting anything.
    """
    out: list[str] = []
    offsets: list[int] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] in _HSPACE:
            start = i
            while i < n and text[i] in _HSPACE:
                i += 1
            out.append(" ")
            offsets.append(start)
        else:
            out.append(text[i])
            offsets.append(i)
            i += 1
    offsets.append(n)
    return "".join(out), offsets


def resolve(src: str, anchor: str) -> tuple[int, int]:
    """Locate `anchor` in `src`, ignoring horizontal-whitespace RUN LENGTH.

    Returns (start, end) as offsets into `src`. Raises AnchorError if the
    anchor is absent or matches more than once.
    """
    if not anchor:
        raise AnchorError("empty anchor")

    # An exact match is preferred when it exists and is unique: it is the
    # cheapest answer and it keeps behaviour identical for anchors that never
    # quoted alignment in the first place.
    if src.count(anchor) == 1:
        start = src.index(anchor)
        return start, start + len(anchor)

    hay, offsets = _collapse(src)
    needle, _ = _collapse(anchor)
    # A leading/trailing collapse can introduce an edge space that the source
    # side spells differently; compare on the stripped form and re-add nothing.
    hits = []
    pos = hay.find(needle)
    while pos != -1:
        hits.append(pos)
        pos = hay.find(needle, pos + 1)

    if not hits:
        raise AnchorError(
            "anchor not found, even ignoring whitespace run length:\n" + anchor
        )
    if len(hits) > 1:
        raise AnchorError(
            "anchor is AMBIGUOUS -- it matches %d places, so seeding would be a "
            "guess about which one the author meant:\n%s" % (len(hits), anchor)
        )

    start = offsets[hits[0]]
    end = offsets[hits[0] + len(needle)]
    return start, end


def apply(src: str, anchor: str, replacement: str) -> str:
    """Return `src` with the span matched by `anchor` replaced.

    Only the matched span changes; the rest of the file is byte-identical, so a
    prover never reformats the tree it is measuring.
    """
    start, end = resolve(src, anchor)
    return src[:start] + replacement + src[end:]


def check_anchors(pairs) -> list[str]:
    """Verify anchors resolve WITHOUT mutating anything.

    `pairs` is an iterable of (path, anchor). Returns a list of human-readable
    problems; empty means every anchor still resolves. This is the preflight
    that lets a prover refuse to claim a verdict on a stale mutation set.
    """
    problems: list[str] = []
    for path, anchor in pairs:
        try:
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
        except OSError as exc:
            problems.append("%s: cannot read (%s)" % (path, exc))
            continue
        try:
            resolve(src, anchor)
        except AnchorError as exc:
            problems.append("%s: %s" % (path, exc))
    return problems


def _main(argv: list[str]) -> int:
    """CLI: read a JSON array of [path, anchor] pairs on stdin, verify them."""
    if len(argv) > 1 and argv[1] == "--check":
        try:
            pairs = json.load(sys.stdin)
        except ValueError as exc:
            sys.stderr.write("UNDETERMINED: anchor manifest is not JSON: %s\n" % exc)
            return 2
        problems = check_anchors((p[0], p[1]) for p in pairs)
        if problems:
            sys.stderr.write(
                "UNDETERMINED: %d of %d anchors no longer resolve. The mutation "
                "set is STALE; it cannot prove anything until it is repaired.\n"
                % (len(problems), len(pairs))
            )
            for p in problems:
                sys.stderr.write("  - %s\n" % p)
            return 2
        sys.stdout.write("anchors: %d/%d resolve\n" % (len(pairs), len(pairs)))
        return 0
    sys.stderr.write(__doc__ or "")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
