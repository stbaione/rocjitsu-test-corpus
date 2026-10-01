#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Compare what several race detectors found on the same corpus.

Each suite run appends its per-case results to
``.pytest-artifacts/race/reports/<detector>-<target>.jsonl``; this reads any
number of those and prints one row per mutant::

    python corpus/race/scripts/compare.py memory_wait race_detector data_hazard

Every cell carries both the verdict and the hazard count, because the count
says things the verdict cannot: two detectors can both "find" a mutant while
one reports a single hazard and the other reports nine, and a detector whose
count jumps between runs is worth looking at even while it keeps passing.

    ok 3     in scope, found 3 hazards
    MISS 0   in scope, found nothing -- a real gap
    skip     outside this detector's declared scope, not scored
    dirty    the unmutated kernel already reported; says nothing about the mutant
    -        this detector has no result for the case
"""

from __future__ import annotations

import argparse
import sys

from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector_protocol import load_records, record_path  # noqa: E402

CORPUS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = CORPUS_ROOT.parent.parent

_CELL = {
    "detected": "ok",
    "missed": "MISS",
    "skipped": "skip",
    "baseline-dirty": "dirty",
    "exempt": "exempt",
}

# Verdicts that carry no hazard count worth printing.
_COUNTLESS = ("skipped", "exempt")


def render_cell(record: dict | None, timed: bool = False) -> str:
    if record is None:
        return "-"
    verdict = _CELL.get(record["verdict"], record["verdict"])
    if record["verdict"] in ("skipped",):
        return verdict
    count = record.get("mutant_count")
    if count is None:
        count = record.get("baseline_count")

    cell = verdict if count is None else f"{verdict} {count}"
    if not timed:
        return cell

    millis = record.get("mutant_ms")
    if millis is None:
        millis = record.get("baseline_ms")
    return f"{cell:<9}{'-' if millis is None else f'{millis:.0f}ms'}"


def _has_timings(by_detector: dict) -> bool:
    """True when any record carries a measurement, i.e. --benchmark was used."""
    return any(
        record.get("mutant_ms") is not None or record.get("baseline_ms") is not None
        for results in by_detector.values()
        for record in results.values()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detectors", nargs="+", help="detector names to compare")
    parser.add_argument("--target", default="gfx950")
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=REPO_ROOT / ".pytest-artifacts",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show every mutant. By default only rows where verdicts disagree.",
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help=(
            "Also treat differing hazard counts as disagreement. Rarely useful "
            "across detectors, which count different units; useful when "
            "comparing one detector against an earlier run of itself."
        ),
    )
    args = parser.parse_args()

    by_detector: dict[str, dict[str, dict]] = {}
    provenance: dict[str, tuple] = {}
    for detector in args.detectors:
        path = record_path(args.artifact_directory, detector, args.target)
        try:
            records = load_records(path)
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        by_detector[detector] = {r["case"]: r for r in records}
        for record in records:
            provenance[detector] = (
                record.get("manifest_sha256", "")[:12],
                record.get("simulator", ""),
                record.get("simulator_mtime", 0),
            )

    _warn_on_mixed_provenance(provenance)

    # Baselines are a property of the kernel, not of a mutation, and are
    # reported per kernel by their own cases; the matrix is about mutants.
    cases = sorted(
        {
            case
            for results in by_detector.values()
            for case, record in results.items()
            if record["mutant"] != "baseline"
        }
    )
    if not cases:
        print("no mutant results found", file=sys.stderr)
        return 2

    rows = []
    for case in cases:
        cells = [by_detector[d].get(case) for d in args.detectors]
        verdicts = {c["verdict"] for c in cells if c is not None}
        differs = len(verdicts) > 1
        if args.counts:
            # Counts are not comparable across detectors by default: they count
            # different units. memory_wait emits one diagnostic per wave/lane
            # occurrence and reports four figures where race_detector reports
            # one per distinct race, so count-difference would flag every row
            # and hide the verdict disagreements that actually matter.
            counts = {
                c.get("mutant_count")
                for c in cells
                if c is not None and c["verdict"] == "detected"
            }
            differs = differs or len(counts) > 1
        if args.all or differs:
            rows.append((case, cells, differs))

    _print_table(rows, args.detectors, by_detector, cases, args.all)
    return 0


def _print_table(rows, detectors, by_detector, all_cases, show_all) -> None:
    timed = _has_timings(by_detector)
    width_case = max([len(r[0].split(".", 2)[-1]) for r in rows] + [len("mutant")]) + 2
    width_res = 9
    widths = [max(len(d), 18 if timed else 12) + 2 for d in detectors]

    header = f"{'mutant':<{width_case}}{'resource':<{width_res}}"
    header += "".join(f"{d:<{w}}" for d, w in zip(detectors, widths))
    print(header)
    print("-" * len(header))

    for case, cells, differs in rows:
        short = case.split(".", 2)[-1]
        resource = next((c["resource"] for c in cells if c is not None), "?")
        line = f"{short:<{width_case}}{resource:<{width_res}}"
        line += "".join(f"{render_cell(c, timed):<{w}}" for c, w in zip(cells, widths))
        print(line)

    if not rows:
        print("(every detector agreed on every mutant, including hazard counts)")

    print("-" * len(header))
    totals = f"{'':<{width_case}}{'':<{width_res}}"
    for detector, width in zip(detectors, widths):
        results = [
            r for r in by_detector[detector].values() if r["mutant"] != "baseline"
        ]
        scored = [r for r in results if r["verdict"] in ("detected", "missed")]
        found = sum(1 for r in scored if r["verdict"] == "detected")
        cell = f"{found}/{len(scored)}"
        if timed:
            # Summed over the cases that actually ran, so the total is
            # comparable between columns even when they skipped different sets.
            measured = [
                r["mutant_ms"] for r in scored if r.get("mutant_ms") is not None
            ]
            cell = f"{cell:<9}{f'{sum(measured) / 1000:.1f}s' if measured else '-'}"
        totals += f"{cell:<{width}}"
    print(totals)

    if not show_all and rows:
        hidden = len({c for c in all_cases}) - len(rows)
        if hidden > 0:
            print(f"\n{hidden} mutant(s) where all detectors agreed are hidden; pass --all")


def _warn_on_mixed_provenance(provenance: dict[str, tuple]) -> None:
    """
    Say so when the columns did not come from one experiment.

    Comparing detectors usually means comparing branches, since a plugin and a
    core feature often do not exist in the same build. That is a legitimate
    thing to want, but it has to be visible: otherwise a difference caused by
    two different simulators reads as a difference between two detectors.
    """
    manifests = {p[0] for p in provenance.values() if p[0]}
    if len(manifests) > 1:
        print(
            f"WARNING: results span {len(manifests)} different corpus manifests "
            f"({', '.join(sorted(manifests))}); the mutants are not the same set.\n",
            file=sys.stderr,
        )
    simulators = {(p[1], p[2]) for p in provenance.values() if p[1]}
    if len(simulators) > 1:
        print("WARNING: results came from different simulator builds:", file=sys.stderr)
        for detector, (_, binary, mtime) in sorted(provenance.items()):
            if binary:
                when = datetime.fromtimestamp(mtime).isoformat(timespec="minutes")
                print(f"    {detector:<16} {binary}  (built {when})", file=sys.stderr)
        print(
            "  A difference between columns may be a difference between builds.\n",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
