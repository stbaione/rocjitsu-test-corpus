# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The ``rj_waitcheck`` static analyzer, adapted to the corpus.

This is the static case, and the reason the detector contract hands over
artifacts rather than orchestrating a run: waitcheck analyzes a code object and
never executes anything. It needs no simulator, no launcher, and no GPU, so it
works under ``--skip-all-runs``.

Scope. waitcheck models wait-counter hazards in the encoded instruction stream
-- a later instruction using, overwriting, or ordering after an event that was
not waited on strongly enough -- via CFG-aware forward dataflow. That covers
RAW, WAR and WAW arising from missing waits, on registers and on LDS ordering.
It cannot see a cross-wave or cross-workgroup *race*, because establishing that
two waves actually touched the same address needs execution. Those mutants are
therefore out of scope and skip rather than counting as misses.

STATUS: written against the documented CLI contract in
``docs/sphinx/reference/waitcheck.md``. At the time of writing, the ``rj_waitcheck``
binary that document describes is not built anywhere in rocm-systems -- only the
``rocjitsu_waitcheck_target`` / ``rocjitsu_waitcheck_state`` object libraries and
their unit tests exist. This adapter therefore reports itself unavailable unless
the binary is found, and its parsing has not been validated against real output.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from detector_protocol import (
    Detection,
    DetectorError,
    DetectorUnavailable,
    MutantArtifacts,
    capability,
)

name = "waitcheck"

# Wait-counter hazards land on the register or LDS slot whose access was left
# unguarded, across all three ordering kinds. Global memory is absent
# deliberately: a cross-workgroup race is not a property of the instruction
# stream, and neither race kind is listed for the same reason -- establishing
# that two waves touched the same address needs execution.
supports = capability(
    {"vgpr", "agpr", "sgpr", "lds", "scratch"},
    {"RAW", "WAR", "WAW"},
)

needs_execution = False

# Exit codes, per the reference documentation.
_EXIT_CLEAN = 0
_EXIT_USAGE = 1
_EXIT_INPUT_OR_ANALYSIS_ERROR = 2
_EXIT_HAZARDS_FOUND = 4


def _resolve_binary() -> Path:
    """Locate rj_waitcheck, preferring an explicit override."""
    override = os.environ.get("RJ_WAITCHECK")
    if override:
        path = Path(override)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise DetectorUnavailable(f"RJ_WAITCHECK={override} is not an executable file")
        return path
    found = shutil.which("rj_waitcheck")
    if found is None:
        raise DetectorUnavailable(
            "rj_waitcheck not found on PATH; set RJ_WAITCHECK to its location. "
            "Note that no rj_waitcheck target is built in rocm-systems as of writing."
        )
    return Path(found)


def detect(artifacts: MutantArtifacts, run_wrapper: str | None = None) -> Detection:
    """
    Analyze the mutant's code object and count the diagnostics reported.

    ``run_wrapper`` is accepted and ignored: there is nothing to wrap, since
    nothing is executed.
    """
    binary = _resolve_binary()
    jsonl = artifacts.workdir / "waitcheck-diagnostics.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)

    # --diagnostics-jsonl requires --all-code-objects (or --exhaustive), and
    # --no-fail keeps "hazards found" off the exit code so a real analysis
    # failure stays distinguishable from a successful detection.
    argv = [
        str(binary),
        str(artifacts.code_object),
        "--all-code-objects",
        "--diagnostics-jsonl",
        str(jsonl),
        "--no-fail",
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, errors="replace", check=False)

    if proc.returncode in (_EXIT_USAGE, _EXIT_INPUT_OR_ANALYSIS_ERROR):
        # An input or analysis failure analyzes nothing, which would otherwise
        # be indistinguishable from a clean result.
        raise DetectorError(
            f"rj_waitcheck exited {proc.returncode} on {artifacts.code_object.name}: "
            f"{proc.stderr.strip()[:500]}"
        )

    diagnostics = _read_jsonl(jsonl)
    return Detection(
        count=len(diagnostics),
        resources=_resources_named(diagnostics),
        raw=proc.stderr,
        command=" ".join(argv),
    )


def _read_jsonl(path: Path) -> list[dict]:
    """One diagnostic per line; a missing file means none were retained."""
    if not path.is_file():
        return []
    records = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise DetectorError(f"{path}:{lineno}: malformed diagnostic JSON: {error}") from error
    return records


# waitcheck names the counter a diagnostic is about rather than a resource, so
# map from counter family back to the corpus vocabulary where one is present.
_COUNTER_TO_RESOURCE = {
    "loadcnt": "vgpr",
    "storecnt": "vgpr",
    "vmcnt": "vgpr",
    "vscnt": "vgpr",
    "samplecnt": "vgpr",
    "bvhcnt": "vgpr",
    "expcnt": "vgpr",
    "kmcnt": "sgpr",
    "dscnt": "lds",
    "lgkmcnt": "sgpr",
    "tensorcnt": "lds",
}


def _resources_named(diagnostics: list[dict]) -> frozenset[str]:
    """
    Best-effort resource extraction, used only for context in failure messages.

    Scoring does not depend on this: the oracle is the count. Any field the
    records do not carry simply yields an empty set.
    """
    resources: set[str] = set()
    for record in diagnostics:
        counter = record.get("counter") or record.get("wait") or ""
        for key, resource in _COUNTER_TO_RESOURCE.items():
            if key in str(counter):
                resources.add(resource)
                break
    return frozenset(resources)
