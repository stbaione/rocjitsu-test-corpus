# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The rocjitsu ``race_detector`` plugin, adapted to the corpus.

Dynamic, and the narrowest scope of the three, which is what makes it the
useful proof that capability scoping works: it must skip a good part of the
corpus rather than be recorded as missing it.

Scope, from the plugin's own sources:

* ``wave_race_state.h`` -- "Per-wave race detection state. Owns VGPR event
  lists ... provides VGPR race checking." Register hazards are per wave.
* ``race_detector.h`` -- "Workgroup-level race detection state. Owns the event
  registry, live LDS event lists, per-byte counters, and per-wave
  WaveRaceStates." LDS is checked across the waves of one workgroup.
* ``MemoryEventType`` (``core/common_register.h``) is GLOBAL_TO_VGPR,
  VGPR_TO_GLOBAL, LDS_TO_VGPR, VGPR_TO_LDS, GLOBAL_TO_LDS. Global memory
  appears only as a *source* or *destination of a transfer*; there is no
  cross-workgroup shadow of global addresses, so a global race is out of scope.
* ``RaceDetector::validateWrite`` carries "TODO(newling): WAW detection (write
  vs outstanding writes) is not implemented" -- on the **LDS** path. Register
  WAW is implemented and covered by its own tests (``sgpr_waw_load_then_mov``,
  ``waw_global_load_then_alu``, ``f64_vgpr_waw``).

Output is free-form text written to a sink with ``std::format``, with no
machine-readable form, so this adapter scrapes prose. That is brittle by
construction; see the module's summary-line contract below.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

from detector_protocol import (
    Detection,
    DetectorError,
    DetectorUnavailable,
    MutantArtifacts,
    capability,
)

name = "race_detector"

# Registers per wave and LDS per workgroup, across all three ordering kinds --
# except LDS WAW, which the plugin does not implement. Global memory is absent
# entirely: it has no cross-workgroup address shadow.
supports = capability(
    {"vgpr", "agpr", "sgpr", "lds"},
    {"RAW", "WAR", "WAW"},
    without={("lds", "WAW")},  # TODO(newling) in RaceDetector::validateWrite
)

needs_execution = True

_TIMEOUT_SECONDS = 240

# Each detected race is one `RACE ... END_RACE` record, matching the format
# tests/race-detector/race_log_expectation.hpp parses.
#
# Counting records rather than reading the plugin's summary banner is
# deliberate: the banner is written from ~RaceDetectorPlugin, and by then the
# file sink is gone, so it never lands in race.log. rocjitsu's own passing
# RaceTest.gfx950_lds_cross_wave_race log contains one RACE record and zero
# occurrences of the banner.
_RACE_RECORD_RE = re.compile(r"^RACE\s+kernel=", re.MULTILINE)

# Written by the plugin on every kernel dispatch, so it is proof the plugin was
# loaded and observed something. Without it, an empty log means the plugin
# never ran -- which must not be read as "no races".
_PLUGIN_ALIVE_RE = re.compile(r"^\[rocjitsu\] Kernel dispatch:", re.MULTILINE)

# Only reachable when the sink is the console, but honoured when present.
_SUMMARY_COUNT_RE = re.compile(r"^\s*(\d+)\s+race\(s\) detected\s*$", re.MULTILINE)
_SUMMARY_NONE_RE = re.compile(r"^\s*No races detected\.\s*$", re.MULTILINE)


def detect(artifacts: MutantArtifacts, run_wrapper: str | None) -> Detection:
    if artifacts.executable is None:
        raise DetectorUnavailable(
            "race_detector needs a linked executable; it cannot run under --skip-all-runs"
        )
    if not run_wrapper:
        raise DetectorUnavailable(
            "race_detector needs --run-wrapper, e.g. "
            "--run-wrapper 'rocjitsu --config /path/to/gfx950.json --'"
        )

    wrapper = shlex.split(run_wrapper)
    base_config = _config_from_wrapper(wrapper)
    config_path = _derive_config(base_config, artifacts.workdir)
    argv = _substitute_config(wrapper, config_path) + [str(artifacts.executable)]

    # The sink appends, so a log left by an earlier run of this same mutant
    # would be counted again on top of this run's findings.
    race_log(artifacts.workdir).unlink(missing_ok=True)

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise DetectorError(
            f"race_detector run timed out after {_TIMEOUT_SECONDS}s: {shlex.join(argv)}"
        ) from error

    text = _sink_text(artifacts.workdir, proc.stdout, proc.stderr)
    return Detection(
        count=_count_races(text, argv, proc.returncode),
        raw=text,
        command=shlex.join(argv),
    )


def _count_races(text: str, argv: list[str], returncode: int) -> int:
    """
    Count the race records the plugin reported.

    Evidence that the plugin actually ran is required before zero can be
    believed: an empty log from a plugin that never loaded looks exactly like a
    clean run, and silently scoring that as "no races" would make every mutant
    a false negative for reasons that have nothing to do with the detector.
    """
    records = len(_RACE_RECORD_RE.findall(text))
    if records:
        return records
    if _PLUGIN_ALIVE_RE.search(text) or _SUMMARY_NONE_RE.search(text):
        return 0
    if (match := _SUMMARY_COUNT_RE.search(text)):
        return int(match.group(1))
    raise DetectorError(
        f"no recognizable race-plugin output (exit {returncode}): neither a RACE "
        f"record nor a kernel-dispatch line was written, so the plugin appears not "
        f"to have run. Check that the config enables the 'race' plugin and a sink.\n"
        f"  command: {shlex.join(argv)}"
    )


def race_log(workdir: Path) -> Path:
    """The file sink names its log after the plugin, so: race.log."""
    return workdir / "race.log"


def _sink_text(workdir: Path, stdout: str, stderr: str) -> str:
    """
    Everything this plugin wrote, wherever its sink put it.

    Only race.log is read, never every *.log in the directory: mutant workdirs
    are shared between detectors, and picking up data_hazard.log here would
    count another detector's findings as this one's.
    """
    parts = [stdout, stderr]
    log = race_log(workdir)
    if log.is_file():
        try:
            parts.append(log.read_text(errors="replace"))
        except OSError:
            pass
    return "\n".join(part for part in parts if part)


def _config_from_wrapper(wrapper: list[str]) -> Path:
    for index, token in enumerate(wrapper):
        if token == "--config" and index + 1 < len(wrapper):
            return Path(wrapper[index + 1])
        if token.startswith("--config="):
            return Path(token.split("=", 1)[1])
    raise DetectorUnavailable(
        "--run-wrapper names no --config, so the race plugin cannot be enabled; "
        "expected something like 'rocjitsu --config /path/to/gfx950.json --'"
    )


def _substitute_config(wrapper: list[str], config_path: Path) -> list[str]:
    out: list[str] = []
    skip_next = False
    for index, token in enumerate(wrapper):
        if skip_next:
            skip_next = False
            continue
        if token == "--config" and index + 1 < len(wrapper):
            out += ["--config", str(config_path)]
            skip_next = True
        elif token.startswith("--config="):
            out.append(f"--config={config_path}")
        else:
            out.append(token)
    return out


def _derive_config(base_config: Path, workdir: Path) -> Path:
    """Enable the race plugin and point its file sink at this mutant's workdir."""
    try:
        config = json.loads(base_config.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DetectorUnavailable(f"cannot read rocjitsu config {base_config}: {error}") from error

    config.setdefault("plugins", {}).setdefault("race", {})
    config["sinks"] = {"types": ["file"], "dir": str(workdir)}

    workdir.mkdir(parents=True, exist_ok=True)
    # Namespaced by detector: several detectors share a mutant's workdir, and an
    # unqualified name means whichever ran last silently decides which plugin
    # the next one enables.
    derived = workdir / f"rocjitsu-config.{name}.json"
    derived.write_text(json.dumps(config, indent=2) + "\n")
    return derived
