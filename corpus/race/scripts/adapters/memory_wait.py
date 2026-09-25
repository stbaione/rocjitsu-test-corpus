# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
rocjitsu's core memory-wait diagnostics, adapted to the corpus.

Not a plugin. This is built into the simulator's compute unit: a scoreboard
tracks in-flight memory operations and warns when a register or LDS byte is
touched before the operation producing it has completed. It is on by default
(``MemoryWaitDiagnostics::Warn``), so no plugin is loaded and none is enabled.
The only configuration is the ``memory_wait_diagnostics`` key, whose values are
"warn" and "off" -- and it belongs to each ``compute_unit`` *component* in the
topology, as a ``{"key", "value"}`` entry in that component's ``config`` array.
Setting it on ``vm.gpu.device``, next to ``lds_size_kb`` and the rest, is
silently ignored and reads exactly like a build without the feature.

That makes it the second distinct consumption shape the corpus serves:

* ``data_hazard``, ``race_detector``  plugins, enabled through ``plugins``
* ``memory_wait``                     a core simulator feature, always present

Diagnostics go through ``util::Logger::warn`` prefixed ``memory-wait:``, one
line per hazard, and the simulator caps them at ``kMaxMemoryWaitDiagnostics`` --
so a count from a very noisy kernel is a floor, not an exact total. That is
harmless for the corpus oracle, which only asks whether the count is nonzero.
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

name = "memory_wait"

# The scoreboard tracks memory results landing in registers, and LDS bytes with
# an unfinished access overlapping them. Global memory is absent: it observes
# in-flight operations of one wave, not cross-workgroup address conflicts.
supports = capability(
    {"vgpr", "agpr", "sgpr", "lds"},
    {"RAW", "WAR", "WAW"},
)

needs_execution = True

_TIMEOUT_SECONDS = 240

# util::Logger::warn(std::format("memory-wait: {} wg={} wave={} pc={:#x}: ...
_DIAGNOSTIC_RE = re.compile(r"^.*\bmemory-wait:", re.MULTILINE)

# The config loader rejects anything that is not "warn" or "off", which is what
# the capability probe below relies on.
_REJECTION_RE = re.compile(r"memory_wait_diagnostics must be warn or off")

_supported: bool | None = None  # probe result, cached for the process


def detect(artifacts: MutantArtifacts, run_wrapper: str | None) -> Detection:
    if artifacts.executable is None:
        raise DetectorUnavailable(
            "memory_wait needs a linked executable; it cannot run under --skip-all-runs"
        )
    if not run_wrapper:
        raise DetectorUnavailable(
            "memory_wait needs --run-wrapper, e.g. "
            "--run-wrapper 'rocjitsu --config /path/to/gfx950.json --'"
        )

    wrapper = shlex.split(run_wrapper)
    base_config = _config_from_wrapper(wrapper)
    _require_support(wrapper, base_config, artifacts)

    config_path = _derive_config(base_config, artifacts.workdir)
    argv = _substitute_config(wrapper, config_path) + [str(artifacts.executable)]

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
            f"memory_wait run timed out after {_TIMEOUT_SECONDS}s: {shlex.join(argv)}"
        ) from error

    text = f"{proc.stdout}\n{proc.stderr}"
    (artifacts.workdir / "memory_wait.log").write_text(text)
    return Detection(
        count=len(_DIAGNOSTIC_RE.findall(text)),
        raw=text,
        command=shlex.join(argv),
    )


def _require_support(wrapper: list[str], base_config: Path, artifacts: MutantArtifacts) -> None:
    """
    Establish that this build actually has the diagnostic, once per process.

    Without the probe, a simulator predating the feature simply never warns, and
    every mutant is recorded as missed for a reason that has nothing to do with
    detection. The config loader rejects any value other than "warn" or "off",
    so feeding it a deliberately invalid one is a direct question: a build that
    complains has the feature, and a build that shrugs does not.

    The probe runs the real executable. rocjitsu reports a missing binary with
    "execvp failed" *before* it parses the config, so probing with a bogus path
    answers a different question entirely and always looks unsupported.
    """
    global _supported
    if _supported is True:
        return
    if _supported is False:
        raise DetectorUnavailable(
            "this rocjitsu build does not support memory_wait_diagnostics; it "
            "predates the core memory-wait diagnostics, so every mutant would "
            "look missed"
        )

    probe_config = _write_config(
        base_config, artifacts.workdir / "memory_wait-probe.json", "probe-invalid"
    )
    proc = subprocess.run(
        _substitute_config(wrapper, probe_config) + [str(artifacts.executable)],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=_TIMEOUT_SECONDS,
    )
    _supported = bool(_REJECTION_RE.search(f"{proc.stdout}\n{proc.stderr}"))
    if not _supported:
        raise DetectorUnavailable(
            "this rocjitsu build does not support memory_wait_diagnostics "
            "(it accepted an invalid value instead of rejecting it), so every "
            "mutant would look missed"
        )


def _set_on_compute_units(node, value: str) -> int:
    """
    Set memory_wait_diagnostics on every compute_unit component in the topology.

    The key belongs to the compute_unit *component*, not to the device
    description: the loader reads it from the CfgMap handed to the
    ``compute_unit`` factory, which comes from that component's ``config``
    array of ``{"key", "value"}`` entries. Setting it on ``vm.gpu.device``
    instead is silently ignored, which reads exactly like a build without the
    feature.
    """
    touched = 0
    if isinstance(node, dict):
        if node.get("type") == "compute_unit":
            entries = node.setdefault("config", [])
            for entry in entries:
                if isinstance(entry, dict) and entry.get("key") == "memory_wait_diagnostics":
                    entry["value"] = value
                    break
            else:
                entries.append({"key": "memory_wait_diagnostics", "value": value})
            touched += 1
        for child in node.values():
            touched += _set_on_compute_units(child, value)
    elif isinstance(node, list):
        for child in node:
            touched += _set_on_compute_units(child, value)
    return touched


def _write_config(base_config: Path, out: Path, value: str) -> Path:
    """Copy the config with memory_wait_diagnostics set on every compute unit."""
    try:
        config = json.loads(base_config.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DetectorUnavailable(f"cannot read rocjitsu config {base_config}: {error}") from error

    if _set_on_compute_units(config, value) == 0:
        raise DetectorUnavailable(
            f"{base_config} declares no compute_unit component, so the memory-wait "
            "diagnostic cannot be configured"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2) + "\n")
    return out


def _derive_config(base_config: Path, workdir: Path) -> Path:
    """
    Set the diagnostic to "warn" explicitly rather than relying on the default.

    The default is already "warn", but a base config that had turned it off
    would otherwise silently report every mutant as missed.
    """
    return _write_config(base_config, workdir / f"rocjitsu-config.{name}.json", "warn")


def _config_from_wrapper(wrapper: list[str]) -> Path:
    for index, token in enumerate(wrapper):
        if token == "--config" and index + 1 < len(wrapper):
            return Path(wrapper[index + 1])
        if token.startswith("--config="):
            return Path(token.split("=", 1)[1])
    raise DetectorUnavailable(
        "--run-wrapper names no --config, so the memory-wait diagnostic cannot be "
        "configured; expected 'rocjitsu --config /path/to/gfx950.json --'"
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
