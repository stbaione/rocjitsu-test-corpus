# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The rocjitsu ``data_hazard`` plugin, adapted to the corpus.

Dynamic: the kernel runs under rocjitsu with the plugin enabled, and the plugin
writes a JSON report the adapter counts. This is the widest-scope detector of
the three -- registers, LDS, and cross-workgroup global memory, across RAW, WAR
and WAW -- because it observes real execution.

Plugin selection is part of the simulator configuration rather than the
environment, so this derives a per-run config from whatever config the
``--run-wrapper`` names, enabling the plugin and pointing its report at this
mutant's workdir.
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

name = "data_hazard"

# The widest scope of the three: ordering hazards on every resource, plus the
# two race kinds, which it can reach because it observes real execution and
# keeps a cross-workgroup shadow of global memory.
supports = capability(
    {"global", "lds", "scratch", "vgpr", "agpr", "sgpr"},
    {"RAW", "WAR", "WAW"},
) | frozenset({("lds", "LocalMemoryRace"), ("global", "GlobalMemoryRace")})

needs_execution = True

_TIMEOUT_SECONDS = 240


def detect(artifacts: MutantArtifacts, run_wrapper: str | None) -> Detection:
    if artifacts.executable is None:
        raise DetectorUnavailable(
            "data_hazard needs a linked executable; it cannot run under --skip-all-runs"
        )
    if not run_wrapper:
        raise DetectorUnavailable(
            "data_hazard needs --run-wrapper, e.g. "
            "--run-wrapper 'rocjitsu --config /path/to/gfx950.json --'"
        )

    wrapper = shlex.split(run_wrapper)
    base_config = _config_from_wrapper(wrapper)
    _check_simulated_target(base_config, artifacts.target)

    report_path = artifacts.workdir / "data_hazard.json"
    config_path = _derive_config(base_config, artifacts.workdir, report_path)
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
            f"data_hazard run timed out after {_TIMEOUT_SECONDS}s: {shlex.join(argv)}"
        ) from error

    count, raw = _read_report(report_path)
    if count == 0 and not report_path.is_file() and proc.returncode != 0:
        # No report and a failed run analyzed nothing. Reporting zero hazards
        # here would be indistinguishable from a clean run.
        raise DetectorError(
            f"data_hazard run failed (exit {proc.returncode}) and wrote no report: "
            f"{proc.stderr.strip()[:500]}"
        )
    return Detection(count=count, raw=raw, command=shlex.join(argv))


def _config_from_wrapper(wrapper: list[str]) -> Path:
    """The config path the wrapper names, which the derived config is built from."""
    for index, token in enumerate(wrapper):
        if token == "--config" and index + 1 < len(wrapper):
            return Path(wrapper[index + 1])
        if token.startswith("--config="):
            return Path(token.split("=", 1)[1])
    raise DetectorUnavailable(
        "--run-wrapper names no --config, so the data_hazard plugin cannot be enabled; "
        "expected something like 'rocjitsu --config /path/to/gfx950.json --'"
    )


def _substitute_config(wrapper: list[str], config_path: Path) -> list[str]:
    """The wrapper argv with its config replaced by the derived one."""
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


def _derive_config(base_config: Path, workdir: Path, report_path: Path) -> Path:
    """Write a copy of the config with the plugin enabled and its report placed."""
    try:
        config = json.loads(base_config.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DetectorUnavailable(f"cannot read rocjitsu config {base_config}: {error}") from error

    plugins = config.setdefault("plugins", {})
    plugins.setdefault("data_hazard", {})["report_path"] = str(report_path)
    config.setdefault("sinks", {"types": ["file"], "dir": str(workdir)})

    workdir.mkdir(parents=True, exist_ok=True)
    # Namespaced by detector: several detectors share a mutant's workdir, and an
    # unqualified name means whichever ran last silently decides which plugin
    # the next one enables.
    derived = workdir / f"rocjitsu-config.{name}.json"
    derived.write_text(json.dumps(config, indent=2) + "\n")
    return derived


def _gfx_target_version(target: str) -> int | None:
    """Encode a gfx target the way the configs do: gfx1250 -> 120500."""
    match = re.fullmatch(r"gfx(\d+)([0-9a-f])([0-9a-f])", target)
    if not match:
        return None
    return int(match.group(1)) * 10000 + int(match.group(2), 16) * 100 + int(match.group(3), 16)


def _check_simulated_target(config_path: Path, target: str) -> None:
    """
    Reject a config that simulates a different target than the kernels were
    built for.

    A kernel the simulated device cannot run reports no hazards at all, which
    looks exactly like a clean run rather than a misconfiguration. It is the
    check, not the mismatch, that has failed when the config cannot be read.
    """
    wanted = _gfx_target_version(target)
    if wanted is None:
        return
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DetectorUnavailable(f"cannot read rocjitsu config {config_path}: {error}") from error

    for section in (("vm", "gpu", "device"), ("dbt_guest", "guest_device")):
        node = config
        for key in section:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict) and node.get("gfx_target_version") is not None:
            found = node["gfx_target_version"]
            if found != wanted:
                raise DetectorError(
                    f"config {config_path} simulates gfx_target_version {found}, but kernels "
                    f"are built for {target} ({wanted}); the mismatch reports zero hazards"
                )
            return

    raise DetectorError(
        f"config {config_path} names no gfx_target_version, so the target it simulates "
        f"cannot be checked against {target}"
    )


def _read_report(path: Path) -> tuple[int, str]:
    """The plugin writes a JSON list of findings; its length is the count."""
    if not path.is_file():
        return 0, ""
    try:
        raw = path.read_text()
    except OSError as error:
        raise DetectorError(f"cannot read data_hazard report {path}: {error}") from error
    try:
        findings = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DetectorError(f"data_hazard report {path} is not valid JSON: {error}") from error
    if not isinstance(findings, list):
        raise DetectorError(
            f"data_hazard report {path} is {type(findings).__name__}, expected a list"
        )
    return len(findings), raw
