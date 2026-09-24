# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Mutation-based race-detection suite.

Each case is one baseline or one mutant. A mutant is built by compiling its
kernel to assembly, applying a recipe from ``corpus/race/mutants.toml`` --
"remove the Nth wait", "rewrite the Nth sub-dword load as a store" -- and
taking the edited assembly the rest of the way to an executable. The selected
detector then reports how many hazards it finds, and the case passes when a
clean baseline reports none and a mutant reports some.

Which detector runs is chosen with ``--detector``; every one of them declares
the ``(resource, hazard_kind)`` pairs it covers, and a mutant outside that set
is skipped rather than counted as a miss. That is what lets detectors with very
different scopes share one corpus: ``race_detector`` has no global-memory
shadow and no LDS WAW, so it skips those and is scored only on what it claims.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

import pytest

from support.define_contracts import (
    BuildResult,
    BuildState,
    CorpusCase,
    RunContext,
    TargetSpec,
)
from support.prepare_inputs import load_suite_target_configs, supports_target

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "corpus" / "race"
CONFIGS_ROOT = CORPUS_ROOT / "configs"
KERNELS_ROOT = CORPUS_ROOT / "kernels"
ASM_ROOT = CORPUS_ROOT / "asm"
SCRIPTS_ROOT = CORPUS_ROOT / "scripts"

# The corpus owns its own protocol and engine, the way corpus/semantics does.
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import adapters  # noqa: E402
import build as corpus_build  # noqa: E402
import mutate  # noqa: E402
from detector_protocol import (  # noqa: E402
    Detection,
    DetectorError,
    DetectorUnavailable,
    MutantArtifacts,
    Verdict,
    out_of_scope_reason,
    score,
    validate_tags,
)

BASELINE = "baseline"


# --- Discovery ---------------------------------------------------------------


def default_config_files() -> tuple[Path, ...]:
    return tuple(sorted(CONFIGS_ROOT.glob("*.json")))


def load_target_configs(config_files: tuple[str, ...] | list[str]) -> list[dict]:
    return load_suite_target_configs(config_files, repo_root=REPO_ROOT)


def _load_manifest() -> dict:
    path = CORPUS_ROOT / "mutants.toml"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; generate it with "
            "`python corpus/race/scripts/generate.py --target <gfx>`"
        )
    manifest = tomllib.loads(path.read_text())
    if manifest.get("manifest", {}).get("schema") != 1:
        raise ValueError(f"{path}: unsupported manifest schema")
    return manifest


def discover(target: TargetSpec, target_configs: list[dict]) -> list[CorpusCase]:
    """
    One case per baseline and per mutant, read from the committed manifest.

    Reading recipes rather than compiling is what keeps collection free of any
    toolchain: discovery happens before ``build()``, so the mutant list has to
    be knowable without a compiler.
    """
    manifest = _load_manifest()
    mutants = [row for row in manifest.get("mutant", []) if row["target"] == target.target]

    discovered: list[CorpusCase] = []
    for target_config in target_configs:
        if not supports_target(target, target_config):
            continue

        # A baseline per kernel, so an unmutated kernel reporting hazards is
        # visible as its own failure instead of poisoning every mutant of it.
        for kernel in sorted({row["kernel"] for row in mutants}):
            discovered.append(
                _case(
                    target=target,
                    target_config=target_config,
                    kernel=kernel,
                    mutant_id=BASELINE,
                    kind=BASELINE,
                    ordinal=-1,
                    resource="",
                    hazard="",
                    case_name=f"{kernel}.{BASELINE}",
                )
            )

        for row in mutants:
            validate_tags(row["resource"], row["hazard"], f"mutants.toml:{row['id']}")
            exempt_reason = row.get("exempt_reason", "") if row.get("exempt") else ""
            discovered.append(
                _case(
                    target=target,
                    target_config=target_config,
                    kernel=row["kernel"],
                    mutant_id=row["id"],
                    kind=row["kind"],
                    ordinal=row["ordinal"],
                    resource=row["resource"],
                    hazard=row["hazard"],
                    case_name=row["id"],
                    selector_names=(row["case"],),
                    exempt_reason=exempt_reason,
                )
            )
    return discovered


def _case(
    *,
    target: TargetSpec,
    target_config: dict,
    kernel: str,
    mutant_id: str,
    kind: str,
    ordinal: int,
    resource: str,
    hazard: str,
    case_name: str,
    selector_names: tuple[str, ...] = (),
    exempt_reason: str = "",
) -> CorpusCase:
    return CorpusCase(
        id=f"race.{target.target}.{case_name}",
        suite="race",
        target=target.target,
        collection=kernel,
        backend=None,
        path=KERNELS_ROOT / f"{kernel}.hip",
        build={
            "system": "assembly",
            "config_name": target_config["config_name"],
            "kernel": kernel,
            "mutant": mutant_id,
        },
        run={"kind": "race-mutant"},
        metadata={
            "name": case_name,
            "kernel": kernel,
            "mutant": mutant_id,
            "mutation_kind": kind,
            "ordinal": ordinal,
            "resource": resource,
            "hazard": hazard,
            "exempt_reason": exempt_reason,
            "target_config": target_config,
        },
        selector_names=selector_names,
    )


# --- Build -------------------------------------------------------------------


def _kernel_cache(context: RunContext, target: str, kernel: str) -> Path:
    return context.artifact_directory / "race" / target / kernel


def _ensure_kernel_artifacts(
    kernel_name: str,
    target: str,
    cache: Path,
    tools: corpus_build.Toolchain,
) -> tuple[Path, Path]:
    """
    The per-kernel halves: the committed baseline assembly, and the host object.

    The assembly is read straight out of ``asm/<target>/`` rather than compiled.
    It is frozen in the repository so that the set of mutation sites cannot
    shift under a toolchain change -- one mutant is derived per ``s_wait_*``, so
    recompiling here would let a compiler upgrade silently renumber every case.

    The host object still has to be compiled, and cannot be committed beside the
    assembly: its fatbin symbol is a per-kernel hash read back with llvm-nm at
    embed time, and it is an x86-64 object tied to the HIP runtime ABI. It
    depends only on the kernel, so it is cached here and shared by every mutant
    of that kernel, written via a temporary file and renamed so concurrent
    workers cannot observe a half-written object.
    """
    cache.mkdir(parents=True, exist_ok=True)
    source = KERNELS_ROOT / f"{kernel_name}.hip"
    asm = ASM_ROOT / target / f"{kernel_name}.s"
    if not asm.is_file():
        pytest.skip(
            f"{asm.relative_to(CORPUS_ROOT)} is missing; "
            "run corpus/race/scripts/regenerate_asm.py"
        )

    host_obj = cache / "host.o"
    if not host_obj.is_file():
        staged = cache / f"host.o.{os.getpid()}"
        corpus_build.compile_host_object(source, staged, tools=tools)
        staged.replace(host_obj)
    return asm, host_obj


def build(case: CorpusCase, context: RunContext, build_state: BuildState) -> BuildResult:
    detector = _detector_for(context)
    kernel = case.metadata["kernel"]
    mutant = case.metadata["mutant"]
    target = case.target

    try:
        tools = corpus_build.resolve_toolchain()
    except corpus_build.BuildError as error:
        pytest.skip(f"race suite needs a ROCm toolchain: {error}")

    cache = _kernel_cache(context, target, kernel)
    baseline_asm, host_obj = _ensure_kernel_artifacts(kernel, target, cache, tools)

    workdir = cache / mutant
    workdir.mkdir(parents=True, exist_ok=True)

    if mutant == BASELINE:
        asm = baseline_asm
    else:
        asm = workdir / f"{kernel}.s"
        try:
            mutate.apply_mutation(
                baseline_asm, case.metadata["mutation_kind"], case.metadata["ordinal"], asm
            )
        except mutate.StaleOrdinalError as error:
            # A toolchain that emits fewer sites than the manifest records is a
            # stale recipe, not a failure: degrade rather than break the suite.
            pytest.skip(f"{error}")

    # A static detector needs only the code object, so skip linking for it --
    # that is what lets it work under --skip-all-runs.
    link = detector.needs_execution and not context.skip_all_runs
    built = corpus_build.build_from_asm(asm, host_obj, target, workdir, tools=tools, link=link)

    return BuildResult(
        build_dir=workdir,
        executable_path=built.executable,
        metadata={
            "asm": built.asm,
            "code_object": built.code_object,
            "workdir": workdir,
            "kernel_cache": cache,
            "host_obj": host_obj,
            "baseline_asm": baseline_asm,
        },
    )


# --- Run ---------------------------------------------------------------------


def _detector_for(context: RunContext):
    name = getattr(context, "detector", None) or "data_hazard"
    try:
        return adapters.load(name)
    except (ValueError, TypeError) as error:
        raise pytest.UsageError(f"--detector: {error}") from error


def _artifacts(case: CorpusCase, build_result: BuildResult, workdir: Path) -> MutantArtifacts:
    return MutantArtifacts(
        asm=build_result.metadata["asm"],
        code_object=build_result.metadata["code_object"],
        executable=build_result.executable_path,
        workdir=workdir,
        target=case.target,
    )


def _baseline_detection(
    case: CorpusCase,
    build_result: BuildResult,
    context: RunContext,
    detector,
) -> Detection:
    """
    The kernel's unmutated hazard count, built and cached once per kernel.

    Every mutant needs it to interpret its own count, and rebuilding it per
    mutant would multiply the work by the number of mutants a kernel has.
    """
    cache = build_result.metadata["kernel_cache"]
    workdir = cache / BASELINE
    marker = workdir / f".{detector.name}.count"
    if marker.is_file():
        return Detection(count=int(marker.read_text().strip()))

    tools = corpus_build.resolve_toolchain()
    link = detector.needs_execution and not context.skip_all_runs
    built = corpus_build.build_from_asm(
        build_result.metadata["baseline_asm"],
        build_result.metadata["host_obj"],
        case.target,
        workdir,
        tools=tools,
        link=link,
    )
    detection = detector.detect(
        MutantArtifacts(
            asm=built.asm,
            code_object=built.code_object,
            executable=built.executable,
            workdir=workdir,
            target=case.target,
        ),
        context.run_wrapper,
    )
    marker.write_text(str(detection.count))
    return detection


def run(case: CorpusCase, build_result: BuildResult, context: RunContext) -> None:
    detector = _detector_for(context)
    metadata = case.metadata
    mutant = metadata["mutant"]
    workdir = build_result.metadata["workdir"]

    if detector.needs_execution and context.skip_all_runs:
        pytest.skip(f"{detector.name} needs execution; --skip-all-runs was given")

    if mutant == BASELINE:
        try:
            detection = detector.detect(_artifacts(case, build_result, workdir), context.run_wrapper)
        except DetectorUnavailable as error:
            pytest.skip(str(error))
        except DetectorError as error:
            pytest.fail(str(error))
        assert detection.count == 0, (
            f"{metadata['kernel']} baseline is not clean: {detector.name} reports "
            f"{detection.count} hazard(s) before anything was mutated.\n"
            f"  command: {detection.command}\n{detection.raw[:2000]}"
        )
        return

    # Checked before scope: an exempt mutation is a no-op for every detector,
    # so reporting nothing is the right answer rather than a miss. Recorded as
    # EXEMPT rather than SKIPPED so a comparison can tell "nobody should find
    # this" apart from "this detector does not cover it".
    if metadata["exempt_reason"]:
        reason = f"exempt: {metadata['exempt_reason']}"
        pytest.skip(reason)

    reason = out_of_scope_reason(detector, metadata["resource"], metadata["hazard"])
    if reason is not None:
        pytest.skip(reason)

    try:
        baseline = _baseline_detection(case, build_result, context, detector)
        detection = detector.detect(_artifacts(case, build_result, workdir), context.run_wrapper)
    except DetectorUnavailable as error:
        pytest.skip(str(error))
    except (DetectorError, corpus_build.BuildError) as error:
        pytest.fail(str(error))

    verdict = score(baseline, detection)
    if verdict is Verdict.BASELINE_DIRTY:
        pytest.skip(
            f"{metadata['kernel']} baseline already reports {baseline.count} hazard(s); "
            f"see the {metadata['kernel']}.baseline case"
        )
    if verdict is Verdict.MISSED:
        pytest.fail(
            f"{detector.name} did not report the injected "
            f"{metadata['hazard']} on {metadata['resource']}: "
            f"removed {metadata['mutation_kind']} ordinal {metadata['ordinal']} from "
            f"{metadata['kernel']}, baseline={baseline.count} mutant={detection.count}\n"
            f"  command: {detection.command}"
        )
