#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Check that ``cases.toml`` agrees with ``kernels/`` and the committed ``asm/``.

``regenerate_asm.py --check`` compares assembly text and needs a compiler. This
needs neither: it checks the kernel inventory against the files on disk, so a
case and its pinned assembly cannot drift apart unnoticed::

    python corpus/race/scripts/check_cases.py            # exit 1 on any mismatch
    python corpus/race/scripts/check_cases.py --suggest  # print derived waits tables

It checks that:

* every case uses only known fields, and no two cases share an id or a kernel;
* every target a case names has a directory under ``asm/``;
* every case has ``kernels/<kernel>.hip`` and, for each target it builds for,
  ``asm/<target>/<kernel>.s``;
* every ``asm/<target>/*.s`` belongs to a case that builds for that target;
* ``waits`` has exactly one entry per target, and that entry lists exactly the
  counters the target's assembly waits on (see ``wait_counters``);
* each ``[[case.exempt]]`` ordinal names, under the counting rule in
  ``wait_sites``, the instruction its ``site`` says it does.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

CORPUS_ROOT = Path(__file__).resolve().parent.parent
ASM_DIR = CORPUS_ROOT / "asm"
KERNELS_DIR = CORPUS_ROOT / "kernels"
CASES_PATH = CORPUS_ROOT / "cases.toml"

KNOWN_FIELDS = frozenset(
    # Inventory, for regenerate_asm.py.
    {"id", "kernel", "requires", "headers"}
    # Mutation rules, for generate.py.
    | {"mutate", "hazard", "description", "waits", "resource", "exempt"}
)

# s_wait_xcnt tracks address-translation replay rather than data completion.
# It is never a mutation site and never counts as a wait the kernel exercises.
EXCLUDED_WAITS = frozenset({"s_wait_xcnt"})

# Matches a wait mnemonic and its first operand on a stripped line. Kept in
# step with mutate.find_wait_sites, so an ordinal means the same instruction
# here and when a mutant is built.
_WAIT_RE = re.compile(r"(s_wait\w+)\s+(\S+)")

# Counter fields named inside a gfx9-era combined s_waitcnt operand.
_LEGACY_FIELD_RE = re.compile(r"\b(vmcnt|lgkmcnt|expcnt|vscnt)\(")


@dataclass(frozen=True)
class WaitSite:
    ordinal: int  # 0-based, see wait_sites
    line: int  # 1-based line in the .s file, for messages
    mnemonic: str
    counters: frozenset[str]


def _site_counters(mnemonic: str, stripped_line: str) -> frozenset[str]:
    """
    The counters one wait instruction drains, spelled as the target spells them.

    gfx9 names its counters in the operand of a combined ``s_waitcnt``; gfx10+
    names one in the mnemonic (``s_waitcnt_vscnt``, ``s_wait_loadcnt``) or two
    (``s_wait_loadcnt_dscnt``). A wait naming two counters drains both.
    """
    if mnemonic == "s_waitcnt":
        return frozenset(_LEGACY_FIELD_RE.findall(stripped_line))
    suffix = mnemonic.removeprefix("s_waitcnt_").removeprefix("s_wait_")
    return frozenset(part for part in suffix.split("_") if part.endswith("cnt"))


def wait_sites(asm_path: Path) -> list[WaitSite]:
    """
    Every wait mutation site in *asm_path*, with its ordinal.

    The counting rule: one domain for all counters (a ``kmcnt`` wait and a
    ``loadcnt`` wait share it), in file order, starting at 0. Comment and
    directive lines are skipped, and so is ``s_wait_xcnt``. A wait naming two
    counters is one site.
    """
    sites: list[WaitSite] = []
    for number, line in enumerate(asm_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith((";", ".")):
            continue
        match = _WAIT_RE.match(stripped)
        if not match or match.group(1) in EXCLUDED_WAITS:
            continue
        sites.append(
            WaitSite(
                ordinal=len(sites),
                line=number,
                mnemonic=match.group(1),
                counters=_site_counters(match.group(1), stripped),
            )
        )
    return sites


def wait_counters(asm_path: Path) -> set[str]:
    """Every counter some wait site in *asm_path* drains."""
    return {counter for site in wait_sites(asm_path) for counter in site.counters}


def asm_targets() -> list[str]:
    return sorted(path.name for path in ASM_DIR.iterdir() if path.is_dir())


def case_targets(case: dict, all_targets: list[str]) -> list[str]:
    """The targets a case builds for: its ``requires``, or every target."""
    return sorted(case.get("requires") or all_targets)


def check(cases: list[dict], all_targets: list[str]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_kernels: set[str] = set()
    owned: set[tuple[str, str]] = set()

    for case in cases:
        where = f"cases.toml [{case.get('id', '?')}]"
        for field in sorted(set(case) - KNOWN_FIELDS):
            errors.append(f"{where}: unknown field {field!r}")
        if case["id"] in seen_ids:
            errors.append(f"{where}: duplicate id")
        if case["kernel"] in seen_kernels:
            errors.append(f"{where}: kernel {case['kernel']} is already used by another case")
        seen_ids.add(case["id"])
        seen_kernels.add(case["kernel"])

        if not (KERNELS_DIR / f"{case['kernel']}.hip").is_file():
            errors.append(f"{where}: kernels/{case['kernel']}.hip is missing")
        for target in case.get("requires", []):
            if target not in all_targets:
                errors.append(f"{where}: requires {target}, which has no asm/{target}/ directory")

        targets = case_targets(case, all_targets)
        waits = case.get("waits")
        if not isinstance(waits, dict):
            errors.append(
                f"{where}: waits must be a table keyed by target, e.g. "
                f"waits = {{ {targets[0]} = [...] }}"
            )
            waits = {}
        for target in sorted(set(waits) - set(targets)):
            errors.append(f"{where}: waits names {target}, which the case does not build for")

        for target in targets:
            owned.add((target, f"{case['kernel']}.s"))
            asm = ASM_DIR / target / f"{case['kernel']}.s"
            if not asm.is_file():
                errors.append(f"{where}: asm/{target}/{case['kernel']}.s is missing")
                continue
            errors += _check_waits(where, target, waits.get(target), asm)
            errors += _check_exemptions(where, case, target, asm)

    for target in all_targets:
        for asm in sorted((ASM_DIR / target).glob("*.s")):
            if (target, asm.name) not in owned:
                errors.append(f"asm/{target}/{asm.name}: no case builds {asm.stem} for {target}")
    return errors


def _check_waits(where: str, target: str, declared: list[str] | None, asm: Path) -> list[str]:
    actual = wait_counters(asm)
    if declared is None:
        return [f"{where}: waits has no {target} entry; the asm waits on {sorted(actual)}"]
    if set(declared) == actual and len(declared) == len(set(declared)):
        return []
    detail = []
    if (missing := sorted(actual - set(declared))):
        detail.append(f"asm also waits on {missing}")
    if (extra := sorted(set(declared) - actual)):
        detail.append(f"asm never waits on {extra}")
    if len(declared) != len(set(declared)):
        detail.append("duplicate entries")
    return [f"{where}: waits.{target} {declared}: {'; '.join(detail)}"]


def _check_exemptions(where: str, case: dict, target: str, asm: Path) -> list[str]:
    errors: list[str] = []
    sites = wait_sites(asm)
    for exemption in case.get("exempt", []):
        if exemption.get("target") not in (None, target):
            continue
        if exemption["kind"] != "wait":
            errors.append(
                f"{where}: cannot verify a {exemption['kind']!r} exemption; "
                f"only wait ordinals are checked"
            )
            continue
        ordinal = exemption["ordinal"]
        if ordinal >= len(sites):
            errors.append(
                f"{where}: exempt wait ordinal {ordinal} is out of range; "
                f"asm/{target}/{case['kernel']}.s has {len(sites)} wait sites"
            )
        elif sites[ordinal].mnemonic != exemption["site"]:
            site = sites[ordinal]
            errors.append(
                f"{where}: exempt wait ordinal {ordinal} is {site.mnemonic} at "
                f"asm/{target}/{case['kernel']}.s:{site.line}, not {exemption['site']}"
            )
    return errors


def suggest(cases: list[dict], all_targets: list[str]) -> None:
    """Print the waits table each case's committed assembly implies."""
    for case in cases:
        entries = []
        for target in case_targets(case, all_targets):
            asm = ASM_DIR / target / f"{case['kernel']}.s"
            counters = sorted(wait_counters(asm)) if asm.is_file() else []
            entries.append(f"{target} = [" + ", ".join(f'"{c}"' for c in counters) + "]")
        print(f"{case['id']:24} waits = {{ {', '.join(entries)} }}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Print the waits table each case's committed assembly implies, and exit.",
    )
    args = parser.parse_args()

    manifest = tomllib.loads(CASES_PATH.read_text())
    if manifest.get("corpus", {}).get("schema") != 1:
        print("cases.toml: unsupported corpus schema", file=sys.stderr)
        return 2
    cases = manifest["case"]
    targets = asm_targets()

    if args.suggest:
        suggest(cases, targets)
        return 0

    errors = check(cases, targets)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} mismatch(es) between cases.toml and the corpus", file=sys.stderr)
        return 1
    print(f"cases.toml agrees with kernels/ and asm/ ({len(cases)} cases, targets: {', '.join(targets)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
