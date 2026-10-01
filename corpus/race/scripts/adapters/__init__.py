# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Detector adapters, resolved by ``--detector``.

Each module here adapts one race detector to the corpus contract in
``detector_protocol``: it declares which ``(resource, hazard_kind)`` pairs its
tool covers, whether it needs the kernel executed, and how to turn that tool's
native output into a ``Detection``. Everything tool-specific -- report formats,
config fragments, CLI flags -- is confined to these modules, so adding a
detector means adding one file here and nothing else.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from detector_protocol import Detector

# Adapters ship as modules rather than classes: a module already is a singleton
# with named attributes, which is all the Detector protocol asks for.
KNOWN_DETECTORS = ("data_hazard",  "memory_wait", "race_detector")


def load(name: str) -> "Detector":
    """Import the adapter named *name*, rejecting an unknown one by name."""
    if name not in KNOWN_DETECTORS:
        raise ValueError(
            f"unknown detector {name!r}; known: {', '.join(KNOWN_DETECTORS)}"
        )
    module = importlib.import_module(f"adapters.{name}")
    _validate(module, name)
    return module  # type: ignore[return-value]


def _validate(module, name: str) -> None:
    """
    Check an adapter satisfies the contract at import, not mid-run.

    A missing attribute here would otherwise surface as an AttributeError deep
    in a test run, long after the useful context is gone.
    """
    from detector_protocol import validate_tags

    for attribute in ("name", "supports", "needs_execution", "detect"):
        if not hasattr(module, attribute):
            raise TypeError(f"detector adapter {name!r} is missing '{attribute}'")
    if not module.supports:
        raise TypeError(f"detector adapter {name!r} declares no capabilities")
    for pair in module.supports:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(
                f"detector adapter {name!r}: 'supports' holds {pair!r}, expected "
                "(resource, hazard_kind) pairs"
            )
        validate_tags(pair[0], pair[1], f"adapters.{name}.supports")
