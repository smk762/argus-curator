"""Every module imports on its own, in a fresh interpreter: an import cycle only shows in some orders."""

from __future__ import annotations

import pkgutil
import subprocess
import sys

import pytest

import argus_curator

# Needs torch + timm, the [mivolo-convert] extra; nothing on the inference path imports it.
NEEDS_EXTRA = {"argus_curator._mivolo_arch"}

MODULES = sorted(
    m.name
    for m in pkgutil.walk_packages(argus_curator.__path__, "argus_curator.")
    if m.name not in NEEDS_EXTRA and not m.name.endswith("__main__")
)


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_first(module: str) -> None:
    proc = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
