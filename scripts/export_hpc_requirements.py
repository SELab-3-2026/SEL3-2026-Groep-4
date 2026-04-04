#!/usr/bin/env python3
"""Export HPC pip requirements from pyproject.toml.

This is a LOCAL DEVELOPER UTILITY — run it on your own machine before pushing
code whenever pyproject.toml dependencies change. It reads the full dependency
list from pyproject.toml and subtracts packages already provided by HPC modules
(listed in env/hpc/modules.txt), then writes the remainder to
env/hpc/requirements.txt for use by vsc-venv on the cluster.

Usage:
    uv run scripts/export_hpc_requirements.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Packages provided by HPC modules (PyPI name → Python import name).
# We skip any pyproject.toml dep whose import can be found after loading
# the HPC modules. As this script runs locally (without those modules), we
# maintain an explicit exclusion list keyed by normalised PyPI package name.
EXCLUDED_BY_MODULE = {
    "jax",
    "flax",
    "optax",
    "wandb",
    "matplotlib",
    "pyyaml",  # PyYAML module
    "ffmpeg",  # FFmpeg is a system tool, not a Python package
}


def normalise(name: str) -> str:
    """Normalise a PyPI package name for comparison."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pkg_name(dep: str) -> str:
    """Extract the bare package name from a PEP 508 dependency string."""
    return re.split(r"[\[=><~!;]", dep)[0].strip()


def main() -> None:
    try:
        import tomllib  # Python ≥ 3.11
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject_path = ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    deps: list[str] = data.get("project", {}).get("dependencies", [])

    missing: list[str] = []
    for dep in deps:
        name = normalise(pkg_name(dep))
        if name in EXCLUDED_BY_MODULE:
            print(f"  [skip – provided by HPC module] {dep}", file=sys.stderr)
            continue
        missing.append(dep)
        print(f"  [pip] {dep}", file=sys.stderr)

    output_path = ROOT / "env" / "hpc" / "requirements.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(missing) + "\n")
    print(f"\nWrote {len(missing)} requirement(s) to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
