#!/usr/bin/env python3
"""Export HPC pip requirements from pyproject.toml.

This is a LOCAL DEVELOPER UTILITY — run it on your own machine before pushing
code whenever pyproject.toml dependencies change. It reads the modules from
env/hpc/modules.txt and the full dependency list from pyproject.toml, then
writes the remainder to env/hpc/requirements.txt.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def normalise(name: str) -> str:
    """Normalise a PyPI package name for comparison."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pkg_name(dep: str) -> str:
    """Extract the bare package name from a PEP 508 dependency string."""
    return re.split(r"[\[=><~!;]", dep)[0].strip()


def main() -> None:
    import tomllib

    modules_path = ROOT / "env" / "hpc" / "modules.txt"
    if not modules_path.exists():
        print(f"Error: {modules_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Read normalized module names from base modules only
    # Library modules (like PyTorch) are kept in requirements for portability
    module_names = [
        normalise(line.split()[0].split("/")[0])
        for line in modules_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    pyproject_path = ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    # Collect all dependencies, merging 'cuda' extras into base dependencies
    dep_dict: dict[str, str] = {}
    for dep in data.get("project", {}).get("dependencies", []):
        dep_dict[normalise(pkg_name(dep))] = dep

    # Add cuda extras (takes precedence for HPC)
    optional_deps = data.get("project", {}).get("optional-dependencies", {})
    for group in ["cuda"]:
        for dep in optional_deps.get(group, []):
            dep_dict[normalise(pkg_name(dep))] = dep

    deps = list(dep_dict.values())

    final_deps: list[str] = []
    print("Checking dependencies against HPC module list...", file=sys.stderr)
    for dep in deps:
        name = normalise(pkg_name(dep))
        # Smart check: if the package name is a substring of any loaded module name
        # (e.g. 'torch' in 'pytorch', 'scipy' in 'scipy-bundle')
        if any(name in mod for mod in module_names):
            print(f"  [skip – module provider found] {dep}", file=sys.stderr)
            continue

        final_deps.append(dep)
        print(f"  [pip] {dep}", file=sys.stderr)

    hpc_dir = ROOT / "env" / "hpc"
    output_path = hpc_dir / "requirements.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(final_deps) + "\n")
    print(f"\nWrote {len(final_deps)} requirement(s) to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
