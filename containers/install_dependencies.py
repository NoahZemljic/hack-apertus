"""Install shared application requirements while freezing the release engine.

The ordinary uv.lock targets upstream Transformers. Apertus 1.5 needs the
release's fork, so this image deliberately resolves the root version bounds
against its inherited packages instead. Record the complete resulting manifest.
"""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import platform
import subprocess
import sys
import tomllib


def installation_files(project: dict, installed: dict[str, str]) -> tuple[str, str]:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    installed = {canonicalize_name(name): version for name, version in installed.items()}
    specs = project["project"]["dependencies"] + project["project"]["optional-dependencies"]["vision"]
    for spec in specs:
        requirement = Requirement(spec)
        version = installed.get(canonicalize_name(requirement.name))
        if version and not requirement.specifier.contains(version, prereleases=True):
            raise RuntimeError(f"Release package {requirement.name}=={version} conflicts with {spec}; "
                               "choose a compatible release image; do not replace the Apertus engine")
    constraints = "\n".join(f"{name}=={version}" for name, version in sorted(installed.items())) + "\n"
    return "\n".join(specs) + "\n", constraints


def main() -> None:
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError("Build on an ARM64 node with --platform linux/arm64")
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    installed = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}
    requirements, constraints = installation_files(project, installed)
    destination = root / "environment_manifest"
    destination.mkdir(exist_ok=True)
    (destination / "application.txt").write_text(requirements)
    (destination / "engine_constraints.txt").write_text(constraints)
    subprocess.run([sys.executable, "-m", "pip", "install", "--pre", "--only-binary=:all:",
                    "--constraint", str(destination / "engine_constraints.txt"),
                    "--requirement", str(destination / "application.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    (destination / "installed.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True))


if __name__ == "__main__":
    main()
