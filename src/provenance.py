"""Record the exact working tree and planning policy used for a run."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_run_manifest(output_dir: Path, settings: dict) -> dict:
    policy_path = ROOT / "prompts" / "apertus_system.txt"
    policy = policy_path.read_text()
    sources = [ROOT / "run_grounded.py"]
    for directory in ("src", "schema", "prompts"):
        sources.extend(sorted((ROOT / directory).rglob("*.py" if directory == "src" else "*")))
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sources if path.is_file()}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT), "python": sys.executable,
        "planning_policy_path": str(policy_path), "planning_policy_sha256": sha256_text(policy),
        "source_sha256": hashes, "settings": settings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "planning_policy.txt").write_text(policy)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Source: {ROOT}; planning policy SHA256: {manifest['planning_policy_sha256']}", flush=True)
    return manifest
