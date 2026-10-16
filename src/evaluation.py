"""Run the official scorer only after the final submission has been written."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def evaluate_submission(results: Path, data: Path) -> None:
    """Save scoring status; missing private annotations are an explicit skip.

    Scoring errors propagate so the one-command run cannot report success when
    annotations exist but evaluation fails. Ground truth is never a planner input.
    """
    status_path = results.parent / "evaluation.json"
    report_path = results.parent / "report.json"
    # A reused output directory must not retain a report from an earlier run.
    report_path.unlink(missing_ok=True)
    status = {"results": str(results), "data": str(data)}
    if not (data / "ground_truth" / "ground_truth.json").is_file():
        status.update(status="skipped", reason="Ground-truth answers are unavailable; local scoring is not possible.")
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"Evaluation skipped: {status['reason']} Submission: {results}", flush=True)
        return
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "starter/evaluate.py"),
               str(results), "--data", str(data), "--report", str(report_path)]
    print("Evaluating final submission with the official scorer...", flush=True)
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        report_path.unlink(missing_ok=True)
        status.update(status="failed", error=str(error))
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        raise
    status.update(status="completed", report=str(report_path))
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"Evaluation report: {report_path}", flush=True)
