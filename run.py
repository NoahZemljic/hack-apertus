import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_DIRECTORY = Path(__file__).resolve().parent
BASELINE_SCRIPT = PROJECT_DIRECTORY.joinpath("baseline/baseline.py")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official instructor baseline.")
    # Defaults stay relative to the project even when invoked from another directory.
    parser.add_argument("--data", type=Path, default=PROJECT_DIRECTORY.joinpath("dataset"))
    parser.add_argument("--out", type=Path, default=PROJECT_DIRECTORY.joinpath("outputs/results.json"))
    parser.add_argument("--mock", action="store_true", help="Check the setup without a model")
    parser.add_argument("--model", help="Model identifier served by the local server")
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    if not BASELINE_SCRIPT.is_file():
        parser.error(f"Baseline script does not exist: {BASELINE_SCRIPT}")
    if not args.data.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data}")
    if not any(args.data.glob("*/rgb.png")):
        parser.error(f"No scene images found in {args.data}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Reuse this interpreter so the baseline gets the same virtual environment.
    command = [
        sys.executable,
        str(BASELINE_SCRIPT),
        "--data", str(args.data.resolve()),
        "--out", str(args.out.resolve()),
    ]
    if args.mock:
        command.append("--mock")
    if args.model:
        command.extend(["--model", args.model])
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
