"""Compatibility entrypoint; use run_grounded.py for the full perception/planning pipeline."""

from run_grounded import create_parser, main


if __name__ == "__main__":
    raise SystemExit(main())
