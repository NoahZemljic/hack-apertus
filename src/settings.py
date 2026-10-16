"""Shared defaults for the fixed perception model and persistent run outputs."""

import os
from pathlib import Path

QWEN_MODEL = "Qwen/Qwen3-VL-32B-Instruct"


def output_root() -> Path:
    return Path(os.environ.get("HACK_APERTUS_OUTPUT", Path.home() / "hack-apertus" / "outputs"))
