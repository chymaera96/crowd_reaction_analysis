#!/usr/bin/env python3
"""Compatibility entry point; prefer the installed crowd-reaction commands."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crowd_reaction.inference import *  # noqa: F401,F403,E402

if __name__ == "__main__":
    main()
