"""Compatibility entry point for the am_ie package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    import runpy
    runpy.run_module("am_ie.param_category", run_name="__main__")
else:
    from am_ie.param_category import *  # noqa: F401,F403
