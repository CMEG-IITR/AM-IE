"""Compatibility entry point for the am_ie package."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("am_ie.compare_vs_ground_truth", run_name="__main__")
else:
    from am_ie.compare_vs_ground_truth import *  # noqa: F401,F403
