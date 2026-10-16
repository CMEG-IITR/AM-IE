"""Compatibility entry point for the am_ie package."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("am_ie.ab_compare_tiers", run_name="__main__")
else:
    from am_ie.ab_compare_tiers import *  # noqa: F401,F403
