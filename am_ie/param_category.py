"""
Symbolic (non-AI) property-category tagger.

Flags every extracted parameter row with exactly one of the taxonomy's
five L8 property categories:

    mechanical | physical | thermal | microstructural | electrical

Two-step lookup, no LLM call:
  1. Exact match: if the parameter name IS one of the ~40 L8 property
     names itself (UTS, Ra, Tg, Grain Size, ...), read its category
     straight off am_taxonomy_v4.html's s-tag-{mech,phys,therm,micro,elec}
     class — ground truth, not guessed.
  2. Otherwise (an L4 feedstock spec or L5 process setting, e.g. Layer
     Thickness, Laser Power, D50) fall back to a hand-curated map of which
     single category that parameter most directly bears on.

Rows that don't fit any of the five (post-processing steps, categorical
fields like Scan Strategy, materials info) get category=None rather than
a forced guess.
"""
import re

TAXONOMY_PATH = "am_taxonomy_v4.html"

CATEGORY_NAMES = {
    "mech": "mechanical",
    "phys": "physical",
    "therm": "thermal",
    "micro": "microstructural",
    "elec": "electrical",
}


def _base_name(name):
    n = re.sub(r"<[^>]+>", "", str(name))
    n = re.sub(r"\s*\([^)]*\)\s*", " ", n)
    return re.sub(r"\s+", " ", n).strip().lower()


def build_l8_index(taxonomy_path=TAXONOMY_PATH):
    """{base_name: 'mechanical'|'physical'|'thermal'|'microstructural'|'electrical'}
    read directly from the taxonomy's L8 shared-block s-tag classes."""
    html = open(taxonomy_path, encoding="utf-8").read()
    index = {}
    for code, label in CATEGORY_NAMES.items():
        for raw in re.findall(r'class="s-tag s-tag-%s"[^>]*>(.*?)</span>' % code, html, re.S):
            index[_base_name(raw)] = label
    return index


# Fallback for L4/L5 (feedstock/process) params: which single L8 category
# they most directly bear on. Only used when a param isn't itself an L8
# name. Picking ONE category (unlike the earlier multi-block draft) —
# first entry wins where a setting could plausibly touch two.
FALLBACK_CATEGORY = {
    # geometry / resolution
    "layer thickness": "physical",
    "layer height": "physical",
    "hatch spacing": "microstructural",
    "nozzle diameter": "physical",
    "spot size": "microstructural",
    "scan rotation": "microstructural",
    "scan strategy": "microstructural",
    "raster angle": "mechanical",
    "infill density": "mechanical",
    "infill pattern": "mechanical",
    "number of perimeters": "mechanical",

    # energy input
    "laser power": "microstructural",
    "scan speed": "microstructural",
    "print speed": "mechanical",
    "travel speed": "physical",
    "energy density": "microstructural",
    "beam current": "microstructural",
    "accelerating voltage": "microstructural",
    "melt pool width": "microstructural",
    "deposition rate": "physical",
    "wire feed speed": "microstructural",
    "dilution ratio": "microstructural",

    # thermal history
    "bed temperature": "microstructural",
    "part bed temperature": "microstructural",
    "chamber temp": "physical",
    "interpass temp": "microstructural",
    "nozzle temperature": "mechanical",
    "print head temp": "physical",
    "sintering temp": "microstructural",
    "sintering time": "microstructural",
    "sintering atmosphere": "microstructural",
    "expected shrinkage": "physical",

    # photopolymer dose
    "exposure time": "mechanical",
    "uv cure dose": "mechanical",
    "light intensity": "mechanical",
    "curing temp": "thermal",
    "curing time": "mechanical",
    "post-cure uv dose": "mechanical",
    "post-cure temp": "thermal",

    # binder jet
    "binder saturation": "physical",
    "debinding temp": "microstructural",
    "debinding time": "physical",

    # feedstock properties
    "d50": "physical",
    "d10": "physical",
    "d90": "physical",
    "particle size d50": "physical",
    "o content": "mechanical",
    "n content": "mechanical",
    "moisture content": "mechanical",
    "apparent density": "physical",
    "tap density": "physical",
    "flowability": "physical",
    "solid loading": "mechanical",
    "filler content": "mechanical",
    "metal/ceramic loading": "mechanical",
    "viscosity": "physical",
    "chemical composition": "mechanical",
    "density": "physical",
    "bulk density": "physical",
}


def classify_param(parameter_name, taxonomy_index):
    base = _base_name(parameter_name)
    if base in taxonomy_index:
        return taxonomy_index[base]
    return FALLBACK_CATEGORY.get(base)  # None if not covered


def tag_rows(rows, taxonomy_path=TAXONOMY_PATH):
    idx = build_l8_index(taxonomy_path)
    out = []
    for r in rows:
        r = dict(r)
        r["property_category"] = classify_param(r.get("parameter", ""), idx)
        out.append(r)
    return out


if __name__ == "__main__":
    import json
    import sys

    rows = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else json.load(sys.stdin)
    if isinstance(rows, dict):
        rows = rows.get("parameters", {}).get("verified", rows)
    print(json.dumps(tag_rows(rows), indent=2, ensure_ascii=False))
