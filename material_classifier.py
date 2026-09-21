"""
Symbolic (non-AI) material classification layer.

Sits right after triage_pipeline.run_triage(). Takes triage's LLM-extracted
`process_categories` / `process_subtypes` / `materials_mentioned` and maps
them to:

  1. a base feedstock material class, pulled from the taxonomy's OWN closed
     vocabulary (am_taxonomy_v4.html's fs-material tags: Metal, Polymer,
     Ceramic, Photopolymer, Composite, Wax, Sand, Paper, Concrete, Bio) via
     lookup on the resolved subtype code — not guessed from free text.
  2. composite/functionalization modifiers found via a deterministic keyword
     scan of materials_mentioned (e.g. "MOF", "nanoparticle", "graphene").
  3. which L8 shared-property blocks (Mechanical / Physical / Thermal /
     Microstructural / Electrical — see am_taxonomy_v4.html's L8 shared
     panel) are worth checking for this paper.

No LLM call anywhere in this module. Pure lookup table + regex, using
materials/categories the triage LLM call already extracted once.
"""
import re

# ---------- 1. Subtype -> base feedstock (form, material class) ----------
# Extracted directly from am_taxonomy_v4.html's feedstock-card tags
# (fs-form / fs-material). Some subtypes offer more than one material
# variant (e.g. VPP-DLP: standard Photopolymer resin OR ceramic-loaded
# resin) — listed with the default first; a later variant is only selected
# if a matching keyword fires (see CERAMIC_LOADED_KEYWORDS below).
SUBTYPE_MATERIAL_VARIANTS = {
    "BJT-M":   [("Powder", "Metal")],
    "BJT-C":   [("Powder", "Ceramic")],
    "BJT-S":   [("Powder", "Sand")],
    "BJT-P":   [("Powder", "Polymer")],
    "DED-LB":  [("Powder", "Metal"), ("Wire", "Metal")],
    "DED-EB":  [("Wire", "Metal")],
    "DED-HY":  [("Powder", "Metal"), ("Wire", "Metal")],
    "PBF-LB":  [("Powder", "Metal")],
    "PBF-EB":  [("Powder", "Metal")],
    "PBF-SLS": [("Powder", "Polymer"), ("Powder", "Composite"), ("Powder", "Ceramic")],
    "PBF-MJF": [("Powder", "Polymer")],
    "MEX-FDM": [("Filament", "Polymer"), ("Filament", "Composite")],
    "MEX-BMD": [("Filament", "Metal"), ("Filament", "Ceramic")],
    "MEX-DIW": [("Paste", "Polymer"), ("Paste", "Ceramic"), ("Paste", "Bio"), ("Paste", "Concrete")],
    "MEX-SCR": [("Pellet", "Polymer"), ("Pellet", "Composite")],
    "MJT-PJ":  [("Liquid Droplet", "Photopolymer"), ("Liquid Droplet", "Wax")],
    "MJT-NPJ": [("Liquid Droplet", "Ceramic")],
    "MJT-DOD": [("Liquid Droplet", "Wax"), ("Liquid Droplet", "Metal")],
    "SHL-LOM": [("Sheet", "Paper"), ("Sheet", "Polymer"), ("Sheet", "Ceramic")],
    "SHL-UAM": [("Foil", "Metal")],
    "SHL-SDL": [("Sheet", "Polymer")],
    "VPP-SLA":  [("Resin", "Photopolymer")],
    "VPP-DLP":  [("Resin", "Photopolymer"), ("Resin", "Ceramic")],
    "VPP-CLIP": [("Resin", "Photopolymer")],
    "VPP-2PP":  [("Resin", "Photopolymer")],
    "VPP-MSLA": [("Resin", "Photopolymer")],
}

# Fallback when no subtype was resolved — best-guess default per L1 category.
CATEGORY_DEFAULT_MATERIAL = {
    "BJT": ("Powder", "Metal"),
    "DED": ("Wire", "Metal"),
    "PBF": ("Powder", "Metal"),
    "MEX": ("Filament", "Polymer"),
    "MJT": ("Liquid Droplet", "Photopolymer"),
    "SHL": ("Sheet", "Polymer"),
    "VPP": ("Resin", "Photopolymer"),
}

# ---------- 2. Composite / functionalization indicator keywords ----------
# Deterministic keyword scan of triage's materials_mentioned for signs the
# base feedstock has been loaded/functionalized with a second material
# class — which changes which L8 property blocks are worth checking.
CERAMIC_LOADED_KEYWORDS = [
    "alumina", "zirconia", "silica", "hydroxyapatite",
    "ceramic-loaded", "ceramic loaded", "al2o3", "zro2", "sio2",
]
METAL_LOADED_KEYWORDS = [
    "metal-organic framework", "metal organic framework", "mof",
    "nanoparticle", "nano-particle", "silver nanoparticle", "gold nanoparticle",
    "copper nanoparticle", "metal nanoparticle", "metal powder loaded",
]
CARBON_FILLER_KEYWORDS = [
    "graphene", "carbon nanotube", "cnt", "carbon fiber", "carbon fibre",
    "carbon black",
]
COMPOSITE_KEYWORDS = CERAMIC_LOADED_KEYWORDS + METAL_LOADED_KEYWORDS + CARBON_FILLER_KEYWORDS + [
    "composite", "filler", "reinforced", "loaded resin", "hybrid material",
]

ELECTRICAL_KEYWORDS = [
    "conductive", "conductivity", "electrode", "sensor", "circuit",
    "dielectric", "piezoelectric", "semiconductor", "resistive", "capacitive",
]

# ---------- 2b. L9 (post-processing) indicator keywords ----------
# The taxonomy's own shared panel (am_taxonomy_v4.html) renders L8 Properties
# and L9 Post-Processing together as ONE block — "L8 + L9 — Final Properties
# & Post-Processing" — so they're combined here too instead of tracked as
# separate levels. Block labels below are copied VERBATIM from the taxonomy's
# shared-block-title text so they can be cross-referenced directly against
# taxonomy section names downstream.
#
# Unlike L8 (a material-composition question, answerable from
# materials_mentioned alone), whether an L9 post-processing step happened is
# a PROCESS question — described in prose ("annealed at...", "polished
# with...", "12 h of UV curing"), not in the materials list. So these
# keywords are meant to be scanned against actual paper/section text, not
# just materials_mentioned (see classify_material's `scan_text` argument).
L8_MECHANICAL = "L8 \u00b7 Mechanical Properties"
L8_PHYSICAL = "L8 \u00b7 Physical Properties"
L8_THERMAL = "L8 \u00b7 Thermal Properties"
L8_MICROSTRUCTURAL = "L8 \u00b7 Microstructural Properties"
L8_ELECTRICAL = "L8 \u00b7 Electrical Properties"

L9_HEAT_TREATMENT = "L9 \u00b7 Heat Treatment"
L9_SURFACE_TREATMENT = "L9 \u00b7 Surface Treatment"
L9_COATING = "L9 \u00b7 Coating"
L9_INSPECTION = "L9 \u00b7 Inspection / Characterisation"
L9_STERILISATION = "L9 \u00b7 Sterilisation (Biomedical)"

BASELINE_L8_BLOCKS = [L8_MECHANICAL, L8_PHYSICAL, L8_THERMAL]

# Post-cure / UV-cure keywords deliberately live under Heat Treatment: a
# bulk post-print UV cure (hours-scale, done in a UV chamber after printing)
# is functionally the polymer-AM analogue of Heat Treatment's Annealing/HIP
# steps — a bake to finish the part, not the taxonomy's L5 "Exposure Time
# (s)" process parameter, which is the seconds-scale per-layer light dose
# during printing itself. Same word, different pipeline stage.
POSTCURE_KEYWORDS = [
    "post-cure", "post cure", "postcure", "post-curing", "postcuring",
    "uv curing", "uv chamber", "uv light chamber", "uv oven",
]
HEAT_TREATMENT_KEYWORDS = POSTCURE_KEYWORDS + [
    "anneal", "annealing", "aging", " hip ", "hot isostatic press",
    "solution treat", "heat treatment", "heat-treated", "heat treated",
]
SURFACE_TREATMENT_KEYWORDS = [
    "polished", "sandblasted", "shot peened", "electropolished",
    "chemically etched", "machined", "laser polished",
]
COATING_KEYWORDS = [
    "pvd", "cvd", "anodis", "anodiz", "coating", "powder coat", "painted",
]
INSPECTION_KEYWORDS = [
    " sem ", "xrd", "ebsd", "ct scan", "tensile test", "hardness test",
    "fatigue test", "cmm", "optical microscopy", "characterised",
    "characterized",
]
STERILISATION_KEYWORDS = [
    "sterili", "autoclave", " eto ", "gamma irradiation",
    "plasma steriliz", "iso 13485", "biocompatib",
]


def _text_hit(keywords, *texts):
    joined = " " + " ".join(str(t).lower() for t in texts if t) + " "
    return sorted({kw.strip() for kw in keywords if kw in joined})


def _resolve_subtype_code(categories, subtypes_text):
    """Lightweight symbolic match of triage's free-text process_subtypes
    against known L2 codes — same substring-match style already used by
    agentic_pipeline.match_subtype(), kept self-contained here so this
    module has no import dependency on agentic_pipeline."""
    for code in SUBTYPE_MATERIAL_VARIANTS:
        cat_prefix, suffix = code.split("-", 1)
        if categories and cat_prefix not in categories:
            continue
        suffix_norm = suffix.lower()
        for st in subtypes_text:
            st_norm = st.lower().replace(" ", "").replace("-", "")
            if suffix_norm in st_norm or st_norm in suffix_norm:
                return code
    return None


def classify_material(triage, resolved_subtype_code=None, scan_text=None):
    """
    triage: dict returned by triage_pipeline.run_triage() (uses
            process_categories, process_subtypes, materials_mentioned).
    resolved_subtype_code: optional L2 code (e.g. "VPP-SLA") if the caller
            already resolved one (agentic_pipeline.match_subtype); if
            omitted, this function does its own lightweight match.
    scan_text: optional full/relevant paper text (e.g. get_relevant_text()'s
            output) to scan for L9 post-processing keywords ("annealed at",
            "12 h of UV curing", "polished with"...). L9 is a PROCESS
            question, not a materials-list question, so without this the L9
            scan falls back to materials_mentioned only and will usually
            find nothing — pass the paper text for a real L9 read.

    Returns:
      {
        "base_form": "Resin",
        "base_material_class": "Photopolymer",
        "composite_modifiers": ["mof", ...],
        "final_material_class": "Photopolymer (mof-functionalized composite)",
        "applicable_l8_l9_blocks": [
            "L8 \u00b7 Mechanical Properties", "L8 \u00b7 Physical Properties",
            "L8 \u00b7 Thermal Properties", "L9 \u00b7 Heat Treatment", ...
        ],
      }
    Pure lookup + regex — no LLM call. L8 and L9 are returned as one merged
    list, matching the taxonomy's own shared panel (they're the same block
    in am_taxonomy_v4.html, not two separate levels to track).
    """
    categories = triage.get("process_categories", [])
    subtypes_text = triage.get("process_subtypes", [])
    materials = triage.get("materials_mentioned", [])

    code = resolved_subtype_code or _resolve_subtype_code(categories, subtypes_text)

    if code and code in SUBTYPE_MATERIAL_VARIANTS:
        variants = SUBTYPE_MATERIAL_VARIANTS[code]
    elif categories and categories[0] in CATEGORY_DEFAULT_MATERIAL:
        variants = [CATEGORY_DEFAULT_MATERIAL[categories[0]]]
    else:
        variants = [("Unknown", "Unknown")]

    base_form, base_material = variants[0]

    # --- L8: material-composition signals, scanned from materials_mentioned ---
    ceramic_hits = _text_hit(CERAMIC_LOADED_KEYWORDS, *materials)
    metal_hits = _text_hit(METAL_LOADED_KEYWORDS, *materials)
    carbon_hits = _text_hit(CARBON_FILLER_KEYWORDS, *materials)
    composite_hits = sorted(set(ceramic_hits + metal_hits + carbon_hits))
    electrical_hits = _text_hit(ELECTRICAL_KEYWORDS, *materials)

    # If a ceramic-loaded variant exists for this subtype and ceramic
    # keywords fired, prefer that variant over the plain default.
    if ceramic_hits:
        for form, mat in variants:
            if mat == "Ceramic":
                base_form, base_material = form, mat
                break

    l8_blocks = list(BASELINE_L8_BLOCKS)
    if composite_hits:
        l8_blocks.append(L8_MICROSTRUCTURAL)
    if electrical_hits:
        l8_blocks.append(L8_ELECTRICAL)

    # --- L9: process-step signals, scanned from paper text (falls back to
    # materials_mentioned if no paper text given, which will usually find
    # nothing — post-processing steps are described in prose, not listed
    # as materials) ---
    scan_sources = [scan_text] if scan_text else list(materials)
    l9_blocks = []
    if _text_hit(HEAT_TREATMENT_KEYWORDS, *scan_sources):
        l9_blocks.append(L9_HEAT_TREATMENT)
    if _text_hit(SURFACE_TREATMENT_KEYWORDS, *scan_sources):
        l9_blocks.append(L9_SURFACE_TREATMENT)
    if _text_hit(COATING_KEYWORDS, *scan_sources):
        l9_blocks.append(L9_COATING)
    if _text_hit(INSPECTION_KEYWORDS, *scan_sources):
        l9_blocks.append(L9_INSPECTION)
    if _text_hit(STERILISATION_KEYWORDS, *scan_sources):
        l9_blocks.append(L9_STERILISATION)

    if composite_hits:
        modifier_label = "/".join(composite_hits)
        final_class = f"{base_material} ({modifier_label}-functionalized composite)"
    else:
        final_class = base_material

    return {
        "base_form": base_form,
        "base_material_class": base_material,
        "composite_modifiers": composite_hits,
        "final_material_class": final_class,
        "applicable_l8_l9_blocks": l8_blocks + l9_blocks,
    }


if __name__ == "__main__":
    import json
    import sys

    triage = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else json.load(sys.stdin)
    scan_text = open(sys.argv[2], encoding="utf-8").read() if len(sys.argv) > 2 else None
    print(json.dumps(classify_material(triage, scan_text=scan_text), indent=2))
