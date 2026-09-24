"""
Agentic extraction framework — SKILL-based, not multi-agent.

Follows the AtomisticSkills pattern: ONE orchestrating agent (a single LLM
call loop), and a library of per-category SKILLS it loads on demand. A skill
is not a separate running agent — it's a scoped bundle of taxonomy knowledge
+ instructions that gets injected into the orchestrator's context only when
that category is relevant. This keeps each call's context small (only the
category's own parameters, not all 373) without spinning up N separate agents.

Flow:
  1. Triage identifies which categories (skills) are relevant for this paper.
  2. Orchestrator loads only those skills' schemas.
  3. For each relevant skill, orchestrator makes one extraction call scoped
     to that skill's taxonomy subtree.
  4. Results merged, each row tagged with which skill produced it.
"""
import json
from dataclasses import dataclass, field

from llm_pipeline import call_gpt_mini, parse_response, verify_materials, verify_parameters


# ---------- 1. Skill definition ----------
@dataclass
class CategorySkill:
    code: str                      # e.g. "PBF" — skill id
    name: str                      # e.g. "Powder Bed Fusion" — skill description
    param_sections: dict           # {section_name: [param strings]} — this skill's scope
    synonyms: dict = field(default_factory=dict)     # {taxonomy_param: [common paper phrasings]}
    domain_hints: list = field(default_factory=list) # short bullet tips specific to this category

    def build_schema_text(self):
        lines = []
        for section, params in self.param_sections.items():
            lines.append(f"{section}:")
            lines.append("  " + ", ".join(params))
        return "\n".join(lines)

    def build_synonym_text(self):
        if not self.synonyms:
            return ""
        lines = ["Known alternate phrasings for this category (map these to the taxonomy name):"]
        for canonical, alts in self.synonyms.items():
            lines.append(f"  - {canonical} <- {', '.join(alts)}")
        return "\n".join(lines)

    def build_hints_text(self):
        if not self.domain_hints:
            return ""
        return "Category-specific extraction notes:\n" + "\n".join(f"  - {h}" for h in self.domain_hints)


# ---------- 2. Skill library, built from your taxonomy tree + hand-curated specifics ----------
# Per-category domain knowledge — this is what makes each skill actually specific,
# rather than the same generic template with a different vocabulary list swapped in.
CATEGORY_SPECIFICS = {
    "PBF": {
        "synonyms": {
            "Scan Speed": ["scanning speed", "laser scan velocity", "hatch velocity", "laser speed"],
            "Layer Thickness": ["build layer height", "powder layer thickness", "slice thickness"],
            "Laser Power": ["irradiation power", "beam power"],
        },
        "domain_hints": [
            "Laser power and scan speed are very often reported together in one clause "
            "(e.g. '200 W and 1000 mm/s') — check for both even if only one is named explicitly nearby.",
            "Hatch spacing and layer thickness are frequently listed in the same sentence as a parameter set.",
            "Watch for energy density (e.g. J/mm^3) reported instead of raw laser power/speed — "
            "if only energy density is given, note it but do not back-calculate power/speed.",
        ],
    },
    "DED": {
        "synonyms": {
            "Laser Power": ["deposition power", "beam power"],
            "Travel Speed": ["scan speed", "traverse speed", "deposition speed"],
            "Wire Feed Rate": ["feed rate", "wire feed speed"],
        },
        "domain_hints": [
            "Wire-fed and powder-fed DED report different feedstock parameters — check which "
            "feedstock type is used before assuming which parameter set applies.",
            "Travel speed and wire/powder feed rate are usually reported as a paired ratio "
            "(feed-to-travel-speed ratio) — extract both values even if only the ratio is emphasized.",
        ],
    },
    "MEX": {
        "synonyms": {
            "Nozzle Temperature": ["extrusion temperature", "print head temperature", "hot end temperature"],
            "Print Speed": ["extrusion speed", "printing speed", "travel speed"],
        },
        "domain_hints": [
            "Bed temperature and nozzle temperature are usually both reported — don't stop after finding one.",
            "Infill percentage and pattern are often in a separate sentence from thermal parameters.",
        ],
    },
    "VPP": {
        "synonyms": {
            "Exposure Time": ["cure time", "curing time", "UV curing", "irradiation time", "UV exposure duration"],
            "Light Intensity": ["UV intensity", "irradiance", "power density"],
            "XY Resolution": ["print resolution", "XY resolution", "planar resolution"],
            "Z Resolution": ["layer resolution", "Z resolution", "vertical resolution"],
        },
        "domain_hints": [
            "VPP papers often report intensity in mW/cm^2 rather than W — do not convert units, "
            "report as stated.",
            "Layer cure time and light intensity are the two parameters most often paired together.",
            "Cure/curing time is very often reported as a duration alone without the word 'time' "
            "attached (e.g. '40 s' or 'a base curing time of 40 s') — check numeric durations near "
            "any mention of curing/UV/exposure even if not explicitly labeled a parameter.",
            "XY resolution and Z (layer) resolution are frequently reported in micrometers as a "
            "print-quality spec, separate from layer thickness — extract both if given. Watch for "
            "the parenthetical phrasing pattern '(XY around 50 µm)' or '(XYZ around 15 µm)', where "
            "the value sits inside a qualitative descriptor like 'Good resolution (...)' rather than "
            "next to the parameter name directly.",
            "Resin viscosity is often described qualitatively ('low viscosity resin preferred') as "
            "well as with a numeric value — capture the qualitative statement too if no number is given, "
            "flagging confidence as medium/low.",
        ],
    },
    "BJT": {
        "synonyms": {
            "Binder Saturation": ["saturation level", "binder saturation ratio"],
        },
        "domain_hints": [
            "Powder D10/D50/D90 particle size values are usually reported as a set — check for all three.",
            "Curing/sintering parameters are often in a separate post-processing paragraph, not near the "
            "binder jetting parameters themselves.",
        ],
    },
    "MJT": {
        "synonyms": {
            "Droplet Volume": ["droplet size", "jetted droplet volume"],
            "XY Resolution": ["jetting resolution", "print resolution"],
            "Z Resolution": ["layer resolution"],
        },
        "domain_hints": [
            "Droplet volume and jetting frequency are usually reported together.",
            "PolyJet/MultiJet papers commonly use dual materials — a 'build material' (the actual part) "
            "and a 'support material' (removed after printing) — extract both material names and tag "
            "which role each plays if stated.",
            "Support removal method (e.g. water jetting, soaking, dissolving) is often reported as a "
            "separate post-processing step, not near the print parameters themselves.",
            "A roller/leveling mechanism between layers is specific to inkjet-based (PolyJet) printing — "
            "if mentioned, it usually accompanies build/support material ratio details nearby.",
        ],
    },
    "SHL": {"synonyms": {}, "domain_hints": [
        "Sheet thickness and bonding parameters (e.g. ultrasonic amplitude, weld temperature) are usually "
        "reported in separate sentences — check both.",
    ]},
    "OTHER": {"synonyms": {}, "domain_hints": []},
}

def build_skill_library(vocab_tree_path="vocab_tree.json"):
    """Returns dict {category_code: CategorySkill} — one skill per AM process category,
    each with its own domain-specific synonyms and extraction hints."""
    tree = json.load(open(vocab_tree_path))
    skills = {}
    for code, data in tree.items():
        l1 = data.get("L1")
        if not l1:
            continue
        param_sections = {k: v for k, v in data["params"].items() if k != "_shared"}
        param_sections.update(data["params"].get("_shared", {}))
        specifics = CATEGORY_SPECIFICS.get(code, {"synonyms": {}, "domain_hints": []})
        skills[code] = CategorySkill(
            code=code,
            name=l1,
            param_sections=param_sections,
            synonyms=specifics["synonyms"],
            domain_hints=specifics["domain_hints"],
        )
    return skills


# ---------- 3. L2 SUBTYPE skills — finer-grained than the L1 category skills above ----------
# Maps each L2 subtype code to the taxonomy section(s) that actually apply to it.
# A subtype only needs its OWN process-parameter section + whichever feedstock
# section matches it, not every section under the parent category.
SUBTYPE_SECTION_MAP = {
    # VPP subtypes
    "VPP-SLA":  ["Process Parameters — VAT-SLA", "Feedstock Properties — Resin"],
    "VPP-DLP":  ["Process Parameters — DLP", "Feedstock Properties — Resin"],
    "VPP-CLIP": ["Process Parameters — CLIP / CDLP", "Feedstock Properties — Ceramic-Loaded Resin",
                 "Feedstock Properties — Resin"],
    "VPP-2PP":  ["Process Parameters — Two-Photon Polymerization",
                 "Feedstock Properties — Nonlinear Photoresist (2PP)"],
    "VPP-MSLA": ["Process Parameters — MSLA / Mono LCD", "Feedstock Properties — Resin"],
    # MJT subtypes
    "MJT-PJ":   ["Process Parameters — MJ-PJ", "Feedstock Properties — Resin/Droplet"],
    "MJT-NPJ":  ["Process Parameters — Material Jetting", "Feedstock Properties — Nanoparticle Suspension"],
    "MJT-DOD":  ["Process Parameters — Material Jetting", "Process Parameters — Liquid Metal Jetting",
                 "Feedstock Properties — Wax", "Feedstock Properties — Liquid Metal / Metal Ink"],
}

# Subtype-specific overrides/additions on top of the parent category's synonyms/hints.
# Only add entries here once you've confirmed them against real papers for that subtype —
# left empty means "inherits parent VPP/MJT knowledge only," not "nothing to extract."
SUBTYPE_SPECIFICS = {
    "VPP-DLP": {
        "synonyms": {"Exposure Time": ["projection time", "pixel exposure time"]},
        "domain_hints": ["DLP projects a full layer at once (per-layer exposure time), "
                          "unlike laser-scanned SLA — do not confuse layer exposure time with "
                          "per-point/scan-based timing terminology from SLA papers."],
    },
    "VPP-SLA": {
        "synonyms": {"Scan Speed": ["laser scan speed", "galvo scan speed"]},
        "domain_hints": ["Laser-based SLA reports scan speed (mm/s) in addition to exposure time — "
                          "DLP/MSLA subtypes do not have this parameter since they expose a full layer at once."],
    },
    "MJT-PJ": {
        "synonyms": {},
        "domain_hints": ["PolyJet dual-material (build+support) terminology applies specifically "
                          "here — see general MJT hints for build/support material extraction."],
    },
}

def build_subtype_skill_library(vocab_tree_path="vocab_tree.json"):
    """
    Returns dict {L2_subtype_code: CategorySkill}, e.g. {"VPP-SLA": ..., "VPP-DLP": ...}.
    Each subtype skill is scoped to ONLY its own process-parameter/feedstock sections
    (via SUBTYPE_SECTION_MAP) plus the parent category's shared sections (post-processing etc.),
    and inherits the parent category's synonyms/hints merged with any subtype-specific overrides.
    """
    tree = json.load(open(vocab_tree_path))
    skills = {}
    for l1_code, data in tree.items():
        l2_map = data.get("L2", {})
        shared = data["params"].get("_shared", {})
        parent_specifics = CATEGORY_SPECIFICS.get(l1_code, {"synonyms": {}, "domain_hints": []})

        for l2_code, l2_name in l2_map.items():
            section_names = SUBTYPE_SECTION_MAP.get(l2_code)
            if not section_names:
                continue  # no mapping defined yet for this subtype — skip rather than guess

            param_sections = {}
            for sec in section_names:
                if sec in data["params"]:
                    param_sections[sec] = data["params"][sec]
            param_sections.update(shared)  # shared sections (post-processing etc.) always included

            overrides = SUBTYPE_SPECIFICS.get(l2_code, {"synonyms": {}, "domain_hints": []})
            merged_synonyms = {**parent_specifics["synonyms"], **overrides["synonyms"]}
            merged_hints = parent_specifics["domain_hints"] + overrides["domain_hints"]

            skills[l2_code] = CategorySkill(
                code=l2_code,
                name=f"{l2_name} ({l1_code})",
                param_sections=param_sections,
                synonyms=merged_synonyms,
                domain_hints=merged_hints,
            )
    return skills


def match_subtype(category_code, process_subtypes, subtype_skills):
    """
    Tries to match triage's free-text process_subtypes (e.g. ["LPBF", "DLP"]) against
    the L2 subtype skills available for THIS category (e.g. VPP-SLA, VPP-DLP, ...).
    Returns the matched CategorySkill, or None if no confident match found.

    Matching is deliberately conservative (substring match on subtype code suffix or
    name) — a wrong subtype match is worse than falling back to the broader L1 skill,
    since a wrong L2 skill's taxonomy may not even contain the right parameters.
    """
    candidates = {code: skill for code, skill in subtype_skills.items()
                  if code.startswith(category_code + "-")}
    if not candidates:
        return None

    for subtype_text in process_subtypes:
        text_lower = subtype_text.lower().replace(" ", "").replace("-", "")
        for l2_code, skill in candidates.items():
            suffix = l2_code.split("-", 1)[1].lower()  # e.g. "sla", "dlp"
            name_lower = skill.name.lower().replace(" ", "").replace("-", "")
            if suffix in text_lower or text_lower in name_lower:
                return skill
    return None


def find_subtype_anywhere(process_subtypes, subtype_skills):
    """
    Searches ALL L2 skills (across every category), not just a pre-guessed one.
    Used to catch triage errors where the subtype guess is right but the parent
    category guess is wrong (e.g. category="MJT", subtype="Stereolithography" —
    Stereolithography is actually a VPP subtype). Returns the matched CategorySkill
    or None.
    """
    for subtype_text in process_subtypes:
        text_lower = subtype_text.lower().replace(" ", "").replace("-", "")
        for l2_code, skill in subtype_skills.items():
            suffix = l2_code.split("-", 1)[1].lower()
            name_lower = skill.name.lower().replace(" ", "").replace("-", "")
            if suffix in text_lower or text_lower in name_lower:
                return skill
    return None


def reconcile_categories(triage_result, subtype_skills):
    """
    Cross-checks triage's process_categories against process_subtypes and corrects
    mismatches: if a named subtype doesn't belong to any guessed category, but DOES
    match a subtype under a different category, that category is added (the wrong
    one is left in place rather than removed, since triage might still be right that
    the paper touches that category too — dispatch will just find nothing useful there).

    Returns a (possibly extended) list of category codes, and prints a warning for
    every correction made so you can see when triage's category guess was off.
    """
    categories = list(triage_result.get("process_categories", []))
    subtypes = triage_result.get("process_subtypes", [])

    for subtype_text in subtypes:
        # does this subtype already belong to one of the guessed categories?
        already_covered = any(
            match_subtype(cat, [subtype_text], subtype_skills) for cat in categories
        )
        if already_covered:
            continue

        # search everywhere for the real parent category
        found_skill = find_subtype_anywhere([subtype_text], subtype_skills)
        if found_skill:
            real_category = found_skill.code.split("-", 1)[0]
            if real_category not in categories:
                print(f"Note: triage listed subtype '{subtype_text}' under {categories}, "
                      f"but it actually belongs to {real_category} — adding {real_category} "
                      f"to process_categories.")
                categories.append(real_category)

    return categories


# ---------- 3. Extraction prompt for a loaded skill ----------
PINNED_CATEGORY_PROMPT = """You are an orchestrating extraction agent. You have just loaded the
"{category_name}" ({category_code}) skill — a scoped taxonomy of parameters
relevant to this AM process category. Use ONLY this skill's parameters for
this pass; do not use knowledge from other process categories.

The material has ALREADY BEEN IDENTIFIED for you: everything you extract in
this pass is about "{target_material}" specifically, processed via
{category_code}. Do NOT extract parameters belonging to any other material
mentioned in the paper (e.g. reagents, solvents, or a second material the
paper compares against) — only {target_material}. Because the material is
fixed, do NOT include a "material" field in your output rows.

Extract every value from the paper below that matches a parameter in this
skill's taxonomy AND belongs to {target_material}'s process. Return ONLY
valid JSON, no prose, in this shape:

{{
  "extracted_parameters": [
    {{
      "parameter": "<exact taxonomy name>",
      "value": "...",
      "unit": "...",
      "quote": "..."
    }}
  ]
}}

If the paper does not report any parameters from this skill's taxonomy for
{target_material} — for example, because it is a review, or because
{category_code} is used only incidentally and no process parameters are
reported — return: {{"extracted_parameters": []}}

An empty result is the CORRECT answer in that case, not a failure.

Rules:
- Only use parameter names from the {category_code} skill's taxonomy below.
- The "quote" must be an exact sentence/clause copied from the paper, containing the value.
- Do not include parameters the paper doesn't report. Do not fabricate.
- Every extracted row is about {target_material}. If a value clearly belongs
  to a different material, skip it — do not extract it under this pass.

**CRITICAL — how to handle absent parameters:**
- If the paper does not mention a parameter at all, OMIT it entirely.
- If the paper mentions a parameter but gives no numeric value (e.g. "at an
  adaptable layer height"), return it with `"value": null` and add
  `"note": "mentioned but no value reported"`. Do NOT put "not reported" or
  any other string into `value`.
- NEVER emit a row with an empty `value` AND empty `quote`. If you have
  nothing, return `{{"extracted_parameters": []}}` or omit the parameter —
  an empty row is treated as a failed extraction.
- Do not fabricate quotes. Every `quote` must be a contiguous substring of the
  paper text. Do not stitch together two sentences from different sections.
- Do not pull numbers from non-printing sections (e.g. stirring speeds from a
  chemical extraction protocol) and label them as print parameters. If the
  quote is not from a section describing the printing process, do not extract.

{synonym_text}

{hints_text}

{category_code} SKILL TAXONOMY:
{schema}

PAPER TEXT:
{paper_text}
"""

CATEGORY_PROMPT = """You are an orchestrating extraction agent. You have just loaded the
"{category_name}" ({category_code}) skill — a scoped taxonomy of parameters
relevant to this AM process category. Use ONLY this skill's parameters for
this pass; do not use knowledge from other process categories.

Extract every value from the paper below that matches a parameter in this
skill's taxonomy. Return ONLY valid JSON, no prose, in this shape:

{{
  "materials": [
    {{"material": "...", "material_form": "...", "quote": "..."}}
  ],
  "extracted_parameters": [
    {{
      "parameter": "<exact taxonomy name>",
      "value": "...",
      "unit": "...",
      "material": "...",
      "quote": "..."
    }}
  ]
}}

If the paper does not report any parameters from this skill's taxonomy —
for example, because it is a review, or because it uses this 3D printing
technique only incidentally and reports no process parameters — return:

{{"materials": [], "extracted_parameters": []}}

An empty result is the CORRECT answer in that case, not a failure.

Rules:
- Only use parameter names from the {category_code} skill's taxonomy below.
- The "quote" must be an exact sentence/clause copied from the paper, containing the value.
- Do not include parameters the paper doesn't report. Do not fabricate.

**CRITICAL — how to handle absent parameters:**
- If the paper does not mention a parameter at all, OMIT it entirely.
- If the paper mentions a parameter but gives no numeric value (e.g. "at an
  adaptable layer height"), return it with `"value": null` and add
  `"note": "mentioned but no value reported"`. Do NOT put "not reported" or
  any other string into `value`.
- NEVER emit a row with an empty `value` AND empty `quote`. If you have
  nothing, return `{{"materials": [], "extracted_parameters": []}}` or omit
  the parameter — an empty row is treated as a failed extraction.
- Do not fabricate quotes. Every `quote` must be a contiguous substring of the
  paper text. Do not stitch together two sentences from different sections.
- Do not pull numbers from non-printing sections (e.g. stirring speeds from a
  chemical extraction protocol) and label them as print parameters. If the
  quote is not from a section describing the printing process, do not extract.

{synonym_text}

{hints_text}

{category_code} SKILL TAXONOMY:
{schema}

PAPER TEXT:
{paper_text}
"""

REVIEW_CATEGORY_PROMPT = """You are an extraction agent working on a REVIEW or SURVEY paper.
Your job is to harvest every value the paper reports about "{category_name}"
({category_code}) processes — including values the review authors state as
general facts AND values they attribute to other cited papers. Reviews are
high-value sources for this task; do not skip them.

Return ONLY valid JSON, no prose, in this shape:

{{
  "materials": [
    {{"material": "...", "material_form": "...", "quote": "..."}}
  ],
  "extracted_parameters": [
    {{
      "parameter": "<exact taxonomy name>",
      "value": "...",
      "unit": "...",
      "material": "",
      "context": "<column header + row label, or short sentence fragment containing the value>",
      "quote": "",
      "provenance": {{
        "type": "general-author-statement | cited-secondary | primary-measurement",
        "cited_work": "<author-year or ref marker, or empty>",
        "citation_marker": "<e.g. [62], or empty>"
      }}
    }}
  ]
}}

What to extract:

1. General statements by the review authors (provenance.type = "general-author-statement").
   Example: "Commercially available SLA printers typically achieve XY resolutions
   of around 50 μm and Z resolutions around 10 μm." → extract
   parameter = "XY Resolution (μm)", value = "50", context = the sentence,
   provenance.type = "general-author-statement".

2. Values attributed to other papers (provenance.type = "cited-secondary").
   Example: "Scotti et al. used SLS with 316L stainless steel powder (median
   particle size of 31 μm) [80]." → extract parameter = "D50 (μm)", value = "31",
   provenance.type = "cited-secondary", provenance.cited_work = "Scotti et al. 2019",
   provenance.citation_marker = "[80]".

3. Values in TABLES. This is important. Table cells do not appear in any sentence,
   so set "quote" to "" and use "context" instead. Set "context" to the column
   header + row label, e.g. "SLA / XY resolution (μm)". Column header is
   parameter name; cell content is value.
   For tables rendered as:
       [TABLE: 3D printing technologies and materials for MS]
       Columns: Technology | Material | Strengths | Limitations
       Row: SLA | Silica Glass | Outstanding thermal stability; ... | Good resolution (XY around 50 μm, Z around 10 μm)
   → extract parameter = "XY Resolution (μm)", value = "50",
     context = "SLA / XY resolution (μm)", provenance.type = "general-author-statement".

4. Primary measurements by the review authors themselves, if the review
   also reports its own experiments (provenance.type = "primary-measurement").

Rules:
- Only use parameter names from the {category_code} skill's taxonomy below.
- Every row MUST have a non-empty "value".
- Every row MUST have either a non-empty "quote" (exact sentence) OR a non-empty
  "context" (column header + row label, or short fragment).
- Every row MUST have a "provenance" object with "type" set to one of the three
  allowed values.
- Do NOT fabricate. If a number doesn't appear in the paper, don't report it.
- Return an empty list if nothing matches.

{synonym_text}

{hints_text}

{category_code} SKILL TAXONOMY:
{schema}

PAPER TEXT:
{paper_text}
"""

def build_skill_prompt(skill, paper_text, source_kind="research-article", target_material=None):
    if target_material and source_kind != "review":
        return PINNED_CATEGORY_PROMPT.format(
            category_name=skill.name,
            category_code=skill.code,
            target_material=target_material,
            schema=skill.build_schema_text(),
            synonym_text=skill.build_synonym_text(),
            hints_text=skill.build_hints_text(),
            paper_text=paper_text,
        )
    template = REVIEW_CATEGORY_PROMPT if source_kind == "review" else CATEGORY_PROMPT
    return template.format(
        category_name=skill.name,
        category_code=skill.code,
        schema=skill.build_schema_text(),
        synonym_text=skill.build_synonym_text(),
        hints_text=skill.build_hints_text(),
        paper_text=paper_text,
    )

# ---------- 4. Orchestrator invokes ONE skill for ONE call ----------

def _is_hollow_row(row, source_kind="research-article"):
    value = str(row.get("value", "") or "").strip()
    if not value:
        return True
    if source_kind == "review":
        quote = str(row.get("quote", "") or "").strip()
        context = str(row.get("context", "") or "").strip()
        return not quote and not context
    quote = str(row.get("quote", "") or "").strip()
    return not quote

def _is_explicitly_missing(row):
    """Model said 'not reported in the paper' — treat as a null, not a value."""
    value = str(row.get("value", "") or "").strip().lower()
    return value in {"not reported", "not reported in the paper", "n/a", "na",
                     "none", "not stated", "not specified", "not given", "unknown"}


def _build_param_to_section_index(skill):
    """{exact taxonomy parameter name -> section name} for this skill, built
    from skill.param_sections. Pure dict — no LLM call. Used to tag every
    extracted row with WHICH taxonomy section (L4/L5/L8/L9/...) it actually
    came from, since the flat prompt sent to the model loses that grouping
    and the model's response is just a bare 'parameter' string with no
    section attached."""
    index = {}
    for section_name, params in skill.param_sections.items():
        for p in params:
            index[p] = section_name
    return index


def _tag_taxonomy_section(rows, param_to_section):
    """Adds 'taxonomy_section' to each row: the exact section name if the
    row's 'parameter' string matches a real taxonomy tag for this skill,
    else 'NOT IN SCHEMA (parameter name not found in this skill's taxonomy
    — check for a hallucinated/mismatched tag)'. Mutates and returns rows."""
    for row in rows:
        row["taxonomy_section"] = param_to_section.get(
            row.get("parameter"),
            "NOT IN SCHEMA (parameter name not found in this skill's taxonomy)",
        )
    return rows


def invoke_skill(skill, paper_text, source_kind="research-article", model="gpt-5-mini",
                  target_material=None):
    """
    target_material: if set (and source_kind != "review"), the material is
    PINNED before extraction even starts — build_skill_prompt uses
    PINNED_CATEGORY_PROMPT, which doesn't ask the model for a "material"
    field at all (nothing to get wrong). Every returned row gets
    target_material force-set as its material, and verify_parameters is
    given known_materials={target_material} directly — there is no
    ambiguity left to verify, so the old "material not confirmed" flag
    class is structurally eliminated for this path rather than patched
    around after the fact.
    """
    prompt = build_skill_prompt(skill, paper_text, source_kind=source_kind,
                                 target_material=target_material)
    raw = call_gpt_mini(prompt, model=model)
    parsed = parse_response(raw)

    pinned = bool(target_material) and source_kind != "review"

    if pinned:
        mat_verified = [{"material": target_material, "material_form": "", "quote": "",
                          "verified": True, "note": "pinned by triage's printed_materials"}]
        mat_flagged = []
        known_materials = {target_material}
    else:
        mat_verified, mat_flagged = verify_materials(parsed.get("materials", []), paper_text)
        known_materials = {m["material"] for m in mat_verified}

    raw_params = parsed.get("extracted_parameters", [])
    if pinned:
        for r in raw_params:
            r["material"] = target_material  # force — model wasn't even asked for this field

    hollow = [r for r in raw_params if _is_hollow_row(r, source_kind)]
    missing = [r for r in raw_params if _is_explicitly_missing(r)]
    real = [r for r in raw_params if not _is_hollow_row(r, source_kind) and not _is_explicitly_missing(r)]

    if hollow:
        print(f"Warning: {len(hollow)} hollow rows discarded from skill {skill.code}.")
    if missing:
        print(f"Note: {len(missing)} rows marked 'not reported' from skill {skill.code}.")

    param_verified, param_flagged = verify_parameters(
        real, paper_text, known_materials, source_kind=source_kind
    )

    # Symbolic (no LLM call): flag whether each row's value actually parses
    # as a number, so "find a parameter and note its numerical value" is
    # answerable directly from the output — some taxonomy parameters are
    # legitimately categorical (e.g. Photo-initiator Type), so non-numeric
    # rows are flagged, not dropped.
    try:
        from si_units import _extract_numeric
        for row in param_verified + param_flagged:
            row["value_is_numeric"] = _extract_numeric(row.get("value")) is not None
    except ImportError:
        pass

    for row in param_verified + param_flagged:
        row["skill_used"] = skill.code

    # Symbolic (no LLM call): tag every row with the exact taxonomy section
    # it came from — L4/L5/L8/L9/etc — by exact-matching skill.param_sections,
    # the same schema that was sent in the prompt. A row whose 'parameter'
    # string doesn't match anything in that schema gets flagged as
    # off-schema rather than silently passing through as if it were valid.
    param_to_section = _build_param_to_section_index(skill)
    _tag_taxonomy_section(param_verified, param_to_section)
    _tag_taxonomy_section(param_flagged, param_to_section)

    return {
        "skill": skill.code,
        "material": target_material,  # None for the unpinned/review fallback path
        "materials": {"verified": mat_verified, "flagged": mat_flagged},
        "parameters": {
            "verified": param_verified,
            "flagged": param_flagged,
            "missing": missing,
            "hollow_discarded": hollow,
        },
    }


# ---------- 5. Orchestrator loop: triage -> load relevant skills -> invoke each -> merge ----------
def run_orchestrator(triage_result, relevant_text_by_category, skill_library,
                      subtype_skill_library=None, model="gpt-5-mini"):
    """
    Primary path: iterates triage_result['printed_materials'] — each entry
    pins ONE build material to the ONE process that made it, e.g.
    {"material": "Clear IV resin", "process_category": "VPP",
     "process_subtype": "stereolithography"}. For each pair, resolves the
    matching skill and calls invoke_skill with that material FIXED, so every
    extracted parameter is unambiguously tied to a known material — no
    "material not confirmed" guesswork left for verify_parameters to do.
    Only taken for source_kind != "review" — see below.

    Fallback path: taken whenever triage found no printed_materials (e.g. a
    pure review paper with no build material of its own) OR the paper is a
    review, regardless of how many printed_materials entries triage
    returned. printed_materials is capped at 1-2 entries by design (see
    triage_pipeline.py's prompt) because that fits a primary paper's usual
    "one or two build materials" shape — but a review/survey paper routinely
    reports parameters for many materials against the same process (this
    paper: PLA, ABS, PP, PEEK all via MEX). Looping only over the (at most 2)
    pinned entries would silently drop every other material the review
    reports, AND would call the same REVIEW_CATEGORY_PROMPT redundantly once
    per pinned material — the review prompt already harvests every material
    for a category in ONE call, so per-material pinning has no purpose here.
    Falling back to the old unpinned, one-call-per-category loop is strictly
    better for reviews: single call per category, no material cap, and this
    is the same path already verified against review papers separately.

    relevant_text_by_category: dict {category_code: paper_text_to_use}.
    skill_library / subtype_skill_library: from build_skill_library() /
                   build_subtype_skill_library().
    model: the GPT-5-mini model string used for every skill invocation.
    """
    printed_materials = triage_result.get("printed_materials", [])
    source_kind = triage_result.get("_source_kind", "research-article")
    results = {}

    if printed_materials and source_kind != "review":
        # if two entries share a process_category, their result keys need to
        # be disambiguated by material; otherwise the plain skill code is fine
        codes_seen = [e.get("process_category") for e in printed_materials]
        needs_material_suffix = len(codes_seen) != len(set(codes_seen))

        for entry in printed_materials:
            material = entry.get("material")
            code = entry.get("process_category")
            subtype_text = entry.get("process_subtype", "")
            if not material or not code:
                print(f"Warning: skipping malformed printed_materials entry {entry}")
                continue

            text = relevant_text_by_category.get(code)
            if not text:
                print(f"Warning: no text provided for category {code} "
                      f"(material '{material}'), skipping")
                continue

            skill, used_level = None, "L1"
            if subtype_skill_library and subtype_text:
                skill = match_subtype(code, [subtype_text], subtype_skill_library)
                if skill:
                    used_level = "L2"
            if not skill:
                skill = skill_library.get(code)
            if not skill:
                print(f"Warning: no skill defined for category {code}, skipping")
                continue

            print(f"Orchestrator loading skill: {skill.code} ({skill.name}) [{used_level}] "
                  f"for material '{material}'...")
            key = f"{skill.code}::{material}" if needs_material_suffix else skill.code
            results[key] = invoke_skill(
                skill, text, model=model, source_kind=source_kind, target_material=material
            )
        json.dump(results, open("agentic_extraction_results.json", "w"), indent=2)
        return results

    # --- fallback: no printed_materials, OR a review paper regardless of
    # how many printed_materials entries triage returned — old behavior ---
    if source_kind == "review":
        print("Note: source is a review — using category-level extraction "
              "with no material pinned (printed_materials cap doesn't apply "
              "to reviews covering multiple materials).")
    else:
        print("Note: triage found no printed_materials — falling back to "
              "category-level extraction with no material pinned.")
    process_subtypes = triage_result.get("process_subtypes", [])
    if subtype_skill_library:
        categories = reconcile_categories(triage_result, subtype_skill_library)
    else:
        categories = triage_result.get("process_categories", [])

    for code in categories:
        text = relevant_text_by_category.get(code)
        if not text and relevant_text_by_category:
            print(f"Warning: no dedicated text for {code} — skipping")
            continue
        if not text:
            print(f"Warning: no text provided for category {code}, skipping")
            continue

        skill = None
        used_level = "L1"
        if subtype_skill_library:
            skill = match_subtype(code, process_subtypes, subtype_skill_library)
            if skill:
                used_level = "L2"
        if not skill:
            skill = skill_library.get(code)
        if not skill:
            print(f"Warning: no skill defined for category {code}, skipping")
            continue

        print(f"Orchestrator loading skill: {skill.code} ({skill.name}) [{used_level}]...")
        results[skill.code] = invoke_skill(skill, text, model=model, source_kind=source_kind)

    json.dump(results, open("agentic_extraction_results.json", "w"), indent=2)
    return results


# =====================================================================
# 6. LEAN HARNESS  (retrieve -> one schema-locked call per skill ->
#    code self-checks -> targeted repair)
#
# Same result shape as run_orchestrator, so downstream code is unaffected.
# The fixed path above stays available (mode="fixed") and is also the fallback
# if a lean call errors out.
#
#   chunk_retrieval   numbered chunks + BM25/unit search        (step 1)
#   _run_job          ONE call per skill: model only sees the retrieved chunks
#                     and can only emit real parameter names / real chunk ids /
#                     real materials (strict tool schema)        (step 2)
#   self_check_row    code checks each row; rejected rows go back to the
#                     model ONCE with the reason and only their own chunk (step 3)
#   run_lean          skills run in parallel, reasoning effort is configurable,
#                     tokens/time are recorded                   (step 4)
# =====================================================================
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import llm_pipeline
from chunk_retrieval import (build_chunks, pick_section_names, ChunkRetriever, normalize_spaces,
                             render_chunks, approx_tokens, split_param, unit_regex, tokenize)

# Extra retrieval vocabulary for CATEGORICAL parameters (no unit, so unit-matching
# can't help and the bare name alone finds the wrong sentence). Keys are lowercase
# bare taxonomy names; merged with each skill's own synonyms. Edit freely.
GLOBAL_SYNONYMS = {
    "machine make/model": ["printer", "printed with", "printed using", "system", "purchased from", "manufactured by"],
    "machine make / model": ["printer", "printed with", "printed using", "system", "purchased from"],
    "support type": ["support material", "support removal", "removed by"],
    "build orientation": ["orientation", "oriented"],
    "support removal": ["removed by", "water jetting", "soaking", "dissolved"],
}

REVIEW_PROVENANCE = ["general-author-statement", "cited-secondary", "primary-measurement"]


# ---------- 6a. Fixed schema: the tool the model must call ----------
def _unique_params(skill):
    seen, out = set(), []
    for params in skill.param_sections.values():
        for p in params:
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def build_extraction_tool(skill, chunk_ids, material_mode, materials, source_kind,
                          extra_skill_codes=(), name="submit_extraction"):
    """Strict function schema. What it makes impossible:
      - parameter names outside this skill's taxonomy   (enum)
      - citing a chunk that wasn't sent                 (enum)
      - a material other than triage's                  (enum, when >1 pinned)
    material_mode: 'single' (one pinned material: no field, harness sets it),
                   'enum'   (several pinned materials: must pick one),
                   'free'   (unpinned / review: free text, checked against the paper)."""
    props = {
        "parameter": {"type": "string", "enum": _unique_params(skill)},
        "value": {"type": "string",
                  "description": "Copied EXACTLY as written in the cited excerpt (ranges/lists as written)."},
        "unit": {"type": "string",
                 "description": "Unit as written next to the value, or empty string."},
        "chunk_id": {"type": "string", "enum": list(chunk_ids),
                     "description": "The single excerpt that contains the value."},
    }
    if material_mode == "enum":
        props["material"] = {"type": "string", "enum": list(materials)}
    elif material_mode == "free":
        props["material"] = {"type": "string",
                             "description": "Material this value belongs to, as named in the excerpts; empty string if unclear."}
    if source_kind == "review":
        props["provenance_type"] = {"type": "string", "enum": REVIEW_PROVENANCE}
        props["cited_work"] = {"type": "string", "description": "Author-year of the cited work, else empty."}
        props["citation_marker"] = {"type": "string", "description": "Reference marker such as [62], else empty."}

    row = {"type": "object", "properties": props, "required": list(props),
           "additionalProperties": False}
    schema = {"type": "object",
              "properties": {"rows": {"type": "array", "items": row}},
              "required": ["rows"], "additionalProperties": False}
    if extra_skill_codes:
        schema["properties"]["additional_skills"] = {
            "type": "array", "maxItems": 1,
            "items": {"type": "string", "enum": list(extra_skill_codes)},
            "description": "Skill codes to load next. Almost always empty.",
        }
        schema["required"].append("additional_skills")
    return {"name": name,
            "description": "Submit the parameter values found in the excerpts (empty list if none).",
            "schema": schema}


# ---------- 6b. Prompts (short: the paper text is only the retrieved chunks) ----------
LEAN_PROMPT = """You are extracting additive-manufacturing (AM) process data from numbered excerpts of ONE paper.
Loaded skill: {skill_name} ({skill_code}). Only this skill's parameters exist for this pass
(the "parameter" field is restricted to them). Call submit_extraction with one row per value.

{material_block}
Rules:
- "value" must be copied EXACTLY as written inside the cited excerpt (same digits and characters). Ranges/lists: copy as written. Never convert units or calculate.
- "chunk_id" is the single excerpt that contains the value. One row per (parameter, value, material); if a parameter has different values for different samples/conditions, emit one row for each.
- "unit" is the unit as written next to the value, or "" if none.
- Only extract what the authors themselves used, printed, processed or measured. Do NOT take numbers from unrelated protocols (chemical extraction, chromatography, simulation settings) and label them as print parameters.
- If a parameter is mentioned but no value is given, omit it. Never invent values.
- An empty "rows" list is the CORRECT answer when the excerpts report nothing from this skill.
{review_block}{skill_request_block}
{synonym_text}

{hints_text}

EXCERPTS:
{excerpts}
"""

REVIEW_BLOCK = """- This paper is a REVIEW. Also extract values the authors state as general facts and values attributed to cited work
  (set provenance_type; for cited-secondary give cited_work and citation_marker when visible in the excerpt, else empty strings).
  Table-row excerpts count: the column header names the parameter.
"""

SKILL_REQUEST_BLOCK = """- additional_skills: list a code from {codes} ONLY if the excerpts show the AUTHORS THEMSELVES used that other AM process
  (not merely mention it). Almost always leave it empty.
"""

REPAIR_PROMPT = """An automatic checker rejected some rows you extracted from numbered excerpts of a paper.
For each rejected row, either return a CORRECTED row via submit_corrections (fix value / unit / parameter / chunk_id)
or leave it out if the excerpt does not actually support it. Do not add unrelated new rows.
"value" must be copied EXACTLY as written in the cited excerpt. Never invent values.

REJECTED ROWS:
{problems}

EXCERPTS:
{excerpts}
"""


def _material_block(material_mode, materials):
    if material_mode == "single":
        return (f'The build material is ALREADY IDENTIFIED: everything you extract is about "{materials[0]}". '
                f'Skip values that clearly belong to another material (reagents, solvents, a comparison material).\n')
    if material_mode == "enum":
        return ("The paper prints these materials: " + "; ".join(f'"{m}"' for m in materials) +
                '. Set "material" to the one each value belongs to; if a value applies to several, emit one row per material.\n')
    return ('Material is not pinned: set "material" to the material each value belongs to, as named in the excerpts '
            '(empty string if unclear).\n')


def _synonym_text(skill):
    lines = []
    for canonical, alts in (skill.synonyms or {}).items():
        lines.append(f"  - {canonical} <- {', '.join(alts)}")
    return ("Known alternate phrasings (map to the taxonomy name):\n" + "\n".join(lines)) if lines else ""


# ---------- 6c. Self-check: code, not LLM ----------
_NUM_RE = re.compile(r"(?<![\w.])[-+\u2212]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def parse_numbers(text):
    """All numbers in a value string ('150-170', '1,100', '\u22125.2', '30 x 30')."""
    out = []
    for m in _NUM_RE.findall(str(text)):
        s = m.replace("\u2212", "-")
        s = s.replace(",", "") if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s) else s.replace(",", ".")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


_PARAM_BOUNDS = {"Relative Density (%)": (0, 100.5), "Porosity (%)": (0, 100)}
_UNIT_BOUNDS = {"\u00b0C": (-273.15, 6000.0), "K": (0.0, 10000.0), "vol%": (0, 100), "wt%": (0, 100)}
_NONNEG_UNITS = {"W", "mW", "kW", "mm/s", "\u00b5m/s", "mm", "\u00b5m", "nm", "s", "min", "hr", "MPa", "GPa",
                 "kPa", "kV", "mA", "A", "V", "Hz", "kHz", "mPa\u00b7s", "g/cm\u00b3", "mm\u00b3", "pL", "N"}


def range_problem(param, unit, nums):
    bounds = _PARAM_BOUNDS.get(param) or _UNIT_BOUNDS.get(unit)
    if bounds is None and unit in _NONNEG_UNITS:
        bounds = (0.0, 1e9)
    if bounds is None:
        return None
    lo, hi = bounds
    bad = [n for n in nums if n < lo or n > hi]
    if bad:
        return f"value {bad[0]:g} is outside the plausible range [{lo:g}, {hi:g}] for {param}"
    return None


def _norm_unit(u):
    return re.sub(r"[\s^]", "", u.lower().replace("\u00b5", "u").replace("\u03bc", "u"))


def self_check_row(row, chunk, name_tokens):
    """None if the row passes, else a short reason (this text is sent back to the model).
      1. value appears verbatim in the cited chunk        (same test verify_parameters uses)
      2. numeric values sit in a physically plausible range for the unit
      3. parameters with a real unit are supported by the chunk: it either has a
         number in that unit, or contains a word from the parameter name/synonyms
    A non-numeric value for a unit parameter is NOT rejected (qualitative statements
    are allowed, as in the fixed pipeline) - it is just marked value_is_numeric=False."""
    value = row.get("value", "")
    if chunk is None:
        return f"unknown chunk id {row.get('chunk_id')!r}"
    if value.lower() not in chunk.quote.lower():
        return f"value '{value}' does not appear verbatim in chunk {chunk.id}"
    bare, unit = split_param(row["parameter"])
    nums = parse_numbers(value)
    if nums:
        prob = range_problem(row["parameter"], unit, nums)
        if prob:
            return prob
    urx = unit_regex(unit)
    if urx is not None:
        toks = name_tokens.get(row["parameter"], set())
        if not urx.search(chunk.text) and not (toks & set(tokenize(chunk.text))):
            return (f"chunk {chunk.id} has neither a number in '{unit}' nor any word of "
                    f"'{bare}', so it does not support this parameter")
    return None


# ---------- 6d. Row materialisation ----------
def _materialize(raw_rows, chunk_by_id, fixed_material, source_kind):
    rows = []
    for r in raw_rows:
        c = chunk_by_id.get(r.get("chunk_id"))
        row = {
            "parameter": str(r.get("parameter") or ""),
            "value": str(r.get("value") or "").strip(),
            "unit": str(r.get("unit") or "").strip(),
            "material": fixed_material if fixed_material is not None else str(r.get("material") or "").strip(),
            "chunk_id": r.get("chunk_id"),
            "section": c.section if c else "",
            "quote": c.quote if c else "",       # copied by code from the chunk, never typed by the model
        }
        if c is not None and c.kind == "table":
            row["context"] = c.text                # caption + column headers + row
        if source_kind == "review":
            row["provenance"] = {"type": r.get("provenance_type") or "general-author-statement",
                                 "cited_work": r.get("cited_work") or "",
                                 "citation_marker": r.get("citation_marker") or ""}
        if c is not None and row["unit"]:
            row["unit_in_chunk"] = _norm_unit(row["unit"]) in _norm_unit(c.text)
        rows.append(row)
    return rows


def _dedupe_rows(rows):
    seen, out = set(), []
    for r in rows:
        k = (r["parameter"], r["value"].lower(), r["material"].lower(), r["chunk_id"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


# ---------- 6e. One skill job ----------
def _call_model(ctx, prompt, tool, usage):
    """One forced tool call, one retry on malformed/missing output (same policy as triage)."""
    last = None
    for attempt in range(2):
        try:
            args, u = ctx.caller(prompt, tool["name"], tool["description"], tool["schema"],
                                 model=ctx.model, reasoning_effort=ctx.effort)
            for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "seconds"):
                usage[k] = usage.get(k, 0) + (u or {}).get(k, 0)
            usage["calls"] = usage.get("calls", 0) + 1
            return args
        except (ValueError, json.JSONDecodeError) as e:
            last = e
            print(f"Warning: tool call attempt {attempt + 1} failed ({e}).")
    raise RuntimeError(f"model call failed twice: {last}")


def _run_job_lean(job, ctx):
    skill, materials = job["skill"], job["materials"]
    usage = {}
    selected, _ = ctx.retriever.retrieve_for_skill(skill, ctx.per_param_k, ctx.max_chunks)
    stats = {"skill": skill.code, "chunks_sent": len(selected),
             "chunk_tokens_est": sum(approx_tokens(c.text) for c in selected)}
    if not selected:
        stats.update(rows_raw=0, note="no chunks retrieved")
        return {"rows": [], "rejected": [], "dropped": [], "hollow": [], "missing": [],
                "requested": [], "stats": stats, "usage": usage}

    chunk_by_id = {c.id: c for c in selected}
    if materials is None:
        mode, fixed = "free", None
    elif len(materials) == 1:
        mode, fixed = "single", materials[0]
    else:
        mode, fixed = "enum", None

    syn = dict(GLOBAL_SYNONYMS)
    for k, v in (skill.synonyms or {}).items():
        syn[k.lower()] = list(syn.get(k.lower(), [])) + list(v)
    name_tokens = {}
    for p in _unique_params(skill):
        bare, _ = split_param(p)
        toks = set(tokenize(bare))
        for s in syn.get(bare.lower(), []):
            toks |= set(tokenize(s))
        name_tokens[p] = toks

    # the retriever should see the same extended synonyms as the self-check
    skill_for_retrieval = SimpleNamespace(param_sections=skill.param_sections, synonyms=syn)
    if syn != (skill.synonyms or {}):
        selected, _ = ctx.retriever.retrieve_for_skill(skill_for_retrieval, ctx.per_param_k, ctx.max_chunks)
        chunk_by_id = {c.id: c for c in selected}
        stats["chunks_sent"] = len(selected)
        stats["chunk_tokens_est"] = sum(approx_tokens(c.text) for c in selected)

    tool = build_extraction_tool(skill, [c.id for c in selected], mode, materials, ctx.source_kind,
                                 extra_skill_codes=job.get("extra_codes", []))
    codes = ", ".join(job.get("extra_codes", []))
    prompt = LEAN_PROMPT.format(
        skill_name=skill.name, skill_code=skill.code,
        material_block=_material_block(mode, materials),
        review_block=REVIEW_BLOCK if ctx.source_kind == "review" else "",
        skill_request_block=SKILL_REQUEST_BLOCK.format(codes=codes) if codes else "",
        synonym_text=_synonym_text(skill), hints_text=skill.build_hints_text(),
        excerpts=render_chunks(selected),
    )
    args = _call_model(ctx, prompt, tool, usage)
    raw_rows = args.get("rows", []) or []
    requested = list(args.get("additional_skills", []) or [])

    rows = _dedupe_rows(_materialize(raw_rows, chunk_by_id, fixed, ctx.source_kind))
    hollow = [r for r in rows if _is_hollow_row(r, ctx.source_kind)]
    missing = [r for r in rows if _is_explicitly_missing(r)]
    rows = [r for r in rows if r not in hollow and r not in missing]

    def check_all(rs):
        ok, bad = [], []
        for r in rs:
            reason = self_check_row(r, chunk_by_id.get(r["chunk_id"]), name_tokens)
            if reason:
                r["flag_reason"] = f"self-check: {reason}"
                bad.append(r)
            else:
                ok.append(r)
        return ok, bad

    accepted, rejected = check_all(rows)
    stats.update(rows_raw=len(raw_rows), rejected_first_pass=len(rejected))

    dropped = []          # rejected rows the model chose to drop on repair (i.e. unsupported)
    repaired = 0
    for _ in range(ctx.retry_rounds):
        if not rejected:
            break
        rej_ids = sorted({r["chunk_id"] for r in rejected if r["chunk_id"] in chunk_by_id})
        if not rej_ids:
            break
        problems = "\n".join(
            f'- parameter="{r["parameter"]}" value="{r["value"]}" chunk={r["chunk_id"]} -> problem: '
            f'{r["flag_reason"].replace("self-check: ", "")}' for r in rejected)
        rtool = build_extraction_tool(skill, rej_ids, mode, materials, ctx.source_kind,
                                      name="submit_corrections")
        rprompt = REPAIR_PROMPT.format(problems=problems,
                                       excerpts=render_chunks([chunk_by_id[i] for i in rej_ids]))
        try:
            rargs = _call_model(ctx, rprompt, rtool, usage)
        except RuntimeError as e:
            print(f"Warning: repair call failed for {skill.code}: {e}")
            break
        new_rows = _dedupe_rows(_materialize(rargs.get("rows", []) or [], chunk_by_id, fixed, ctx.source_kind))
        ok, bad = check_all(new_rows)
        repaired += len(ok)
        touched = {(r["parameter"], r["chunk_id"]) for r in new_rows}
        dropped += [r for r in rejected if (r["parameter"], r["chunk_id"]) not in touched]
        for r in ok:
            r["repaired"] = True
        accepted += ok
        rejected = bad
    stats.update(repaired_ok=repaired, dropped_on_repair=len(dropped), still_flagged=len(rejected))
    for r in rejected:
        r["retried"] = ctx.retry_rounds > 0

    return {"rows": accepted, "rejected": rejected, "dropped": dropped, "hollow": hollow,
            "missing": missing, "requested": requested, "stats": stats, "usage": usage}


def _finalize(skill, out, materials, ctx):
    """verify_parameters (existing rules) + taxonomy tagging + result dicts in the run_orchestrator shape."""
    accepted = out["rows"]
    pinned = materials is not None
    if pinned:
        known = set(materials)
    else:
        low = ctx.paper_text.lower()
        known = {r["material"] for r in accepted if r["material"] and r["material"].lower() in low}
    verified, flagged = verify_parameters(accepted, ctx.paper_text, known, source_kind=ctx.source_kind)
    flagged = flagged + out["rejected"]

    try:
        from si_units import _extract_numeric
        is_num = lambda v: _extract_numeric(v) is not None
    except ImportError:
        is_num = lambda v: bool(parse_numbers(v))
    p2s = _build_param_to_section_index(skill)
    for row in verified + flagged:
        row["value_is_numeric"] = is_num(row.get("value"))
        row["skill_used"] = skill.code
    _tag_taxonomy_section(verified, p2s)
    _tag_taxonomy_section(flagged, p2s)

    def mats(m_list):
        v = [{"material": m, "material_form": "", "quote": "", "verified": True,
              "note": "pinned by triage's printed_materials"} for m in m_list]
        return v, []

    results = OrderedDict()
    if pinned:
        for m in materials:
            mv, mf = mats([m])
            results[m] = {
                "skill": skill.code, "material": m,
                "materials": {"verified": mv, "flagged": mf},
                "parameters": {
                    "verified": [r for r in verified if r["material"] == m],
                    "flagged": [r for r in flagged if r["material"] == m],
                    "missing": [r for r in out["missing"] if r["material"] == m],
                    "hollow_discarded": [r for r in out["hollow"] if r["material"] == m],
                    "self_check_dropped": [r for r in out["dropped"] if r["material"] == m],
                },
            }
    else:
        seen = {r["material"] for r in verified + flagged if r["material"]}
        mv = [{"material": m, "material_form": "", "quote": "", "verified": True,
               "note": "named by the model and found in the paper text"} for m in sorted(seen & known)]
        mf = [{"material": m, "material_form": "", "quote": "", "verified": False,
               "note": "named by the model but not found in the paper text"} for m in sorted(seen - known)]
        results[None] = {
            "skill": skill.code, "material": None,
            "materials": {"verified": mv, "flagged": mf},
            "parameters": {"verified": verified, "flagged": flagged, "missing": out["missing"],
                           "hollow_discarded": out["hollow"], "self_check_dropped": out["dropped"]},
        }
    return results


def _fallback_fixed(job, ctx, err):
    """Lean call failed -> use the original fixed path for this skill."""
    print(f"Warning: lean path failed for {job['skill'].code} ({err}); falling back to fixed invoke_skill.")
    res = OrderedDict()
    for m in (job["materials"] or [None]):
        r = invoke_skill(job["skill"], ctx.paper_text, source_kind=ctx.source_kind,
                         model=ctx.model, target_material=m)
        r["lean"] = {"fallback": str(err)}
        res[m] = r
    return res, {"skill": job["skill"].code, "fallback": str(err)}, {}, []


def _run_job(job, ctx):
    try:
        out = _run_job_lean(job, ctx)
        res = _finalize(job["skill"], out, job["materials"], ctx)
        for r in res.values():
            r["lean"] = dict(out["stats"], usage=out["usage"])
        return res, out["stats"], out["usage"], out["requested"]
    except Exception as e:  # noqa: BLE001 - any failure -> fixed path
        return _fallback_fixed(job, ctx, e)


# ---------- 6f. Orchestration ----------
def _resolve_skill(code, subtype_texts, skill_library, subtype_skill_library):
    skill = None
    texts = [t for t in (subtype_texts or []) if t and t.strip()]   # "" would match every L2 name
    if subtype_skill_library and texts:
        skill = match_subtype(code, texts, subtype_skill_library)
    return skill or skill_library.get(code)


def run_lean(triage_result, sectioned, skill_library, subtype_skill_library=None,
             model="gpt-5-mini", reasoning_effort="low", per_param_k=3, max_chunks=40,
             retry_rounds=1, allow_skill_requests=True, max_extra_skills=1, max_workers=4,
             caller=None, out_path="agentic_lean_results.json",
             stats_path="agentic_lean_stats.json"):
    """Lean equivalent of run_orchestrator. Returns (results, stats).

    results has the same shape as run_orchestrator's (plus a 'lean' stats block and
    'self_check_dropped' per skill) and is written to `out_path` - a DIFFERENT file
    than the fixed pipeline's agentic_extraction_results.json, so the fixed results
    stay available as the comparison baseline / pseudo-gold.
    caller: LLM function with the signature of llm_pipeline.call_gpt_mini_tool_ex
    (inject a fake one to test without an API key)."""
    t0 = time.time()
    source_kind = triage_result.get("_source_kind", "research-article")
    names = pick_section_names(triage_result, sectioned)
    chunks = build_chunks(sectioned, names)
    ctx = SimpleNamespace(
        retriever=ChunkRetriever(chunks), source_kind=source_kind, model=model,
        effort=reasoning_effort, per_param_k=per_param_k, max_chunks=max_chunks,
        retry_rounds=retry_rounds, caller=caller or llm_pipeline.call_gpt_mini_tool_ex,
        paper_text=normalize_spaces("\n\n".join(sectioned[n] for n in names)),
    )
    print(f"Lean: {len(chunks)} chunks (~{sum(approx_tokens(c.text) for c in chunks)} tokens) "
          f"from {len(names)} sections | source kind: {source_kind}")

    # ---- build jobs (same resolution rules as run_orchestrator) ----
    printed = triage_result.get("printed_materials", [])
    groups = OrderedDict()                       # skill.code -> job
    if printed and source_kind != "review":
        for e in printed:
            material, code = e.get("material"), e.get("process_category")
            if not material or not code:
                print(f"Warning: skipping malformed printed_materials entry {e}")
                continue
            skill = _resolve_skill(code, [e.get("process_subtype", "")], skill_library, subtype_skill_library)
            if not skill:
                print(f"Warning: no skill defined for category {code}, skipping")
                continue
            job = groups.setdefault(skill.code, {"skill": skill, "materials": []})
            if material not in job["materials"]:
                job["materials"].append(material)
    else:
        if source_kind == "review":
            print("Note: review paper -> category-level extraction, materials not pinned.")
        else:
            print("Note: triage found no printed_materials -> category-level extraction, materials not pinned.")
        subtypes = triage_result.get("process_subtypes", [])
        cats = (reconcile_categories(triage_result, subtype_skill_library)
                if subtype_skill_library else triage_result.get("process_categories", []))
        for code in cats:
            skill = _resolve_skill(code, subtypes, skill_library, subtype_skill_library)
            if skill:
                groups.setdefault(skill.code, {"skill": skill, "materials": None})
            else:
                print(f"Warning: no skill defined for category {code}, skipping")
    jobs = list(groups.values())

    l1_run = {j["skill"].code.split("-")[0] for j in jobs}
    catalog = [c for c in skill_library if c != "OTHER" and c not in l1_run]
    for j in jobs:
        j["extra_codes"] = catalog if (allow_skill_requests and catalog) else []

    def run_many(js):
        if not js:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(js)))) as ex:
            return list(ex.map(lambda j: _run_job(j, ctx), js))

    outputs = run_many(jobs)

    # ---- skill calling: the model asked for another skill (depth 1, capped) ----
    extra_jobs = []
    if allow_skill_requests:
        asked = []
        for _, _, _, req in outputs:
            for code in req:
                if code in skill_library and code not in l1_run and code not in asked:
                    asked.append(code)
        for code in asked[:max_extra_skills]:
            print(f"Skill request: model asked to load {code}; running it once (unpinned materials).")
            extra_jobs.append({"skill": skill_library[code], "materials": None, "extra_codes": []})
    outputs += run_many(extra_jobs)
    all_jobs = jobs + extra_jobs

    # ---- merge in the same key style as run_orchestrator ----
    results = OrderedDict()
    for job, (res, _, _, _) in zip(all_jobs, outputs):
        code = job["skill"].code
        multi = job["materials"] is not None and len(job["materials"]) > 1
        for m, r in res.items():
            results[f"{code}::{m}" if multi else code] = r

    usage_total = {}
    for _, _, u, _ in outputs:
        for k, v in u.items():
            usage_total[k] = usage_total.get(k, 0) + v
    stats = {"wall_seconds": round(time.time() - t0, 2), "usage": usage_total,
             "jobs": [s for _, s, _, _ in outputs], "reasoning_effort": reasoning_effort,
             "chunks_total": len(chunks)}
    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
    json.dump(stats, open(stats_path, "w"), indent=2)
    return results, stats


# ---------- 6g. Side-by-side comparison with the fixed pipeline ----------
def _canon_value(v):
    return re.sub(r"\s+", "", str(v or "").lower())


def _verified_keys(results):
    keys = {}
    for r in results.values():
        for row in r.get("parameters", {}).get("verified", []):
            keys[(row.get("parameter"), _canon_value(row.get("value")))] = row
    return keys


def compare_results(fixed, lean, fixed_cost, lean_cost):
    def count(res, k):
        return sum(len(r.get("parameters", {}).get(k, [])) for r in res.values())
    fk, lk = _verified_keys(fixed), _verified_keys(lean)
    print("\n" + "=" * 60)
    print(f"{'':28s}{'fixed':>14s}{'lean':>14s}")
    for label, key in (("LLM calls", "calls"), ("input tokens", "input_tokens"),
                       ("output tokens", "output_tokens"),
                       ("  of which reasoning", "reasoning_tokens"), ("wall seconds", "seconds")):
        print(f"{label:28s}{fixed_cost.get(key, 0):>14.1f}{lean_cost.get(key, 0):>14.1f}")
    print(f"{'verified rows':28s}{count(fixed, 'verified'):>14d}{count(lean, 'verified'):>14d}")
    print(f"{'flagged rows':28s}{count(fixed, 'flagged'):>14d}{count(lean, 'flagged'):>14d}")
    both = set(fk) & set(lk)
    print(f"\nverified in BOTH (same parameter + value): {len(both)}")
    print(f"only in fixed: {len(set(fk) - set(lk))} | only in lean: {len(set(lk) - set(fk))}")
    for label, a, b in (("ONLY IN FIXED", fk, lk), ("ONLY IN LEAN", lk, fk)):
        extra = [k for k in a if k not in b]
        if extra:
            print(f"\n{label} (first 12):")
            for k in extra[:12]:
                print(f"  {k[0]} = {a[k].get('value')}  | \"{(a[k].get('quote') or '')[:90]}\"")
    print("=" * 60)
    return {"both": len(both), "only_fixed": len(set(fk) - set(lk)), "only_lean": len(set(lk) - set(fk))}


def _print_results(results):
    for code, r in results.items():
        n_params = len(r["parameters"]["verified"])
        n_flagged = len(r["parameters"]["flagged"])
        mat = r.get("material")
        mat_label = f" [material: {mat}]" if mat else ""
        print(f"{code}{mat_label}: {n_params} verified, {n_flagged} flagged")


if __name__ == "__main__":
    import argparse
    import sys
    from triage_pipeline import run_triage, get_relevant_text, strip_markup_sectioned

    ap = argparse.ArgumentParser(description="AM parameter extraction: fixed, lean, or side-by-side compare.")
    ap.add_argument("paper", help="path to paper.xml")
    ap.add_argument("vocab", nargs="?", default="vocab_tree.json")
    ap.add_argument("--mode", choices=["fixed", "lean", "compare"], default="fixed",
                    help="fixed = original per-skill full-text calls; lean = chunk retrieval + schema-locked "
                         "calls + self-check; compare = run both and print cost/coverage side by side")
    ap.add_argument("--effort", default="low", choices=["minimal", "low", "medium", "high"],
                    help="reasoning effort for the LEAN path (fixed keeps the model default)")
    ap.add_argument("--per-param-k", type=int, default=3)
    ap.add_argument("--max-chunks", type=int, default=40)
    ap.add_argument("--retry-rounds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-skill-requests", action="store_true")
    ap.add_argument("--triage-json", help="reuse a saved triage_result.json instead of calling the LLM again")
    args = ap.parse_args()

    with open(args.paper, encoding="utf-8") as f:
        raw_markup = f.read()

    if args.triage_json:
        print(f"--- Reusing triage from {args.triage_json} ---")
        triage = json.load(open(args.triage_json))
        sectioned = strip_markup_sectioned(raw_markup)
    else:
        print(f"--- Running triage on {args.paper} ---")
        triage, sectioned = run_triage(raw_markup, )
    relevant_text = get_relevant_text(triage, sectioned)

    skill_library = build_skill_library(args.vocab)
    subtype_skill_library = build_subtype_skill_library(args.vocab)

    # same relevant text handed to every identified category's skill, for a simple start
    full_text = "\n\n".join(sectioned.values())
    codes_needed = set(triage.get("process_categories", [])) | {
        e.get("process_category") for e in triage.get("printed_materials", []) if e.get("process_category")
    }
    per_category_text = {code: (relevant_text if relevant_text else full_text) for code in codes_needed}

    # Symbolic (non-AI) material classification — no extra LLM call, just a
    # lookup/regex pass over triage's materials_mentioned/process_subtypes
    # (L8: material composition) and the paper text (L9: post-processing
    # steps, since those are described in prose, not listed as materials).
    # L8 + L9 come back as ONE merged list — the taxonomy itself renders them
    # as a single shared panel, not two levels to track separately.
    try:
        from material_classifier import classify_material
        material_info = classify_material(triage, scan_text=full_text)
        print("\n--- Material classification (symbolic, no LLM) ---")
        print(json.dumps(material_info, indent=2, ensure_ascii=False))
    except ImportError:
        material_info = None  # material_classifier.py not present — skip

    def run_fixed():
        return run_orchestrator(triage, per_category_text, skill_library, subtype_skill_library, model="gpt-5-mini")

    def run_lean_mode():
        return run_lean(triage, sectioned, skill_library, subtype_skill_library, model="gpt-5-mini",
                        reasoning_effort=args.effort, per_param_k=args.per_param_k,
                        max_chunks=args.max_chunks, retry_rounds=args.retry_rounds,
                        allow_skill_requests=not args.no_skill_requests, max_workers=args.workers)

    if args.mode == "fixed":
        print("\n--- Running category skills (fixed) ---")
        _print_results(run_fixed())

    elif args.mode == "lean":
        print("\n--- Running category skills (lean) ---")
        results, stats = run_lean_mode()
        _print_results(results)
        u = stats["usage"]
        print(f"\nLean cost: {u.get('calls', 0)} calls | {u.get('input_tokens', 0)} in / "
              f"{u.get('output_tokens', 0)} out tokens ({u.get('reasoning_tokens', 0)} reasoning) | "
              f"{stats['wall_seconds']}s wall")

    else:  # compare
        llm_pipeline.USAGE_LOG.clear()
        t = time.time()
        print("\n--- [1/2] fixed pipeline ---")
        fixed = run_fixed()
        fixed_cost = llm_pipeline.summarize_usage(llm_pipeline.USAGE_LOG)
        fixed_cost["seconds"] = round(time.time() - t, 1)      # wall clock, like lean's
        _print_results(fixed)

        llm_pipeline.USAGE_LOG.clear()
        print("\n--- [2/2] lean pipeline ---")
        lean, stats = run_lean_mode()
        _print_results(lean)
        lean_cost = dict(stats["usage"])
        lean_cost["seconds"] = stats["wall_seconds"]
        summary = compare_results(fixed, lean, fixed_cost, lean_cost)
        json.dump({"fixed_cost": fixed_cost, "lean_cost": lean_cost, "overlap": summary},
                  open("compare_report.json", "w"), indent=2)
