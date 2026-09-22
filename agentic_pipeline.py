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

    Fallback path: if triage found no printed_materials (e.g. a pure review
    paper with no build material of its own), falls back to the old
    category-only loop with no material pinned.

    relevant_text_by_category: dict {category_code: paper_text_to_use}.
    skill_library / subtype_skill_library: from build_skill_library() /
                   build_subtype_skill_library().
    model: the GPT-5-mini model string used for every skill invocation.
    """
    printed_materials = triage_result.get("printed_materials", [])
    source_kind = triage_result.get("_source_kind", "research-article")
    results = {}

    if printed_materials:
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

    # --- fallback: no printed_materials (e.g. review paper) — old behavior ---
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


if __name__ == "__main__":
    import sys
    from triage_pipeline import run_triage, get_relevant_text

    if len(sys.argv) < 2:
        print("Usage: python3 agentic_pipeline.py path/to/paper.xml [vocab_tree.json]")
        sys.exit(1)

    paper_path = sys.argv[1]
    vocab_path = sys.argv[2] if len(sys.argv) > 2 else "vocab_tree.json"

    with open(paper_path, encoding="utf-8") as f:
        raw_markup = f.read()

    print(f"--- Running triage on {paper_path} ---")
    triage, sectioned = run_triage(raw_markup)
    relevant_text = get_relevant_text(triage, sectioned)

    skill_library = build_skill_library(vocab_path)
    subtype_skill_library = build_subtype_skill_library(vocab_path)

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

    print("\n--- Running category skills ---")
    results = run_orchestrator(triage, per_category_text, skill_library, subtype_skill_library, model="gpt-5-mini")
    for code, r in results.items():
        n_params = len(r["parameters"]["verified"])
        n_flagged = len(r["parameters"]["flagged"])
        mat = r.get("material")
        mat_label = f" [material: {mat}]" if mat else ""
        print(f"{code}{mat_label}: {n_params} verified, {n_flagged} flagged")
