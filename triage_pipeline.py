"""
Intro / triage pass.

Goal: cheaply scan a paper's raw XML/HTML and answer a few basic questions
BEFORE running the full taxonomy-driven extraction. This lets you narrow
which part of the taxonomy to send in the detailed pass (see llm_pipeline.py),
instead of always sending all ~373 parameters across all 8 AM categories.

Triage answers exactly three questions: which material is being processed,
by which process, and which sections are worth reading for parameter values.
Material/process come from abstract+intro+Materials/Methods; section relevance comes from a
compact listing of every section (see run_triage).
"""
import json
import re
from html.parser import HTMLParser


# ---------- 1. Strip XML/HTML into section-labeled text blocks ----------
HEADING_TAGS = {"h1", "h2", "h3", "h4", "sec-title", "section-title"}
# Note: bare "title" is deliberately excluded — it matches document-level metadata
# tags like <dc:title> (the paper's own title, not a section header), and treating
# it as a heading causes front-matter (journal name, DOI, author list) to get
# misbucketed as if it were that "section's" content. Real section headers use
# "section-title" (e.g. Elsevier's <ce:section-title>), which is still included.

def _local_tag(tag):
    """Strips an XML namespace prefix (e.g. 'ce:section-title' -> 'section-title')
    so heading detection works on namespaced publisher XML (Elsevier, Springer, etc.)
    as well as plain HTML."""
    return tag.split(":", 1)[-1].lower()

class TablePreservingExtractor(HTMLParser):
    """
    Walks markup, bucketing text under the most recent heading (same as before),
    but additionally serializes each <table> (or <ce:table>) as a structured
    text block:

        [TABLE: <caption if any>]
        Columns: A | B | C
        Row: 1 | 2 | 3
        Row: 4 | 5 | 6

    This block is appended to whichever section it appeared in, so downstream
    extraction can see column headers alongside cell values. This is what makes
    table-based extraction (reviews, capabilities tables) actually workable.
    """
    def __init__(self):
        super().__init__()
        self.sections = {}
        self.order = []
        self.current_section = "body"
        self.sections[self.current_section] = []
        self.order.append(self.current_section)
        self._in_heading = False
        self._heading_buf = []
        self._in_table = False
        self._in_cell = False
        self._cell_buf = []
        self._row = []
        self._rows = []
        self._caption = ""
        self._in_caption = False
        self._caption_buf = []

    def handle_starttag(self, tag, attrs):
        t = _local_tag(tag)
        if t in HEADING_TAGS:
            self._in_heading = True
            self._heading_buf = []
        elif t == "table":
            self._in_table = True
            self._rows = []
            self._caption = ""
        elif t == "caption":
            self._in_caption = True
            self._caption_buf = []
        elif t in ("tr",):
            self._row = []
        elif t in ("td", "th", "entry"):
            self._in_cell = True
            self._cell_buf = []

    def handle_endtag(self, tag):
        t = _local_tag(tag)
        if t in HEADING_TAGS and self._in_heading:
            heading = " ".join(self._heading_buf).strip()
            self._in_heading = False
            if heading:
                self.current_section = heading
                if heading not in self.sections:
                    self.sections[heading] = []
                    self.order.append(heading)
        elif t == "caption" and self._in_caption:
            self._caption = " ".join(self._caption_buf).strip()
            self._in_caption = False
        elif t in ("td", "th", "entry") and self._in_cell:
            self._row.append(" ".join(self._cell_buf).strip())
            self._in_cell = False
        elif t == "tr":
            if self._row:
                self._rows.append(self._row)
            self._row = []
        elif t == "table":
            self._in_table = False
            self._emit_table()

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._in_heading:
            self._heading_buf.append(text)
        elif self._in_cell:
            self._cell_buf.append(text)
        elif self._in_caption:
            self._caption_buf.append(text)
        else:
            self.sections[self.current_section].append(text)

    def _emit_table(self):
        if not self._rows:
            return
        lines = []
        if self._caption:
            lines.append(f"[TABLE: {self._caption}]")
        else:
            lines.append("[TABLE]")
        # First row = column headers (best-effort)
        header = self._rows[0]
        lines.append("Columns: " + " | ".join(header))
        for r in self._rows[1:]:
            lines.append("Row: " + " | ".join(r))
        self.sections[self.current_section].append("\n".join(lines))

class SectionAwareExtractor(HTMLParser):
    """
    Walks the markup and buckets text under the most recent heading it saw.
    Falls back to a single "body" bucket if no headings are found at all
    (e.g. plain XML with no explicit heading tags).
    """
    def __init__(self):
        super().__init__()
        self.sections = {}          # section_name -> list of text chunks
        self.order = []             # preserves section order
        self.current_section = "body"
        self._in_heading = False
        self._heading_buf = []
        self.sections[self.current_section] = []
        self.order.append(self.current_section)

    def handle_starttag(self, tag, attrs):
        if _local_tag(tag) in HEADING_TAGS:
            self._in_heading = True
            self._heading_buf = []

    def handle_endtag(self, tag):
        if _local_tag(tag) in HEADING_TAGS and self._in_heading:
            heading = " ".join(self._heading_buf).strip()
            self._in_heading = False
            if heading:
                self.current_section = heading
                if heading not in self.sections:
                    self.sections[heading] = []
                    self.order.append(heading)

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._in_heading:
            self._heading_buf.append(text)
        else:
            self.sections[self.current_section].append(text)

def strip_markup_sectioned(raw_markup):
    """Returns dict: {section_name: full_text}, in document order."""
    parser = TablePreservingExtractor()
    parser.feed(raw_markup)
    result = {}
    for name in parser.order:
        joined = "\n".join(parser.sections[name]).strip()
        if joined:
            result[name] = joined
    return result

def strip_markup(raw_markup):
    """Flat version (no section boundaries) — kept for backward compatibility."""
    sections = strip_markup_sectioned(raw_markup)
    return "\n\n".join(sections.values())

def detect_source_kind(raw_markup):
    """Returns 'review' or 'research-article'. Used to pick extraction mode
    downstream — reviews need a looser prompt that accepts general statements
    and table cells, not just author-voice primary measurements."""
    if re.search(r'<xocs:document-subtype>\s*rev\s*</xocs:document-subtype>',
                 raw_markup, re.IGNORECASE):
        return "review"
    if re.search(r'<pubType>\s*rev\s*</pubType>', raw_markup, re.IGNORECASE):
        return "review"
    if re.search(r'<ce:dochead[^>]*>\s*<ce:textfn>\s*Review\s*</ce:textfn>',
                 raw_markup, re.IGNORECASE):
        return "review"
    return "research-article"


# ---------- 2. Triage prompts (two narrow passes, both small) ----------
# Triage produces exactly three things:
#   (1) which material is being processed   -> printed_materials[].material
#   (2) by which process                    -> printed_materials[].process_category/_subtype
#   (3) which sections to read for values   -> relevant_sections
#
# (1)+(2) are stated in the abstract/intro of a primary paper, so pass 1 sees
# ONLY the opening text. (3) needs to know every section exists, but only needs
# a name + a snippet + a cheap "does this look numeric" hint, so pass 2 sees a
# compact section listing instead of a 1500-char-per-section dump.

AM_CATEGORY_LEGEND = """AM process category codes and what techniques they cover — use this to map
any technique name mentioned in the paper to the correct code:
  VPP = Vat Photopolymerization: stereolithography (SLA), DLP, CLIP, two-photon
        polymerization (2PP), masked SLA (MSLA/LCD) — resin cured by light
  MEX = Material Extrusion: FDM, FFF — thermoplastic filament extruded through a nozzle
  PBF = Powder Bed Fusion: SLM, LPBF, DMLS, SLS, EBM — laser/e-beam fuses powder bed
  DED = Directed Energy Deposition: LENS, wire-arc AM, laser cladding — feedstock melted as deposited
  MJT = Material Jetting: PolyJet, NanoParticle Jetting, drop-on-demand — droplets jetted and cured/solidified
  BJT = Binder Jetting: liquid binder selectively deposited onto a powder bed
  SHL = Sheet Lamination: LOM, ultrasonic AM — sheets bonded layer by layer
  OTHER = use only if the paper's AM process genuinely doesn't fit any of the
        above (e.g. a truly novel/hybrid process), NOT as a default when you're
        merely unsure which of the above it is — re-read the technique name
        against this list first before choosing OTHER.
"""

IDENTIFY_PROMPT = (
    """You are reading the abstract, introduction and Materials/Methods text of an
additive manufacturing (AM) research paper to identify what was printed and how.
Do NOT extract process parameter values — identification only.

""" + AM_CATEGORY_LEGEND + """
Return a single JSON object with EXACTLY one key:

{{
  "printed_materials": [
    {{
      "material": "<the material actually FED INTO THE PRINTER as feedstock, with its FULL composition exactly as the paper states it: base material plus any dopants, fillers, loadings, ratios or additives, e.g. 'Ti-6Al-4V powder', 'ZnO doped with 0.04 wt% Al', 'photopolymer resin loaded with 5 wt% graphene'>",
      "process_category": "<the category code (from the list above) that processes THIS material>",
      "process_subtype": "<the specific technique that processes THIS material, e.g. 'stereolithography'>"
    }}
  ]
}}

Rules:
  - Composition matters: if the paper gives a dopant, filler, weight/volume/mole
    fraction, or mixing ratio for the feedstock, include it in "material" with
    the number and unit as written. Do not round, guess, or add composition
    the paper does not state.
  - List the BUILD MATERIAL only — the thing physically fed into the 3D
    printer (resin/powder/filament/wire/paste/etc). This is almost always
    ONE material. List a SECOND entry ONLY if the paper genuinely compares
    two distinct build materials/processes (e.g. two different resins each
    printed on their own). Never list more than 2.
  - Do NOT list reagents, solvents, catalysts, post-print functionalization
    chemicals, analytes, or anything used in a downstream chemistry/testing
    step. Ask yourself: "did this go INTO the printer?" If no, leave it out.
  - Each entry's process_category/process_subtype describe HOW THAT SPECIFIC
    MATERIAL was processed.
  - If you cannot identify any material that was actually printed (e.g. a pure
    literature review with no build material of its own), return an empty
    list — do not force an entry.

PAPER TEXT:
{front_text}

REMINDER: respond with exactly one JSON object with the single key
"printed_materials" (1-2 entries max, or an empty list). Nothing else.
"""
)

IDENTIFY_RETRY_NOTE = (
    "\n\nYour previous response was invalid. Respond with ONLY a JSON object "
    "with the single key \"printed_materials\": a list of AT MOST 2 objects, each "
    "with exactly the keys \"material\", \"process_category\", \"process_subtype\". "
    "\"process_category\" MUST be one of BJT, DED, MEX, MJT, PBF, SHL, VPP, OTHER "
    "(or an empty string if unknown) — not a free-text description. Use an empty "
    "list if no build material can be identified."
)

SECTION_PROMPT = """You are helping decide which sections of an additive manufacturing (AM) paper
should be read to extract experimental parameter values (laser power, layer
thickness, temperatures, particle sizes, mechanical properties, etc.).

Each section is listed as [SECTION: <name>], followed by an optional hint
(how many numeric values with units and how many tables it contains) and the
first few words of its text.

Return a single JSON object with EXACTLY one key:

{{
  "relevant_sections": ["<exact section names copied from the [SECTION: ...] markers>"]
}}

Include a section if it plausibly contains AM parameters OR generic AM knowledge:
  - Printer / resin / feedstock identification (often "Instruments", "Equipment",
    "Materials", "Chemicals and reagents", "Experimental", "Methods").
  - General process discussion or comparison tables (e.g. "Materials and printing
    processes", "3D printing technologies and materials").
  - Sections reporting the paper's own experiments and results.
  - For REVIEW articles: sections summarizing parameters reported by other papers.
A hint showing many numeric values or tables is a strong sign of relevance.
Skip pure-background sections with no parameter-like content.

SECTIONS:
{section_listing}

REMINDER: respond with exactly one JSON object with the single key
"relevant_sections". Copy section names exactly. Nothing else.
"""

SECTION_RETRY_NOTE = (
    "\n\nYour previous response was invalid. Respond with ONLY a JSON object with "
    "the single key \"relevant_sections\": a list of section-name strings copied "
    "exactly from the [SECTION: ...] markers."
)

# Sections that carry no classification signal and only add noise/token bloat —
# skipped entirely when building triage prompts. Matched as a substring against
# the (lowercased) section name, so it catches variants like "Declaration of
# competing interest" / "Declaration of generative AI...".
NOISE_SECTION_KEYWORDS = [
    "reference", "acknowledg", "declaration", "credit authorship",
    "supplementary data", "conflict of interest", "data availability",
    "graphical abstract", "keywords",
]

def _is_noise_section(name):
    name_lower = name.lower()
    return any(kw in name_lower for kw in NOISE_SECTION_KEYWORDS)


# ---- pass-1 input: abstract + introduction only ----
METHODS_SECTION_HINTS = ("method", "material", "experiment", "instrument",
                         "equipment", "printing", "fabrication")

def build_front_text(sectioned, max_abstract=3000, max_intro=3000, max_fallback=1500):
    """Abstract + introduction text for material/process identification. If
    neither heading exists, falls back to front-matter ('body') plus the first
    couple of real sections. Plain-text input (only a 'body' bucket) just uses
    the start of that text."""
    real = {k: v for k, v in sectioned.items() if k != "body"}
    if not real:
        return sectioned.get("body", "")[: max_abstract + max_intro]

    parts, found = [], False
    for name, text in real.items():
        n = _normalize_section_name(name)
        if "abstract" in n and not _is_noise_section(name):
            parts.append(f"[ABSTRACT]\n{text[:max_abstract]}")
            found = True
        elif n.startswith("introduction") or n == "background":
            parts.append(f"[INTRODUCTION]\n{text[:max_intro]}")
            found = True

    if not found:
        body = sectioned.get("body", "")
        if body:
            parts.append(f"[FRONT MATTER]\n{body[:max_fallback]}")
        taken = 0
        for name, text in real.items():
            if _is_noise_section(name):
                continue
            parts.append(f"[SECTION: {name}]\n{text[:max_fallback]}")
            taken += 1
            if taken >= 2:
                break
    return "\n\n".join(parts)


def build_methods_text(sectioned, max_chars=1500, max_sections=3):
    """Used only when the opening text yielded no printed material: the first
    few Methods/Materials-style sections, truncated."""
    parts = []
    for name, text in sectioned.items():
        if name == "body" or _is_noise_section(name):
            continue
        if any(h in name.lower() for h in METHODS_SECTION_HINTS):
            parts.append(f"[SECTION: {name}]\n{text[:max_chars]}")
            if len(parts) >= max_sections:
                break
    return "\n\n".join(parts)


# ---- pass-2 input: compact section listing ----
UNIT_VALUE_RE = re.compile(
    r"\d+(?:\.\d+)?\s?(?:W|mm/s|m/s|[\u00b5\u03bcu]m|nm|mm|\u00b0C|MPa|GPa|J/mm3|kV|mA|rpm|min|h|s|%)(?![A-Za-z])"
)

def _section_hint(text):
    """Cheap symbolic signal so the LLM doesn't have to read a section to know
    it's number-dense: count of 'value + unit' matches and of table blocks."""
    n_vals = len(UNIT_VALUE_RE.findall(text))
    n_tables = text.count("[TABLE")
    bits = []
    if n_vals:
        bits.append(f"~{n_vals} numeric values w/ units")
    if n_tables:
        bits.append(f"{n_tables} table(s)")
    return f" ({', '.join(bits)})" if bits else ""


def build_section_listing(sectioned, snippet_chars=200, max_sections=60):
    items = [(n, t) for n, t in sectioned.items() if n != "body" and not _is_noise_section(n)]
    if len(items) > 40:
        snippet_chars = 80  # keep the listing small on papers with many headings
    lines = []
    for name, text in items[:max_sections]:
        snippet = re.sub(r"\s+", " ", text[:snippet_chars]).strip()
        lines.append(f"[SECTION: {name}]{_section_hint(text)} {snippet}...")
    return "\n".join(lines)


def _is_hollow(row):
    value = str(row.get("value", "") or "").strip()
    quote = str(row.get("quote", "") or "").strip()
    return not value and not quote

def _is_explicitly_missing(row):
    """Model said 'not reported in the paper' — treat as a null, not a value."""
    value = str(row.get("value", "") or "").strip().lower()
    return value in {"not reported", "not reported in the paper", "n/a", "na",
                     "none", "not stated", "not specified", "not given", "unknown"}


# ---------- 3. Call LLM — backend is togglable: local Ollama (default,
# unchanged) or GPT-5-mini (same call llm_pipeline.py uses for extraction).
# Ollama is NOT a hard dependency: it's only imported/hit over HTTP when
# backend="ollama" is actually used, and gpt5mini needs no local server at
# all — so running triage with backend="gpt5mini" works with no Ollama
# installed or running anywhere.
def _call_llm_ollama(prompt, model="llama3.1", host="http://localhost:11434"):
    import urllib.request, urllib.error
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 4096, "num_ctx": 8192},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        raise RuntimeError(f"Ollama returned {e.code}: {err_body}") from None
    return body["response"]


def _call_llm_gpt5mini(prompt, model="gpt-5-mini"):
    # Lazy import: keeps the openai package (and OPENAI_API_KEY requirement)
    # out of the ollama-backend path entirely.
    from llm_pipeline import call_gpt_mini
    return call_gpt_mini(prompt, model=model)


VALID_CATEGORY_CODES = {"BJT", "DED", "MEX", "MJT", "PBF", "SHL", "VPP", "OTHER"}


# ---------- Tool-calling schemas for the gpt5mini backend ----------
# These mirror IDENTIFY_PROMPT / SECTION_PROMPT's JSON shapes exactly, but as
# an enforced (strict-mode) function-call schema instead of free text the
# model might wrap in prose or ```json fences. This is what makes triage
# "tool callable": each pass is one forced call to one named tool, and the
# return value is already a validated dict — no parse_response/_valid_*
# round-trip needed for this backend.
PRINTED_MATERIALS_TOOL = {
    "name": "submit_printed_materials",
    "description": (
        "Submit the build material(s) identified as physically fed into the "
        "3D printer, and the AM process category/subtype used for each."
    ),
    "schema": {
        "type": "object",
        "properties": {
            "printed_materials": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "material": {
                            "type": "string",
                            "description": "Full composition as stated: base material plus any "
                                           "dopants/fillers/ratios, e.g. 'Ti-6Al-4V powder'.",
                        },
                        "process_category": {
                            "type": "string",
                            "enum": sorted(VALID_CATEGORY_CODES) + [""],
                        },
                        "process_subtype": {
                            "type": "string",
                            "description": "Specific technique name, e.g. 'stereolithography'.",
                        },
                    },
                    "required": ["material", "process_category", "process_subtype"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["printed_materials"],
        "additionalProperties": False,
    },
}

RELEVANT_SECTIONS_TOOL = {
    "name": "submit_relevant_sections",
    "description": "Submit the exact section names worth reading for AM process parameter values.",
    "schema": {
        "type": "object",
        "properties": {
            "relevant_sections": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Section names copied exactly from the [SECTION: ...] markers.",
            }
        },
        "required": ["relevant_sections"],
        "additionalProperties": False,
    },
}


def _call_tool_and_validate(prompt, tool, validate, model="gpt-5-mini"):
    """One tool call + one retry (in case the model returns an empty/degenerate
    but schema-valid shape, e.g. VALID_CATEGORY_CODES rejects nothing since
    the enum already constrains it — this just guards transient API hiccups)."""
    from llm_pipeline import call_gpt_mini_tool
    for attempt in range(2):
        try:
            parsed = call_gpt_mini_tool(
                prompt, tool["name"], tool["description"], tool["schema"], model=model
            )
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Warning: tool call attempt {attempt + 1} failed ({e}).")
            continue
        if validate(parsed):
            return parsed
        print("Warning: tool call response failed schema validation.")
    return None


def call_llm(prompt, model="gpt-5-mini", host="http://localhost:11434", backend="gpt5mini"):
    """backend: "gpt5mini" (default — routes to llm_pipeline.call_gpt_mini;
    ignores `host`) or "ollama" (local server, opt-in).
    `model` should be the model name appropriate to whichever backend is
    selected (e.g. "gpt-5-mini" for gpt5mini, "llama3.1" for ollama) —
    callers that only flip `backend` without changing `model` get the
    default "gpt-5-mini" sent to Ollama's endpoint, so pass both together."""
    if backend == "gpt5mini":
        return _call_llm_gpt5mini(prompt, model=model)
    elif backend == "ollama":
        return _call_llm_ollama(prompt, model=model, host=host)
    raise ValueError(f"backend must be 'ollama' or 'gpt5mini', got {backend!r}")


def _normalize_keys(obj):
    """Recursively strip leading/trailing whitespace from dict keys."""
    if isinstance(obj, dict):
        return {k.strip(): _normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_keys(item) for item in obj]
    return obj

def parse_response(raw_text):
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
    return _normalize_keys(json.loads(cleaned))


# ---------- 4. Validation + the two triage passes ----------
def _valid_printed(parsed):
    """{"printed_materials": [<=2 dicts with material/process_category/process_subtype]}.
    An empty list is valid — a review may genuinely have no build material."""
    if not isinstance(parsed, dict) or "printed_materials" not in parsed:
        return False
    printed = parsed["printed_materials"]
    if not isinstance(printed, list) or len(printed) > 2:
        return False
    for entry in printed:
        if not isinstance(entry, dict):
            return False
        if not {"material", "process_category", "process_subtype"}.issubset(entry.keys()):
            return False
        if entry["process_category"] and entry["process_category"] not in VALID_CATEGORY_CODES:
            return False
    return True


def _valid_sections(parsed):
    return (isinstance(parsed, dict)
            and isinstance(parsed.get("relevant_sections"), list)
            and all(isinstance(s, str) for s in parsed["relevant_sections"]))


def _call_and_validate(prompt, validate, retry_note, **llm_kwargs):
    """One call + one retry. Returns the parsed dict, or None if both attempts
    fail to parse/validate."""
    for attempt in range(2):
        p = prompt if attempt == 0 else prompt + retry_note
        try:
            parsed = parse_response(call_llm(p, **llm_kwargs))
        except json.JSONDecodeError as e:
            print(f"Warning: triage response was not valid JSON ({e}).")
            continue
        if validate(parsed):
            return parsed
        print("Warning: triage response failed schema validation.")
    return None


def identify_material_process(sectioned, source_kind="research-article",
                              model="gpt-5-mini", host="http://localhost:11434",
                              backend="gpt5mini"):
    """Pass 1 — outputs (1) and (2): which material, by which process.
    Sees abstract + introduction + the first few Materials/Methods sections
    (truncated). Returns a list of 0-2 {material, process_category, process_subtype}.

    On the gpt5mini backend this is a forced tool call (submit_printed_materials)
    instead of a free-text-JSON-then-parse round trip."""
    # Abstract/intro name the material and process; the exact composition
    # (dopant %, filler loading, ratios) is usually only in Materials/Methods.
    text = build_front_text(sectioned)
    methods = build_methods_text(sectioned, max_chars=2500)
    if methods:
        text += "\n\n" + methods
    prompt = IDENTIFY_PROMPT.format(front_text=text)

    if backend == "gpt5mini":
        result = _call_tool_and_validate(prompt, PRINTED_MATERIALS_TOOL, _valid_printed, model=model)
    else:
        result = _call_and_validate(prompt, _valid_printed, IDENTIFY_RETRY_NOTE,
                                    model=model, host=host, backend=backend)
    return result["printed_materials"] if result else []


def score_section_relevance(sectioned, model="gpt-5-mini", host="http://localhost:11434",
                            backend="gpt5mini"):
    """Pass 2 — output (3): which sections to read for parameter values.
    Sees every non-noise section as name + numeric-density hint + short snippet.
    Returns a list of section names ([] on failure — get_relevant_text() then
    falls back to all non-'body' sections).

    On the gpt5mini backend this is a forced tool call (submit_relevant_sections)."""
    listing = build_section_listing(sectioned)
    if not listing:
        return []
    prompt = SECTION_PROMPT.format(section_listing=listing)

    if backend == "gpt5mini":
        result = _call_tool_and_validate(prompt, RELEVANT_SECTIONS_TOOL, _valid_sections, model=model)
    else:
        result = _call_and_validate(prompt, _valid_sections, SECTION_RETRY_NOTE,
                                    model=model, host=host, backend=backend)
    return result["relevant_sections"] if result else []


# ---------- Keyword-based category detection (no LLM — deterministic safety net) ----------
# The LLM triage step has repeatedly under-detected categories even with an explicit
# legend (e.g. missing VPP on a paper that says "stereolithography" outright). Technique
# names are a closed, well-known vocabulary — a plain keyword scan catches these reliably
# where the LLM's classification judgment has been inconsistent. Used to supplement
# (never silently replace) the LLM's own guess.
CATEGORY_KEYWORDS = {
    "VPP": ["stereolithography", "vat photopolymerization", "vat polymerization",
            " sla ", "sla printer", "sla-printed", "dlp printer", "digital light processing",
            "clip printing", "two-photon polymerization", "masked sla", "mlsa", "resin printer",
            "photopolymer resin", "uv-curable resin", "form3", "formlabs"],
    "MEX": ["fused deposition modeling", "fused filament fabrication", " fdm ", " fff ",
            "material extrusion", "thermoplastic filament", "extrusion 3d printing"],
    "PBF": ["powder bed fusion", "selective laser melting", " slm ", "laser powder bed fusion",
            " lpbf ", "direct metal laser sintering", " dmls ", "selective laser sintering",
            " sls ", "electron beam melting", " ebm "],
    "DED": ["directed energy deposition", " ded ", "laser engineered net shaping", " lens ",
            "wire arc additive", "laser cladding", "blown powder deposition"],
    "MJT": ["material jetting", "polyjet", "nanoparticle jetting", "drop-on-demand printing",
            "multi jet fusion"],
    "BJT": ["binder jetting", "binder jet printing"],
    "SHL": ["sheet lamination", "laminated object manufacturing", " lom ",
            "ultrasonic additive manufacturing"],
}


# Author-voice patterns: a technique name only counts if the authors are
# talking about THEIR OWN use of it, not mentioning it in a literature
# review or intro. This is what stops "FDM and SL are the most common
# techniques" from dispatching MEX.
AUTHOR_USE_PATTERNS = {
    "VPP": [
        r"\bwe (used|printed|fabricated|employed|built)\b[^.]{0,40}\b(sla|dlp|lfs|clip|msla|stereolithography|vat photopolymerization|two-photon)\b",
        r"\b(print(ed)?|fabricat(ed|ion))\b[^.]{0,30}\b(on|with|using)\b[^.]{0,20}\b(form\s?\d|formlabs|asiga|miicraft|b9\s?creator|anycubic|elegoo|photon|ember)\b",
        r"\bour\b[^.]{0,20}\b(sla|dlp|lfs|stereolithography)\b",
        r"\b(lfs|sla|dlp)\s?(3d\s?)?printer\b",
        r"\bstereolithograph(y|ic)\b[^.]{0,30}\b(printer|print|resin|device)\b",
        r"\bdigital light process(ing|or)\b[^.]{0,30}\b(printer|print|resin|device)\b",
    ],
    "MEX": [
        r"\bwe (used|printed|fabricated|employed|built)\b[^.]{0,40}\b(fdm|fff|fused (deposition|filament)|material extrusion|extruder|nozzle)\b",
        r"\b(print(ed)?|fabricat(ed|ion))\b[^.]{0,30}\b(on|with|using)\b[^.]{0,20}\b(ultimaker|prusa|makerbot|felix|rova|raise3d|ender|creality|3dtouch)\b",
        r"\bour\b[^.]{0,20}\b(fdm|fff|extruder|nozzle)\b",
        r"\bfused (deposition|filament)\b[^.]{0,30}\b(printer|print|filament|extruder)\b",
    ],
    "PBF": [
        r"\bwe (used|printed|fabricated|employed|built)\b[^.]{0,40}\b(slm|lpbf|dmls|sls|ebm|powder bed fusion|selective laser)\b",
        r"\b(print(ed)?|fabricat(ed|ion))\b[^.]{0,30}\b(on|with|using)\b[^.]{0,20}\b(eos|realizer|renishaw|concept laser|slm solutions|m\d{2,3}|eosint)\b",
        r"\blaser powder bed fusion\b[^.]{0,30}\b(system|printer|machine|process)\b",
        r"\belectron beam melt(ing)?\b[^.]{0,30}\b(system|printer|machine|process)\b",
    ],
    "DED": [
        r"\bwe (used|deposited|fabricated|employed|built)\b[^.]{0,40}\b(ded|lens|wire arc|laser cladding|directed energy)\b",
        r"\b(directed energy deposition|laser engineered net shaping|wire arc additive)\b[^.]{0,30}\b(system|process|deposition|cladding)\b",
    ],
    "MJT": [
        r"\bwe (used|printed|fabricated|employed)\b[^.]{0,40}\b(polyjet|material jetting|nanoparticle jetting|drop-on-demand|inkjet)\b",
        r"\b(polyjet|material jetting|nanoparticle jetting)\b[^.]{0,30}\b(printer|print|system|machine)\b",
    ],
    "BJT": [
        r"\bwe (used|printed|fabricated|employed)\b[^.]{0,40}\b(binder jet(ting)?)\b",
        r"\bbinder jet(ting)?\b[^.]{0,30}\b(printer|print|system|machine|process)\b",
    ],
    "SHL": [
        r"\bwe (used|printed|fabricated|employed)\b[^.]{0,40}\b(sheet lamination|laminated object|ultrasonic additive)\b",
        r"\b(sheet lamination|ultrasonic additive manufacturing)\b[^.]{0,30}\b(system|process|machine)\b",
    ],
}

def detect_categories_by_author_voice(text):
    """Returns set of category codes where the authors appear to be describing
    their OWN use of that technique, not just mentioning it. Uses regex patterns
    anchored on first-person/possessive language and printer-brand names."""
    text_lower = " " + text.lower() + " "
    found = set()
    for code, patterns in AUTHOR_USE_PATTERNS.items():
        if any(re.search(p, text_lower) for p in patterns):
            found.add(code)
    return found

def detect_categories_by_keyword(text):
    """Scans raw text (case-insensitive) for known AM technique keywords and returns
    the set of category codes with at least one hit. Deterministic, no LLM involved."""
    text_lower = " " + text.lower() + " "  # pad so " sla " style boundary keywords can match at edges
    found = set()
    for code, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.add(code)
    return found


def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def run_triage(raw_markup_or_text, is_markup=True, model="gpt-5-mini",
                host="http://localhost:11434", backend="gpt5mini"):
    """Returns (triage, sectioned). triage contains:

      printed_materials  [{material, process_category, process_subtype}]  (LLM pass 1)
      relevant_sections  [section names]                                  (LLM pass 2)
      process_categories derived: printed_materials' categories + keyword net
      process_subtypes   derived: printed_materials' process_subtype values
      _source_kind       'review' | 'research-article'

    process_categories / process_subtypes are NOT asked of the LLM — they're
    derived so downstream code (agentic_pipeline's dispatch and its review
    fallback) keeps working with the same keys.

    backend: "gpt5mini" (default — tool-calling, no local Ollama needed) or
    "ollama" (local llama3.1, opt-in, free-text-JSON parsing as before). If
    backend="ollama" and `model` is left at its default, it's swapped to
    "llama3.1" automatically."""
    if backend not in ("ollama", "gpt5mini"):
        raise ValueError(f"backend must be 'ollama' or 'gpt5mini', got {backend!r}")
    if backend == "ollama" and model == "gpt-5-mini":
        model = "llama3.1"

    sectioned = strip_markup_sectioned(raw_markup_or_text) if is_markup else {"body": raw_markup_or_text}
    source_kind = detect_source_kind(raw_markup_or_text) if is_markup else "research-article"
    print(f"Note: source kind = {source_kind}")

    llm = dict(model=model, host=host, backend=backend)
    printed = identify_material_process(sectioned, source_kind, **llm)
    relevant = score_section_relevance(sectioned, **llm)

    # ---- derive process_categories / process_subtypes (no LLM) ----
    full_text = "\n".join(sectioned.values())
    categories = {e["process_category"] for e in printed if e.get("process_category")}
    categories |= detect_categories_by_author_voice(full_text)  # existing safety net
    if source_kind == "review" or not printed:
        # Reviews (and papers where nothing was pinned) discuss several
        # techniques and rarely use first-person "we printed on..." phrasing,
        # so author-voice alone finds nothing. Fall back to the broader
        # mention-based keyword scan so category-level extraction still has
        # categories to dispatch on.
        kw = detect_categories_by_keyword(full_text)
        if kw - categories:
            print(f"Note: keyword scan supplied categories {sorted(kw - categories)}.")
        categories |= kw
    if "OTHER" in categories and len(categories) > 1:
        categories.discard("OTHER")

    triage = {
        "printed_materials": printed,
        "relevant_sections": relevant,
        "process_categories": sorted(categories),
        "process_subtypes": _dedupe(e.get("process_subtype") for e in printed),
        "_source_kind": source_kind,
    }
    json.dump(triage, open("triage_result.json", "w"), indent=2)
    print(json.dumps(triage, indent=2))
    return triage, sectioned


def _normalize_section_name(name):
    """Strips leading numbering ('2.1 ', '2.6 ') and punctuation, lowercases —
    so a hallucinated 'Instruments' vs a real 'Instruments' still match, and so
    does a real name against an LLM guess with numbering added/removed. Also
    strips a literal '[SECTION: ...]' wrapper in case the model echoed the
    marker syntax back instead of just the name."""
    import re
    cleaned = name.strip()
    wrapper_match = re.match(r"^\[SECTION:\s*(.*?)\]$", cleaned)
    if wrapper_match:
        cleaned = wrapper_match.group(1)
    cleaned = re.sub(r"^\s*\d+(\.\d+)*\.?\s*", "", cleaned)  # strip leading "2.1 " style numbering
    return cleaned.strip().lower()


def get_relevant_text(triage, sectioned):
    """
    Filters the full sectioned text down to only what triage flagged as
    relevant, for feeding into the stage-2 detailed extraction pass.

    Matches section names fuzzily (case/numbering-insensitive, substring match)
    since the LLM sometimes paraphrases or adds/removes numbering rather than
    copying the exact detected section name verbatim.

    Falls back to everything EXCEPT the 'body' bucket if triage didn't identify
    any relevant sections — 'body' is pre-heading catch-all text (often XML
    front-matter: DOI, ISSN, author lists) and is essentially never useful
    extraction content, so it's excluded from the safety-net fallback too.
    """
    relevant_names = triage.get("relevant_sections", [])
    real_sections = {k: v for k, v in sectioned.items() if k != "body"}

    if not relevant_names:
        return "\n\n".join(real_sections.values())

    picked = []
    used_keys = set()
    for guess in relevant_names:
        guess_norm = _normalize_section_name(guess)
        for real_name, text in real_sections.items():
            if real_name in used_keys:
                continue
            real_norm = _normalize_section_name(real_name)
            if guess_norm == real_norm or guess_norm in real_norm or real_norm in guess_norm:
                picked.append(f"[SECTION: {real_name}]\n{text}")
                used_keys.add(real_name)
                break

    if not picked:  # no fuzzy matches either — fall back, still excluding 'body'
        return "\n\n".join(real_sections.values())
    return "\n\n".join(picked)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run paper triage.")
    ap.add_argument("paper_path", nargs="?", default=None,
                     help="Path to a paper markup file. Omit to use the built-in sample paper.")
    ap.add_argument("--backend", choices=["ollama", "gpt5mini"], default="gpt5mini",
                     help="Which LLM backend runs triage (default: gpt5mini, tool-calling, "
                          "needs only OPENAI_API_KEY). 'ollama' uses a local llama3.1 server.")
    ap.add_argument("--model", default=None,
                     help="Model name for the chosen backend. Defaults to 'gpt-5-mini' for "
                          "gpt5mini, 'llama3.1' for ollama.")
    ap.add_argument("--host", default="http://localhost:11434",
                     help="Ollama server URL (ignored for --backend gpt5mini).")
    args = ap.parse_args()

    if args.paper_path:
        with open(args.paper_path, encoding="utf-8") as f:
            sample_markup = f.read()
    else:
        sample_markup = """
        <html><body>
        <h2>Introduction</h2>
        <p>Additive manufacturing has grown rapidly in the last decade.</p>
        <h2>Methods</h2>
        <p>Ti-6Al-4V powder was processed using laser powder bed fusion (LPBF)
        with a laser power of 195 W and scan speed of 1100 mm/s.</p>
        <h2>Results</h2>
        <p>Relative density reached 99.4% after hot isostatic pressing.</p>
        <h2>References</h2>
        <p>[1] Smith et al. 2020.</p>
        </body></html>
        """

    model = args.model or ("llama3.1" if args.backend == "ollama" else "gpt-5-mini")
    triage, sectioned = run_triage(sample_markup, model=model, host=args.host, backend=args.backend)
    relevant_text = get_relevant_text(triage, sectioned)
    print("\n--- TEXT TO SEND TO STAGE-2 EXTRACTOR ---")
    print(relevant_text[:2000], "..." if len(relevant_text) > 2000 else "") 