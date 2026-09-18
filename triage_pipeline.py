"""
Intro / triage pass.

Goal: cheaply scan a paper's raw XML/HTML and answer a few basic questions
BEFORE running the full taxonomy-driven extraction. This lets you narrow
which part of the taxonomy to send in the detailed pass (see llm_pipeline.py),
instead of always sending all ~373 parameters across all 8 AM categories.

Output is small and cheap: process category, materials mentioned, and which
taxonomy sections are even worth checking.
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


# ---------- 2. Triage prompt (small, cheap) ----------
TRIAGE_PROMPT = """You are scanning an additive manufacturing (AM) research paper to identify
its basic contents. Do NOT extract detailed parameter values yet — this is
only a quick triage pass.

The paper's actual sections (detected from markup) are listed below as
[SECTION: <name>] markers, so you know the real section names to use.

AM process category codes and what techniques they cover — use this to map
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

You MUST return a single JSON object with EXACTLY these five keys — no more,
no fewer, no other shape:

{{
  "process_categories": ["<AM category codes present, from: BJT, DED, MEX, MJT, PBF, SHL, VPP, OTHER>"],
  "process_subtypes": ["<specific techniques mentioned, e.g. LPBF, FDM, DED-Wire>"],
  "materials_mentioned": ["<material/alloy names as they appear in the text>"],
  "relevant_sections": ["<exact section names from the [SECTION: ...] markers that likely contain extractable parameter values, e.g. Methods, Materials, Results>"],
  "likely_relevant_taxonomy_sections": ["<AM taxonomy section names ONLY, in the exact 'Process Parameters — <subtype>' or 'Feedstock Properties — <material type>' format, e.g. 'Process Parameters — VAT-SLA', 'Feedstock Properties — Resin'. Do NOT use broad subject labels like 'Materials Science' or 'Chemistry' — those are not taxonomy sections.>"]
}}

Include a section in "relevant_sections" if it plausibly contains AM
parameters OR generic AM knowledge. This includes:
  - Sections with printer/resin/feedstock identifications (often called
    "Instruments", "Equipment", "Materials", "Chemicals and reagents",
    "Experimental", "Methods").
  - Sections with general process discussion (e.g. "Materials and printing
    processes", "3D printing technologies", "Materials for AM").
  - Sections with comparison tables (e.g. "3D printing technologies and
    materials", "Technologies and materials for ...").
  - Sections reporting the paper's own experiments.
  - For REVIEW articles: sections summarizing parameters reported by other
    papers are relevant. Include them.
Skip only References, Acknowledgements, Declarations, and pure-background
sections with no parameter-like content.

PAPER TEXT:
{paper_text}

REMINDER: your entire response must be exactly one JSON object with these five
keys: process_categories, process_subtypes, materials_mentioned,
relevant_sections, likely_relevant_taxonomy_sections. Do not return anything
else, do not return a subset of these keys, do not summarize the text instead.
"""

# Sections that carry no classification signal and only add noise/token bloat —
# skipped entirely when building the triage prompt. Matched as a substring against
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


def build_triage_prompt(sectioned_text, max_chars_per_section=1500, max_sections=15):
    """sectioned_text: dict of {section_name: text}, as returned by strip_markup_sectioned.
    Truncates each section for the triage pass — it only needs enough text to
    classify materials/process/relevance, not the full body (that's stage 2's job).
    Boilerplate sections (references, acknowledgements, etc.) are skipped entirely —
    they add token bloat and noise without helping classification, and on smaller
    local models can crowd out the actual instructions. Total section count is also
    capped as a second safeguard against prompt bloat on papers with many sections."""
    parts = []
    included = 0
    for name, text in sectioned_text.items():
        if _is_noise_section(name):
            continue
        if included >= max_sections:
            break
        snippet = text[:max_chars_per_section]
        if len(text) > max_chars_per_section:
            snippet += " ...[truncated]"
        parts.append(f"[SECTION: {name}]\n{snippet}")
        included += 1
    labeled = "\n\n".join(parts)
    return TRIAGE_PROMPT.format(paper_text=labeled)


def _is_hollow(row):
    value = str(row.get("value", "") or "").strip()
    quote = str(row.get("quote", "") or "").strip()
    return not value and not quote

def _is_explicitly_missing(row):
    """Model said 'not reported in the paper' — treat as a null, not a value."""
    value = str(row.get("value", "") or "").strip().lower()
    return value in {"not reported", "not reported in the paper", "n/a", "na",
                     "none", "not stated", "not specified", "not given", "unknown"}


# ---------- 3. Call LLM (same Ollama setup as llm_pipeline.py) ----------
def call_llm(prompt, model="llama3.1", host="http://localhost:11434"):
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


# ---------- 4. Full triage pipeline ----------
REQUIRED_TRIAGE_KEYS = {
    "process_categories", "process_subtypes", "materials_mentioned",
    "relevant_sections", "likely_relevant_taxonomy_sections",
}

VALID_CATEGORY_CODES = {"BJT", "DED", "MEX", "MJT", "PBF", "SHL", "VPP", "OTHER"}


def _validate_triage_schema(triage):
    """Returns True only if the parsed response has all required keys AND
    process_categories contains only real taxonomy codes (not free-text guesses
    like 'Synthesis' or '3D Printing')."""
    if not (isinstance(triage, dict) and REQUIRED_TRIAGE_KEYS.issubset(triage.keys())):
        return False
    categories = triage.get("process_categories", [])
    if not isinstance(categories, list):
        return False
    # empty list is fine (paper may genuinely have no AM content), but any
    # non-empty entry must be a real code
    return all(c in VALID_CATEGORY_CODES for c in categories)


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


def run_triage(raw_markup_or_text, is_markup=True, model="llama3.1", host="http://localhost:11434"):
    sectioned = strip_markup_sectioned(raw_markup_or_text) if is_markup else {"body": raw_markup_or_text}
    prompt = build_triage_prompt(sectioned)

    raw = call_llm(prompt, model=model, host=host)
    triage = parse_response(raw)

    if not _validate_triage_schema(triage):
        bad_categories = [c for c in triage.get("process_categories", [])
                           if c not in VALID_CATEGORY_CODES] if isinstance(triage, dict) else []
        print(f"Warning: triage response invalid "
              f"(keys: {list(triage.keys()) if isinstance(triage, dict) else type(triage)}, "
              f"bad category codes: {bad_categories}). Retrying once...")
        retry_prompt = prompt + (
            "\n\nYour previous response was invalid. Two requirements you may have missed:\n"
            f"1. Respond with ONLY the JSON object containing exactly these five keys: "
            f"{sorted(REQUIRED_TRIAGE_KEYS)}.\n"
            f"2. \"process_categories\" MUST contain ONLY codes from this exact list: "
            f"{sorted(VALID_CATEGORY_CODES)} — not free-text descriptions of what the paper "
            f"does. If the paper's 3D printing process doesn't clearly match one of BJT, DED, "
            f"MEX, MJT, PBF, SHL, VPP, use \"OTHER\". If unsure, use an empty list rather than "
            f"inventing a category name."
        )
        raw = call_llm(retry_prompt, model=model, host=host)
        triage = parse_response(raw)

        if not _validate_triage_schema(triage):
            print("Warning: retry also failed schema validation. Falling back to empty triage "
                  "(no categories/sections identified — extraction will likely find nothing).")
            triage = {k: [] for k in REQUIRED_TRIAGE_KEYS}

    # Deterministic keyword safety net: the LLM has repeatedly under-detected
    # categories even with an explicit legend. Scan the actual paper text (not
    # the LLM's output) for known technique keywords and add any category the
    # LLM missed — never remove what the LLM found, only supplement it.
    full_text = "\n".join(sectioned.values())
    keyword_categories = detect_categories_by_author_voice(full_text)
    llm_categories = set(triage.get("process_categories", []))
    missing = keyword_categories - llm_categories
    if missing:
        print(f"Note: keyword scan found technique terms for {sorted(missing)} that the "
              f"LLM's category list missed — adding to process_categories.")
        triage["process_categories"] = sorted(llm_categories | keyword_categories)
        # OTHER is meaningless once a real category is confirmed by keyword evidence
        if "OTHER" in triage["process_categories"] and len(triage["process_categories"]) > 1:
            triage["process_categories"].remove("OTHER")

    triage["_source_kind"] = detect_source_kind(raw_markup_or_text) if is_markup else "research-article"
    print(f"Note: source kind = {triage['_source_kind']}")
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
    import sys

    if len(sys.argv) > 1:
        # usage: python3 triage_pipeline.py path/to/paper.xml
        with open(sys.argv[1], encoding="utf-8") as f:
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

    triage, sectioned = run_triage(sample_markup)
    relevant_text = get_relevant_text(triage, sectioned)
    print("\n--- TEXT TO SEND TO STAGE-2 EXTRACTOR ---")
    print(relevant_text[:2000], "..." if len(relevant_text) > 2000 else "")
