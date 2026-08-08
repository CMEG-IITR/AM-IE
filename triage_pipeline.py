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
HEADING_TAGS = {"h1", "h2", "h3", "h4", "title", "sec-title", "section-title"}

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
        if tag.lower() in HEADING_TAGS:
            self._in_heading = True
            self._heading_buf = []

    def handle_endtag(self, tag):
        if tag.lower() in HEADING_TAGS and self._in_heading:
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
    parser = SectionAwareExtractor()
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


# ---------- 2. Triage prompt (small, cheap) ----------
TRIAGE_PROMPT = """You are scanning an additive manufacturing (AM) research paper to identify
its basic contents. Do NOT extract detailed parameter values yet — this is
only a quick triage pass.

The paper's actual sections (detected from markup) are listed below as
[SECTION: <name>] markers, so you know the real section names to use.

Return ONLY valid JSON in this shape:

{{
  "process_categories": ["<AM category codes present, from: BJT, DED, MEX, MJT, PBF, SHL, VPP, OTHER>"],
  "process_subtypes": ["<specific techniques mentioned, e.g. LPBF, FDM, DED-Wire>"],
  "materials_mentioned": ["<material/alloy names as they appear in the text>"],
  "relevant_sections": ["<exact section names from the [SECTION: ...] markers that likely contain extractable parameter values, e.g. Methods, Materials, Results>"],
  "likely_relevant_taxonomy_sections": ["<your best guess at which taxonomy sections are worth checking in detail, e.g. 'Process Parameters — PBF-LB Metal', 'Feedstock Properties — Powder'>"]
}}

Only list a section in "relevant_sections" if it plausibly contains process
parameters, material properties, or measured values — skip sections like
References, Acknowledgements, or pure Introduction/Background text with no
reported values.

PAPER TEXT:
{paper_text}
"""

def build_triage_prompt(sectioned_text, max_chars_per_section=1500):
    """sectioned_text: dict of {section_name: text}, as returned by strip_markup_sectioned.
    Truncates each section for the triage pass — it only needs enough text to
    classify materials/process/relevance, not the full body (that's stage 2's job)."""
    parts = []
    for name, text in sectioned_text.items():
        snippet = text[:max_chars_per_section]
        if len(text) > max_chars_per_section:
            snippet += " ...[truncated]"
        parts.append(f"[SECTION: {name}]\n{snippet}")
    labeled = "\n\n".join(parts)
    return TRIAGE_PROMPT.format(paper_text=labeled)


# ---------- 3. Call LLM (same Ollama setup as llm_pipeline.py) ----------
def call_llm(prompt, model="llama3.1", host="http://localhost:11434"):
    import urllib.request, urllib.error
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
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


def parse_response(raw_text):
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
    return json.loads(cleaned)


# ---------- 4. Full triage pipeline ----------
def run_triage(raw_markup_or_text, is_markup=True):
    sectioned = strip_markup_sectioned(raw_markup_or_text) if is_markup else {"body": raw_markup_or_text}
    prompt = build_triage_prompt(sectioned)
    raw = call_llm(prompt)
    triage = parse_response(raw)
    json.dump(triage, open("triage_result.json", "w"), indent=2)
    print(json.dumps(triage, indent=2))
    return triage, sectioned


def get_relevant_text(triage, sectioned):
    """
    Filters the full sectioned text down to only what triage flagged as
    relevant, for feeding into the stage-2 detailed extraction pass.
    Falls back to everything if triage didn't identify any relevant sections
    (better to over-include than silently extract nothing).
    """
    relevant_names = triage.get("relevant_sections", [])
    if not relevant_names:
        return "\n\n".join(sectioned.values())

    picked = []
    for name in relevant_names:
        if name in sectioned:
            picked.append(f"[SECTION: {name}]\n{sectioned[name]}")
    if not picked:  # names didn't match exactly, fall back
        return "\n\n".join(sectioned.values())
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
