"""
Regex-based key-point extractor for AM papers, driven by the taxonomy vocab.

Strategy:
  For each known parameter name (e.g. "Layer Thickness"), build a regex that
  looks for the name (allowing common separators/case) followed within a
  short window by a number and, if the taxonomy specifies one, the expected
  unit (allowing common unit aliases).

  This catches the dominant AM-paper phrasing patterns:
    "layer thickness of 30 µm"
    "layer thickness: 30 um"
    "layer thickness (LT) = 30 μm"
    "a 30 μm layer thickness"
"""
import json, re

from pypdf import PdfReader

flat = json.load(open("vocab_flat.json"))

NUM = r"(\d+(?:\.\d+)?(?:\s*[-–to]{1,3}\s*\d+(?:\.\d+)?)?)"  # supports "20-30" ranges

# common unicode/ascii unit aliases so "µm" also matches "um", "μm"
UNIT_ALIASES = {
    "µm": r"(?:µm|μm|um)",
    "°C": r"(?:°C|degC|deg C)",
    "g/cm³": r"(?:g/cm3|g/cm³|g/cc)",
    "mm/s": r"mm/s",
    "mm³": r"(?:mm3|mm³)",
    "W": r"\bW\b",
    "hr": r"(?:hr|hrs|h\b)",
    "%": r"%",
    "ppm": r"ppm",
    "wt%": r"(?:wt%|wt\.%|weight%)",
    "s/50g": r"s/50\s*g",
    "pL": r"pL",
    "°C/min": r"(?:°C/min|deg C/min)",
}

def unit_pattern(unit):
    if not unit:
        return ""
    return UNIT_ALIASES.get(unit, re.escape(unit))

def build_pattern(name, unit):
    name_esc = re.escape(name)
    # allow "-" or space variants, case-insensitive
    name_pat = name_esc.replace(r"\ ", r"[\s-]+")
    up = unit_pattern(unit)
    if up:
        # name ... connector ... number ... unit   OR   number unit ... name
        pat = rf"(?:{name_pat})\s*(?:\(.*?\))?\s*(?:[:=]|of|was|is|set to|at)?\s*{NUM}\s*{up}"
    else:
        pat = rf"(?:{name_pat})\s*(?:[:=]|of|was|is|set to|at)?\s*{NUM}"
    return re.compile(pat, re.IGNORECASE)

PATTERNS = {name: build_pattern(name, info["unit"]) for name, info in flat.items()}

def extract(text):
    hits = {}
    for name, pat in PATTERNS.items():
        for m in pat.finditer(text):
            hits.setdefault(name, []).append({
                "value": m.group(1),
                "unit": flat[name]["unit"],
                "context": text[max(0, m.start()-25):m.end()+15].strip(),
                "section": flat[name]["sections"][0],
            })
    return hits

def save_results(results, out_path="/home/claude/extraction_results.json"):
    """Flattens results into a list of records and writes JSON."""
    records = []
    for name, occ in results.items():
        for o in occ:
            records.append({
                "parameter": name,
                "value": o["value"],
                "unit": o["unit"],
                "section": o["section"],
                "context": o["context"],
            })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return out_path

if __name__ == "__main__":
    reader = PdfReader("papers/s41467-019-10009-2.pdf")

    sample = ""
    for page in reader.pages:
        sample += page.extract_text() + "\n"

    print(sample)
    out_path = "extraction_results.json"

    results = extract(sample)
    for name, occ in results.items():
        for o in occ:
            print(f"{name:20s} = {o['value']:>6} {o['unit'] or '':6s}  |  ...{o['context']}...")

    saved_to = save_results(results, out_path)
    total = sum(len(v) for v in results.values())
    print(f"\nSaved {total} extracted values ({len(results)} unique parameters) to {saved_to}")