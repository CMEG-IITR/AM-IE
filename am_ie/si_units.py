"""
Symbolic (non-AI) SI unit normalizer.

Runs as a post-processing pass over llm_pipeline's verified extraction rows
(each row already carries "value" and "unit" strings the GPT-5-mini
extraction returned). Parses the value (handling ranges, +/-, scientific
notation) and converts it to a canonical SI or SI-derived unit using a fixed
lookup table pulled from am_taxonomy_v4.html's own L4/L5/L8 unit tags.

No LLM call anywhere in this module — pure regex + arithmetic. A unit not
in the table is left unconverted (value_si/unit_si = None) rather than
guessed, so it's visible as "not normalized" instead of silently wrong.
"""
import re

# unit string (lowercased, whitespace-stripped) -> (si_unit, multiply_by, add_before_multiply)
# value_si = (value + add_before_multiply) * multiply_by
UNIT_TABLE = {
    # pressure / stress — UTS, Yield Strength, Flexural/Compressive Strength,
    # Residual Stress, Tensile Strength all report in MPa/GPa in the taxonomy
    "pa": ("Pa", 1, 0),
    "kpa": ("Pa", 1e3, 0),
    "mpa": ("Pa", 1e6, 0),
    "gpa": ("Pa", 1e9, 0),
    "mpa\u221am": ("Pa\u00b7\u221am", 1e6, 0),      # Fracture Toughness KIc (MPa\u221am)
    "mpasqrt(m)": ("Pa\u00b7\u221am", 1e6, 0),

    # length — D10/D50/D90, Grain/Defect Size, Layer Thickness, Cure Depth, Ra
    "m": ("m", 1, 0),
    "mm": ("m", 1e-3, 0),
    "cm": ("m", 1e-2, 0),
    "\u00b5m": ("m", 1e-6, 0),
    "um": ("m", 1e-6, 0),
    "nm": ("m", 1e-9, 0),

    # velocity — Print/Scan Speed
    "mm/s": ("m/s", 1e-3, 0),
    "m/s": ("m/s", 1, 0),

    # density — True/Apparent/Tap Density
    "g/cm3": ("kg/m3", 1e3, 0),
    "g/cm\u00b3": ("kg/m3", 1e3, 0),
    "kg/m3": ("kg/m3", 1, 0),
    "kg/m\u00b3": ("kg/m3", 1, 0),

    # temperature — Tg, Melting Point, Bed/Resin Temp, Pre-dry Temp (offset conversion)
    "\u00b0c": ("K", 1, 273.15),
    "c": ("K", 1, 273.15),
    "k": ("K", 1, 0),

    # thermal
    "w/m\u00b7k": ("W/(m\u00b7K)", 1, 0),
    "w/mk": ("W/(m\u00b7K)", 1, 0),
    "j/kg\u00b7k": ("J/(kg\u00b7K)", 1, 0),
    "\u00b5m/m\u00b7\u00b0c": ("1/K", 1e-6, 0),      # CTE
    "um/m*c": ("1/K", 1e-6, 0),

    # energy / impact — Impact Strength, Post-cure UV Dose
    "kj/m2": ("J/m2", 1e3, 0),
    "kj/m\u00b2": ("J/m2", 1e3, 0),
    "mj/cm2": ("J/m2", 10, 0),          # 1 mJ/cm^2 = 1e-3 J / 1e-4 m^2 = 10 J/m^2
    "mj/cm\u00b2": ("J/m2", 10, 0),

    # viscosity — resin Viscosity
    "mpa\u00b7s": ("Pa\u00b7s", 1e-3, 0),
    "mpa.s": ("Pa\u00b7s", 1e-3, 0),
    "cp": ("Pa\u00b7s", 1e-3, 0),        # centipoise

    # electrical
    "s/m": ("S/m", 1, 0),
    "\u03a9\u00b7m": ("\u03a9\u00b7m", 1, 0),
    "ohm\u00b7m": ("\u03a9\u00b7m", 1, 0),
    "kv/mm": ("V/m", 1e6, 0),

    # dimensionless / fractional — Relative Density, Porosity, Elongation, wt%
    "%": ("fraction", 1e-2, 0),
    "wt%": ("mass fraction", 1e-2, 0),
    "ppm": ("fraction", 1e-6, 0),
}


# A range written as "90-99" or "90\u201399": two bare (unsigned) numbers
# joined by a dash/en-dash/em-dash with no space before the dash, so it's
# never confused with a signed second number like "90, -99".
_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-\u2013\u2014]\s*(\d+(?:\.\d+)?)"
)
_RANGE_TO_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\bto\b\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)
_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*[eE][-+]?\d+)?")


def _extract_numeric(value_str):
    """Pulls one representative number from a value string. Handles simple
    ranges ('90-99', '90\u201399', '90 to 99') by returning the midpoint —
    checked BEFORE generic signed-number parsing, since a bare range dash
    would otherwise be misread as a minus sign on the second number. Strips
    a trailing '\u00b1 x' uncertainty term before parsing. Returns None if no
    number is found."""
    if value_str is None:
        return None
    s = str(value_str).strip()
    if not s:
        return None
    s = re.sub(r"\u00b1\s*[\d.]+", "", s)

    range_match = _RANGE_RE.search(s) or _RANGE_TO_RE.search(s)
    if range_match:
        lo, hi = float(range_match.group(1)), float(range_match.group(2))
        return (lo + hi) / 2

    nums = _NUMERIC_RE.findall(s)
    if not nums:
        return None
    return float(nums[0].replace(" ", ""))


def _normalize_unit_key(unit_str):
    if not unit_str:
        return None
    return re.sub(r"\s+", "", str(unit_str).strip().lower())


def normalize_value(value_str, unit_str):
    """Returns (value_si, unit_si), or (None, None) if the value couldn't be
    parsed or the unit isn't in UNIT_TABLE. Pure symbolic — no LLM call."""
    numeric = _extract_numeric(value_str)
    unit_key = _normalize_unit_key(unit_str)
    if numeric is None or unit_key not in UNIT_TABLE:
        return None, None
    si_unit, multiply_by, add_before = UNIT_TABLE[unit_key]
    value_si = (numeric + add_before) * multiply_by
    return round(value_si, 10), si_unit


def normalize_row(row):
    """Adds 'value_si' / 'unit_si' to an extracted parameter row (mutates
    and returns it). Original 'value'/'unit' are left untouched. Both are
    set to None when the unit isn't recognized — flagged, never guessed."""
    value_si, unit_si = normalize_value(row.get("value"), row.get("unit"))
    row["value_si"] = value_si
    row["unit_si"] = unit_si
    return row


def normalize_rows(rows):
    return [normalize_row(dict(r)) for r in rows]


if __name__ == "__main__":
    import json
    import sys

    rows = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else json.load(sys.stdin)
    print(json.dumps(normalize_rows(rows), indent=2))
