"""
Basic LLM extraction pipeline, starting point.

Flow:
  1. Load taxonomy schema (flat param list) + paper text
  2. Build the constrained prompt (schema + paper + strict JSON output format)
  3. Call the LLM
  4. Parse JSON response
  5. VERIFY: for every extracted row, check that "value" actually appears
     in "quote" (word-for-word). Drop/flag anything that fails.
  6. Save verified + flagged results separately.

This is deliberately minimal — one call per paper, one verification pass.
Swap in whatever LLM client you have (Anthropic, OpenAI, local model server
with an OpenAI-compatible API, etc.) in call_llm().
"""
import json
import re

# ---------- 1. Load taxonomy ----------
def load_schema(flat_json_path="vocab_flat.json"):
    flat = json.load(open(flat_json_path))
    lines = []
    for name, info in flat.items():
        unit = info["unit"] or ""
        lines.append(f"- {name} ({unit})" if unit else f"- {name}")
    return "\n".join(lines)


# ---------- 2. Build prompt ----------
PROMPT_TEMPLATE = """You are a data extraction assistant for additive manufacturing (AM) research papers.

Extract every value from the paper below that matches a parameter in the taxonomy. Return ONLY valid JSON, no prose, in this shape:

{{
  "materials": [
    {{
      "material": "<exact material/alloy name as reported, e.g. Ti-6Al-4V>",
      "material_form": "<powder | wire | filament | resin | sheet | other, if stated>",
      "quote": "<short verbatim sentence naming this material>"
    }}
  ],
  "extracted_parameters": [
    {{
      "parameter": "<exact taxonomy name>",
      "value": "<value as reported>",
      "unit": "<unit as reported>",
      "material": "<which material from the materials list this parameter/value applies to, if the paper ties them together; else empty string>",
      "quote": "<short verbatim sentence from the paper supporting it>"
    }}
  ]
}}

Rules:
- Only use parameter names from the taxonomy below.
- If a paper studies more than one material (e.g. comparing two alloys or powders), list each separately in "materials", and tag every extracted parameter with the material it belongs to. If a parameter clearly applies to all materials studied or the paper only uses one material, still fill "material" with that name.
- If a parameter's material association is genuinely unclear from context, leave "material" as an empty string rather than guessing.
- The "quote" must be an exact sentence/clause copied from the paper text, containing the value (or material name, for materials list entries).
- Do not include parameters the paper doesn't report.
- Do not fabricate values or materials.

TAXONOMY:
{schema}

PAPER:
{paper_text}
"""

def build_prompt(schema, paper_text):
    return PROMPT_TEMPLATE.format(schema=schema, paper_text=paper_text)


# ---------- 3. Call the LLM (Ollama, local, free) ----------
def call_llm(prompt, model="llama3.1", host="http://localhost:11434"):
    """
    Uses Ollama running locally — free, no API key, no internet call needed
    once the model is pulled.

    Setup (one-time):
        1. Install: https://ollama.com/download
        2. Pull a model:  ollama pull llama3.1        (~4.7GB, general purpose)
                       or  ollama pull mistral         (~4.1GB, faster/smaller)
                       or  ollama pull qwen2.5:7b       (good at following JSON format)
        3. Ollama runs a local server automatically at localhost:11434

    Swap `model=` to whichever you pulled.
    """
    import urllib.request

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # ask Ollama to constrain output to valid JSON
        "options": {"temperature": 0},  # deterministic, less prone to drifting off-schema
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["response"]


# ---------- 4. Parse response ----------
def parse_response(raw_text):
    # strip markdown code fences if present
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
    return json.loads(cleaned)


# ---------- 5. Verify: value/material must actually appear in quote ----------
def _quote_in_paper(quote, paper_text):
    return bool(quote) and quote.strip()[:30].lower() in paper_text.lower()

def verify_materials(materials, paper_text):
    verified, flagged = [], []
    for row in materials:
        name = str(row.get("material", ""))
        quote = row.get("quote", "")
        name_in_quote = name.lower() in quote.lower() if name else False
        row["verified"] = name_in_quote and _quote_in_paper(quote, paper_text)
        (verified if row["verified"] else flagged).append(row)
    return verified, flagged

def verify_parameters(extracted_parameters, paper_text, known_materials):
    verified, flagged = [], []
    for row in extracted_parameters:
        value = str(row.get("value", ""))
        quote = row.get("quote", "")
        material = row.get("material", "")

        value_in_quote = value.lower() in quote.lower()
        quote_ok = _quote_in_paper(quote, paper_text)
        # material tag, if given, must be one we actually verified in the materials list
        material_ok = (not material) or (material in known_materials)

        row["verified"] = value_in_quote and quote_ok and material_ok
        if material and not material_ok:
            row["flag_reason"] = "material not confirmed in materials list"
        (verified if row["verified"] else flagged).append(row)
    return verified, flagged


# ---------- 6. Full pipeline ----------
def run_pipeline(paper_text, flat_json_path="vocab_flat.json"):
    schema = load_schema(flat_json_path)
    prompt = build_prompt(schema, paper_text)
    raw = call_llm(prompt)
    parsed = parse_response(raw)

    mat_verified, mat_flagged = verify_materials(parsed.get("materials", []), paper_text)
    known_materials = {m["material"] for m in mat_verified}

    param_verified, param_flagged = verify_parameters(
        parsed.get("extracted_parameters", []), paper_text, known_materials
    )

    result = {
        "materials": {"verified": mat_verified, "flagged": mat_flagged},
        "parameters": {"verified": param_verified, "flagged": param_flagged},
    }
    json.dump(result, open("llm_extraction_verified.json", "w"), indent=2)
    print(f"Materials  -> verified: {len(mat_verified)}  flagged: {len(mat_flagged)}")
    print(f"Parameters -> verified: {len(param_verified)}  flagged: {len(param_flagged)}")
    return result


if __name__ == "__main__":
    sample_paper = """
    The LPBF process was run with laser power of 195 W and a scan speed
    of 1100 mm/s. The powder had a D50 of 32 microns.
    """
    run_pipeline(sample_paper)
