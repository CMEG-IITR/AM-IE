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
Detailed extraction runs on GPT-5-mini via call_gpt_mini(). (Triage stays on
local Ollama llama3.1 — see triage_pipeline.py's own call_llm().)
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


# ---------- 3. Call the LLM (OpenAI, GPT-5-mini) ----------
# Detailed extraction now runs on GPT-5-mini instead of local Ollama. Triage
# (triage_pipeline.py) keeps its own separate call_llm() targeting Ollama
# llama3.1 — that one is untouched and lives in that file, not here.
import os
from openai import OpenAI


def call_gpt_mini(prompt: str, model: str = "gpt-5-mini") -> str:
    """
    Calls the OpenAI API using the GPT-5-mini model.
    Requires the OPENAI_API_KEY environment variable to be set (e.g. via
    `export OPENAI_API_KEY=...` or a .env loader) — this no longer depends
    on Colab's userdata, so it works the same in a notebook or a plain script.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            # To enforce your JSON structure, add a 'text' parameter:
            # text={"format": {"type": "json_schema", "name": "extraction", "schema": YOUR_SCHEMA}},
        )
        return response.output_text

    except Exception as e:
        # The SDK automatically retries rate limits and connection errors
        print(f"An API error occurred: {e}")
        raise


# ---------- 4. Parse response ----------
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


# ---------- 5. Verify: value/material must actually appear in quote ----------
def _quote_in_paper(quote, paper_text, ngram=6):
    """Check that the quote (or a meaningful chunk of it) actually appears in the
    paper. Uses a sliding n-gram of words, requiring at least one contiguous
    run of `ngram` words to match. More robust than a first-N-chars prefix."""
    if not quote or not paper_text:
        return False
    q = re.sub(r"\s+", " ", quote.strip().lower())
    p = re.sub(r"\s+", " ", paper_text.lower())
    if q in p:
        return True
    q_words = q.split()
    if len(q_words) < ngram:
        return q in p
    for i in range(len(q_words) - ngram + 1):
        chunk = " ".join(q_words[i:i + ngram])
        if chunk in p:
            return True
    return False

def verify_materials(materials, paper_text):
    verified, flagged = [], []
    for row in materials:
        name = str(row.get("material", ""))
        quote = row.get("quote", "")
        name_in_quote = name.lower() in quote.lower() if name else False
        row["verified"] = name_in_quote and _quote_in_paper(quote, paper_text)
        (verified if row["verified"] else flagged).append(row)
    return verified, flagged

def verify_parameters(extracted_parameters, paper_text, known_materials,
                     source_kind="research-article"):
    verified, flagged = [], []
    for row in extracted_parameters:
        value = str(row.get("value", "") or "")
        quote = str(row.get("quote", "") or "")
        context = str(row.get("context", "") or "")
        material = str(row.get("material", "") or "")
        parameter = str(row.get("parameter", "") or "")
        provenance = row.get("provenance") or {}

        if not value.strip():
            row["verified"] = False
            row["flag_reason"] = "empty value"
            flagged.append(row)
            continue

        value_in_paper = value.lower() in paper_text.lower()
        value_in_quote = bool(quote.strip()) and value.lower() in quote.lower()
        value_in_context = bool(context.strip()) and value.lower() in context.lower()

        if source_kind == "review":
            # Review mode: table cells are legit. Accept if value appears in
            # the paper AND either (a) parameter name appears in the quote or
            # context, or (b) row is cited-secondary with an explicit source.
            param_nearby = (
                (bool(quote) and parameter.lower() in quote.lower()) or
                (bool(context) and parameter.lower() in context.lower())
            )
            cited = bool(provenance.get("cited_work") or provenance.get("citation_marker"))
            structural_ok = bool(quote.strip()) or bool(context.strip())
            row["verified"] = value_in_paper and structural_ok and (param_nearby or cited)
            if not row["verified"]:
                if not value_in_paper:
                    row["flag_reason"] = "value not found in paper"
                elif not structural_ok:
                    row["flag_reason"] = "no quote and no context"
                else:
                    row["flag_reason"] = "parameter name not near value and no citation"
        else:
            # Primary mode: strict — value must appear in quote, quote must
            # appear in paper, material (if tagged) must be confirmed.
            quote_ok = _quote_in_paper(quote, paper_text)
            material_ok = (not material) or (material in known_materials)
            row["verified"] = value_in_quote and quote_ok and material_ok
            if not row["verified"]:
                if not value_in_quote:
                    row["flag_reason"] = "value not in quote"
                elif not quote_ok:
                    row["flag_reason"] = "quote not in paper"
                else:
                    row["flag_reason"] = "material not confirmed"

        (verified if row["verified"] else flagged).append(row)
    return verified, flagged


# ---------- 6. Full pipeline ----------
def run_pipeline(paper_text, flat_json_path="vocab_flat.json", model="gpt-5-mini"):
    schema = load_schema(flat_json_path)
    prompt = build_prompt(schema, paper_text)
    raw = call_gpt_mini(prompt, model=model)
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
