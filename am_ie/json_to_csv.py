"""
Converts agentic_extraction_results.json (output of agentic_pipeline.py) into
a flat CSV — one row per extracted parameter, easy to open in Excel/pandas.

Usage:
    python3 json_to_csv.py agentic_extraction_results.json output.csv
    python3 json_to_csv.py agentic_extraction_results.json          # writes extraction_results.csv

Also writes a second CSV for materials, alongside the parameters one
(e.g. output_materials.csv), since they're a different shape of row.
"""
import json
import csv
import sys
import os


PARAM_FIELDS = [
    "skill_used", "parameter", "value", "unit", "material",
    "status", "quote", "flag_reason",
]

MATERIAL_FIELDS = [
    "skill_used", "material", "material_form", "status", "quote",
]


def flatten_parameters(results):
    """results: the dict loaded from agentic_extraction_results.json.
    Yields one flat dict per parameter row, tagging verified/flagged as 'status'."""
    for category_code, block in results.items():
        params = block.get("parameters", {})
        for status in ("verified", "flagged"):
            for row in params.get(status, []):
                yield {
                    "skill_used": row.get("skill_used", category_code),
                    "parameter": row.get("parameter", ""),
                    "value": row.get("value", ""),
                    "unit": row.get("unit", ""),
                    "material": row.get("material", ""),
                    "status": status,
                    "quote": row.get("quote", ""),
                    "flag_reason": row.get("flag_reason", ""),
                }


def flatten_materials(results):
    """Same idea, for the materials list."""
    for category_code, block in results.items():
        materials = block.get("materials", {})
        for status in ("verified", "flagged"):
            for row in materials.get(status, []):
                yield {
                    "skill_used": category_code,
                    "material": row.get("material", ""),
                    "material_form": row.get("material_form", ""),
                    "status": status,
                    "quote": row.get("quote", ""),
                }


def write_csv(rows, fields, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        count = 0
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def convert(json_path, params_csv_path=None, materials_csv_path=None):
    results = json.load(open(json_path, encoding="utf-8"))

    if params_csv_path is None:
        base, _ = os.path.splitext(json_path)
        params_csv_path = base + ".csv"
    if materials_csv_path is None:
        base, _ = os.path.splitext(params_csv_path)
        materials_csv_path = base + "_materials.csv"

    n_params = write_csv(flatten_parameters(results), PARAM_FIELDS, params_csv_path)
    n_materials = write_csv(flatten_materials(results), MATERIAL_FIELDS, materials_csv_path)

    print(f"Wrote {n_params} parameter rows -> {params_csv_path}")
    print(f"Wrote {n_materials} material rows -> {materials_csv_path}")
    return params_csv_path, materials_csv_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 json_to_csv.py agentic_extraction_results.json [output.csv]")
        sys.exit(1)

    json_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert(json_path, out_path)
