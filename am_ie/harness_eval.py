"""Compare a completed harness run with a candidate or reviewed reference."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from . import triage_pipeline


def _value_in_text(value, text):
    """Literal value match with only terminal punctuation ignored as fallback."""
    value = _norm_text(value)
    text = _norm_text(text)
    if not value:
        return False
    return value in text or bool(value.rstrip(".,;:!?…") and value.rstrip(".,;:!?…") in text)


def _quote_in_paper(quote, paper_text, ngram=6):
    if not quote or not paper_text:
        return False
    q = re.sub(r"\s+", " ", quote.strip().lower())
    p = re.sub(r"\s+", " ", paper_text.lower())
    if q in p:
        return True
    q_words = q.split()
    if len(q_words) < ngram:
        return q in p
    return any(" ".join(q_words[i:i + ngram]) in p
               for i in range(len(q_words) - ngram + 1))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_text(value):
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    value = value.replace("μ", "u").replace("µ", "u")
    value = re.sub(r"\bu\s+m\b", "um", value)
    value = re.sub(r"\s+", " ", value).strip().rstrip(".,;:!?…")
    return value.casefold()


def _canonical_parameter(parameter, vocabulary):
    if parameter in vocabulary:
        return parameter
    for key, item in vocabulary.items():
        if item.get("raw") == parameter:
            return key
    return parameter


def _numbers(value):
    normalized = _norm_text(value)
    normalized = re.sub(r"(?<=\d)(?:\s*[a-z°/%·]+)?-\s*(?=\d)", " ", normalized)
    return re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", normalized)


def _quantity(value, unit, parameter=""):
    value_norm, unit_norm = _norm_text(value), _norm_text(unit)
    numbers = _numbers(value)
    known_units = (
        "vol.-%", "vol-%", "k/h", "mm/sec", "mm/min", "mm/s", "mm2", "mm3",
        "g/cm3", "wt%", "khz", "mpa", "kpa", "gpa", "kn", "kw",
        "um", "nm", "mm", "cm", "°c", "m/s", "m/min", "w", "k",
        "n", "hz", "pa", "s", "min", "hr", "h", "pL", "%",
    )
    unit_key = unit_norm or next(
        (candidate for candidate in known_units if value_norm.endswith(candidate)), "",
    )
    if parameter.startswith("Projector Resolution") and unit_key == "um/px":
        # The reference may preserve the paper's “µm pixel size” unit while
        # the extractor expresses the taxonomy's equivalent “µm/px”.
        unit_key = "um"
    if numbers and unit_key:
        return f"{','.join(numbers)}|{unit_key}"
    return _norm_text(f"{value_norm} {unit_norm}")


def _row_key(row, vocabulary):
    return (
        _canonical_parameter(row.get("parameter", ""), vocabulary),
        _quantity(row.get("value", ""), row.get("unit", ""), row.get("parameter", "")),
        _norm_text(row.get("material", "")),
    )


def _content_key(row, vocabulary):
    return _row_key(row, vocabulary)[:2]


def _extracted_rows(extraction, statuses=("verified", "flagged")):
    rows = []
    for skill_result in extraction.values():
        params = skill_result.get("parameters", {})
        for status in statuses:
            for row in params.get(status, []):
                rows.append({**row, "harness_status": status})
    return rows


def _material_names(extraction):
    names = set()
    for skill_result in extraction.values():
        for status in ("verified", "flagged"):
            names.update(
                _norm_text(item.get("material", ""))
                for item in skill_result.get("materials", {}).get(status, [])
                if item.get("material")
            )
    return names


def _value_components_supported(row, quote):
    value = str(row.get("value", "") or "")
    unit = str(row.get("unit", "") or "")
    if _value_in_text(value, quote):
        return True
    normalized_quote = _norm_text(quote)
    numbers = _numbers(value)
    if not numbers or any(number not in normalized_quote for number in numbers):
        return False
    return not unit or _norm_text(unit) in normalized_quote


def _paper_text(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    sections = triage_pipeline.strip_markup_sectioned(raw)
    text = "\n\n".join(sections.values())
    tables = getattr(sections, "tables", [])
    return text + "\n\n" + json.dumps(tables, ensure_ascii=False)


def compare_run(run_dir, reference_path, vocabulary_path="vocab_flat.json",
                reference_status="candidate_unreviewed"):
    run_dir = Path(run_dir).resolve()
    reference_path = Path(reference_path).resolve()
    manifest = _read_json(run_dir / "manifest.json")
    extraction = _read_json(run_dir / "extraction.json")
    reference = _read_json(reference_path)
    vocabulary = _read_json(Path(vocabulary_path).resolve())
    tree_path = Path("vocab_tree.json").resolve()
    taxonomy_names = set(vocabulary)
    if tree_path.is_file():
        tree = _read_json(tree_path)
        for category in tree.values():
            for params in category.get("params", {}).values():
                if isinstance(params, dict):
                    for nested in params.values():
                        taxonomy_names.update(nested)
                else:
                    taxonomy_names.update(params)
    paper = Path(manifest["inputs"]["paper"]["path"])
    paper_text = _paper_text(paper)

    predicted = _extracted_rows(extraction, statuses=("verified",))
    flagged = _extracted_rows(extraction, statuses=("flagged",))
    expected = reference.get("extracted_parameters", [])
    pred_index = {}
    exp_index = {}
    pred_content_index = {}
    exp_content_index = {}
    for row in predicted:
        pred_index.setdefault(_row_key(row, vocabulary), []).append(row)
        pred_content_index.setdefault(_content_key(row, vocabulary), []).append(row)
    for row in expected:
        exp_index.setdefault(_row_key(row, vocabulary), []).append(row)
        exp_content_index.setdefault(_content_key(row, vocabulary), []).append(row)

    matched_keys = sorted(set(pred_index) & set(exp_index))
    expected_only_keys = sorted(set(exp_content_index) - set(pred_content_index))
    predicted_only_keys = sorted(set(pred_content_index) - set(exp_content_index))
    matched_count = sum(min(len(pred_index[k]), len(exp_index[k])) for k in matched_keys)
    matched_content_keys = sorted(set(pred_content_index) & set(exp_content_index))
    matched_content_count = sum(
        min(len(pred_content_index[k]), len(exp_content_index[k]))
        for k in matched_content_keys
    )

    def display(key, row):
        return {
            "parameter": key[0], "value_and_unit": key[1], "material": row.get("material", ""),
            "source_status": row.get("harness_status", "reference"),
            "quote": row.get("quote", ""),
            "quote_in_paper": _quote_in_paper(row.get("quote", ""), paper_text),
            "value_in_quote": _value_in_text(row.get("value", ""), row.get("quote", "")),
            "value_components_supported": _value_components_supported(row, row.get("quote", "")),
            "flag_reason": row.get("flag_reason", ""),
        }

    reference_audit = []
    for row in expected:
        reference_audit.append({
            "parameter": row.get("parameter", ""),
            "value": row.get("value", ""),
            "unit": row.get("unit", ""),
            "material": row.get("material", ""),
            "parameter_in_taxonomy": (
                _canonical_parameter(row.get("parameter", ""), vocabulary) in vocabulary
                or row.get("parameter", "") in taxonomy_names
            ),
            "value_in_quote": _value_in_text(row.get("value", ""), row.get("quote", "")),
            "value_components_supported": _value_components_supported(row, row.get("quote", "")),
            "quote_in_paper": _quote_in_paper(row.get("quote", ""), paper_text),
            "quote": row.get("quote", ""),
        })

    expected_materials = {_norm_text(x.get("material", "")) for x in reference.get("materials", []) if x.get("material")}
    predicted_materials = _material_names(extraction)
    reference_count = len(expected)
    prediction_count = len(predicted)
    metrics = None
    if reference_status == "human_reviewed":
        precision = matched_count / prediction_count if prediction_count else (1.0 if not reference_count else 0.0)
        recall = matched_count / reference_count if reference_count else (1.0 if not prediction_count else 0.0)
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        }
    report = {
        "run_id": manifest.get("run_id"),
        "paper": str(paper),
        "reference": str(reference_path),
        "reference_status": reference_status,
        "accuracy_score_claimed": reference_status == "human_reviewed",
        "metrics": metrics,
        "row_counts": {
            "reference": reference_count, "predicted": prediction_count,
            "flagged_predictions": len(flagged),
            "content_matched_ignoring_material": matched_content_count,
            "fully_matched_including_material": matched_count,
            "reference_only": sum(len(exp_content_index[k]) for k in expected_only_keys),
            "prediction_only": sum(len(pred_content_index[k]) for k in predicted_only_keys),
            "reference_only_including_material": sum(len(rows) for k, rows in exp_index.items() if k not in pred_index),
            "prediction_only_including_material": sum(len(rows) for k, rows in pred_index.items() if k not in exp_index),
        },
        "matched_rows": [display(k, pred_index[k][0]) for k in matched_keys],
        "flagged_predictions": [display(_row_key(row, vocabulary), row) for row in flagged],
        "content_matched_rows": [
            {"reference": display(_row_key(exp_content_index[k][0], vocabulary), exp_content_index[k][0]),
             "prediction": display(_row_key(pred_content_index[k][0], vocabulary), pred_content_index[k][0]),
             "material_matches": _norm_text(exp_content_index[k][0].get("material", "")) ==
                                 _norm_text(pred_content_index[k][0].get("material", ""))}
            for k in matched_content_keys
        ],
        "reference_only_rows": [display(k, exp_content_index[k][0]) for k in expected_only_keys],
        "prediction_only_rows": [display(k, pred_content_index[k][0]) for k in predicted_only_keys],
        "material_attribution_mismatches": [
            {"parameter": k[0], "value_and_unit": k[1],
             "reference_materials": sorted({r.get("material", "") for r in exp_content_index[k]}),
             "predicted_materials": sorted({r.get("material", "") for r in pred_content_index[k]})}
            for k in matched_content_keys
            if {_norm_text(r.get("material", "")) for r in exp_content_index[k]} !=
               {_norm_text(r.get("material", "")) for r in pred_content_index[k]}
        ],
        "material_name_diff": {
            "reference_only": sorted(expected_materials - predicted_materials),
            "prediction_only": sorted(predicted_materials - expected_materials),
        },
        "reference_row_audit": reference_audit,
        "note": "Reference files are drafts until a person adjudicates completeness, parameter mapping, units, and material attribution.",
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare a harness run with a candidate or reviewed reference JSON.")
    parser.add_argument("run_dir", help="Completed harness run directory")
    parser.add_argument("--reference", required=True, help="Reference JSON in the server.json shape")
    parser.add_argument("--vocab", default="vocab_flat.json")
    parser.add_argument("--reference-status", choices=("candidate_unreviewed", "human_reviewed_subset", "human_reviewed"),
                        default="candidate_unreviewed")
    parser.add_argument("--output", help="Report path (default: <run_dir>/comparison.json)")
    args = parser.parse_args(argv)
    report = compare_run(args.run_dir, args.reference, args.vocab, args.reference_status)
    out = Path(args.output) if args.output else Path(args.run_dir) / "comparison.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Comparison report: {out}")
    print(json.dumps(report["row_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
