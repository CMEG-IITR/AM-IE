"""
compare_vs_ground_truth.py -- lean pipeline output ("ours") vs human-curated
extraction ("actual"), across a directory of paper folders.

Expects a layout like the uploaded ran.zip:
    <root>/<paper_id>/server.json                 <- ground truth (actual)
    <root>/<paper_id>/agentic_lean_results.json    <- lean pipeline output (ours)
    <root>/<paper_id>/agentic_lean_stats.json      <- optional, for cost/time

server.json shape:  {"materials": [...], "extracted_parameters": [
    {"parameter": "...", "value": "...", "unit": "...", "material": "...", "quote": "..."}, ...]}
agentic_lean_results.json shape: {"<skill>[::<material>]": {"parameters": {"verified": [...], "flagged": [...]}, ...}}

Ground-truth parameter names have no unit suffix ("Machine Make/Model"); lean's
taxonomy names do ("Laser Power (W)"). Matching therefore strips any trailing
"(...)" before comparing names, then compares values after normalising
whitespace, case, ×/x, and micro-sign variants. Material is used as a tiebreaker,
not a hard filter, since the unpinned (review / no-printed-materials) lean path
can leave material blank or paraphrased.

Usage:
    python compare_vs_ground_truth.py /path/to/ran/ran
    python compare_vs_ground_truth.py /path/to/ran/ran --show-misses 20 --show-extra 10
"""
import argparse
import json
import re
import sys
from pathlib import Path


# ---------- normalisation (mirrors chunk_retrieval / agentic_pipeline's own canon rules) ----------
def norm_param(name):
    """Strip a trailing '(...)' unit annotation and lowercase, so lean's
    'Laser Power (W)' lines up with ground truth's 'Laser Power'."""
    bare = re.sub(r"\s*\([^)]*\)\s*$", "", str(name or "").strip())
    return re.sub(r"\s+", " ", bare).lower()


_MICRO = re.compile("[\u00b5\u03bcu](?=m\\b)")


def norm_value(v):
    v = str(v or "").strip().lower()
    v = v.replace("\u00d7", "x").replace("\u2212", "-")
    v = _MICRO.sub("\u00b5", v)                 # u/µ/μ before 'm' all collapse to µ
    v = re.sub(r"\s+", " ", v)
    v = re.sub(r"(?<=\d),(?=\d{3}\b)", "", v)   # '1,100' -> '1100'
    return v.strip()


def norm_material(m):
    return re.sub(r"\s+", " ", str(m or "").strip().lower())


def numeric_equal(a, b, rel_tol=0.02):
    """Fallback when the strings differ only in formatting ('353' vs '353.0')."""
    try:
        fa, fb = float(re.sub(r"[^\d.\-]", "", a)), float(re.sub(r"[^\d.\-]", "", b))
    except ValueError:
        return False
    if fa == fb:
        return True
    return abs(fa - fb) <= rel_tol * max(abs(fa), abs(fb), 1e-9)


def values_match(a, b):
    na, nb = norm_value(a), norm_value(b)
    return na == nb or numeric_equal(na, nb)


# ---------- loading ----------
def load_gold(path):
    data = json.load(open(path, encoding="utf-8"))
    rows = []
    for p in data.get("extracted_parameters", []):
        rows.append({"parameter": p.get("parameter", ""), "value": p.get("value", ""),
                    "unit": p.get("unit", ""), "material": p.get("material", ""),
                    "quote": p.get("quote", "")})
    return rows, [m.get("material", "") for m in data.get("materials", [])]


def load_ours(path, include_flagged=False):
    data = json.load(open(path, encoding="utf-8"))
    rows = []
    for key, r in data.items():
        bucket = r.get("parameters", {})
        for row in bucket.get("verified", []):
            rows.append(dict(row, _skill_key=key, _status="verified"))
        if include_flagged:
            for row in bucket.get("flagged", []):
                rows.append(dict(row, _skill_key=key, _status="flagged"))
    return rows


# ---------- matching: greedy 1-to-1, parameter+value required, material as tiebreak ----------
def match(gold_rows, our_rows):
    """Returns (matched [(gold, ours)], missed [gold rows], extra [our rows]).
    Each gold row consumes at most one our-row and vice versa (1-to-1), so
    duplicate values don't inflate the match count."""
    gold_idx = [dict(g, _np=norm_param(g["parameter"]), _nm=norm_material(g["material"])) for g in gold_rows]
    our_idx = [dict(o, _np=norm_param(o.get("parameter")), _nm=norm_material(o.get("material")))
               for o in our_rows]
    used_our = [False] * len(our_idx)
    matched, missed = [], []

    for g in gold_idx:
        best = None
        for i, o in enumerate(our_idx):
            if used_our[i] or o["_np"] != g["_np"] or not values_match(g["value"], o.get("value")):
                continue
            score = 2 if (g["_nm"] and g["_nm"] == o["_nm"]) else (1 if not g["_nm"] or not o["_nm"] else 0)
            if best is None or score > best[0]:
                best = (score, i)
        if best is not None:
            used_our[best[1]] = True
            matched.append((g, our_idx[best[1]]))
        else:
            missed.append(g)
    extra = [o for i, o in enumerate(our_idx) if not used_our[i]]
    return matched, missed, extra


# ---------- per-paper + aggregate ----------
def run(root, include_flagged, show_misses, show_extra):
    root = Path(root)
    papers = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    rows_out = []
    tot = dict(gold=0, matched=0, missed=0, extra=0)
    tot_cost = dict(calls=0, input_tokens=0, output_tokens=0, reasoning_tokens=0, seconds=0.0)
    skipped = []

    for p in papers:
        gold_path, our_path, stats_path = p / "server.json", p / "agentic_lean_results.json", p / "agentic_lean_stats.json"
        if not our_path.exists():
            skipped.append((p.name, "no lean output (not run / errored before writing results)"))
            continue
        if not gold_path.exists() or gold_path.stat().st_size == 0:
            skipped.append((p.name, "no ground truth (server.json missing or empty)"))
            continue
        try:
            gold_rows, gold_materials = load_gold(gold_path)
        except json.JSONDecodeError as e:
            skipped.append((p.name, f"server.json unreadable: {e}"))
            continue
        our_rows = load_ours(our_path, include_flagged)
        matched, missed, extra = match(gold_rows, our_rows)

        cost = {}
        if stats_path.exists():
            s = json.load(open(stats_path))
            cost = dict(s.get("usage", {}))
            cost["wall_seconds"] = s.get("wall_seconds", 0)
            for k in ("calls", "input_tokens", "output_tokens", "reasoning_tokens"):
                tot_cost[k] += cost.get(k, 0)
            tot_cost["seconds"] += cost.get("seconds", 0)

        n_gold = len(gold_rows)
        recall = len(matched) / n_gold if n_gold else float("nan")
        n_our = len(matched) + len(extra)
        precision = len(matched) / n_our if n_our else float("nan")
        rows_out.append(dict(paper=p.name, gold=n_gold, matched=len(matched), missed=len(missed),
                             extra=len(extra), recall=recall, precision=precision,
                             missed_rows=missed, extra_rows=extra, cost=cost))
        tot["gold"] += n_gold
        tot["matched"] += len(matched)
        tot["missed"] += len(missed)
        tot["extra"] += len(extra)

    # ---- table ----
    print(f"{'paper':<40}{'gold':>5}{'match':>6}{'miss':>5}{'extra':>6}{'recall':>8}{'precis':>8}{'calls':>7}{'in_tok':>8}{'out_tok':>8}{'sec':>7}")
    for r in rows_out:
        c = r["cost"]
        print(f"{r['paper']:<40}{r['gold']:>5}{r['matched']:>6}{r['missed']:>5}{r['extra']:>6}"
              f"{r['recall']*100:>7.0f}%{r['precision']*100:>7.0f}%"
              f"{c.get('calls',0):>7}{c.get('input_tokens',0):>8}{c.get('output_tokens',0):>8}{c.get('seconds',0):>7.1f}")

    n = len(rows_out)
    macro_recall = sum(r["recall"] for r in rows_out if r["gold"]) / max(1, sum(1 for r in rows_out if r["gold"]))
    macro_prec = sum(r["precision"] for r in rows_out if r["matched"] + r["extra"]) / max(1, sum(1 for r in rows_out if r["matched"] + r["extra"]))
    micro_recall = tot["matched"] / tot["gold"] if tot["gold"] else float("nan")
    micro_prec = tot["matched"] / (tot["matched"] + tot["extra"]) if (tot["matched"] + tot["extra"]) else float("nan")
    print("-" * 110)
    print(f"{'TOTAL (' + str(n) + ' papers)':<40}{tot['gold']:>5}{tot['matched']:>6}{tot['missed']:>5}{tot['extra']:>6}"
          f"{micro_recall*100:>7.0f}%{micro_prec*100:>7.0f}%"
          f"{tot_cost['calls']:>7}{tot_cost['input_tokens']:>8}{tot_cost['output_tokens']:>8}{tot_cost['seconds']:>7.1f}")
    print(f"\nmicro recall/precision pool every row together; macro averages per-paper rates "
          f"(macro recall {macro_recall*100:.0f}%, macro precision {macro_prec*100:.0f}%).")
    print("(recall = gold rows we matched; precision = our rows that matched a gold row. "
          "'extra' is NOT necessarily wrong -- ground truth may simply not list it; see --show-extra.)")

    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for name, why in skipped:
            print(f"  {name}: {why}")

    if show_misses:
        print(f"\n=== missed gold rows (up to {show_misses} per paper) ===")
        for r in rows_out:
            if not r["missed_rows"]:
                continue
            print(f"\n{r['paper']}:")
            for g in r["missed_rows"][:show_misses]:
                print(f"  - {g['parameter']} = {g['value']} {g.get('unit','')} [{g.get('material','')}] "
                      f"| \"{(g.get('quote') or '')[:90]}\"")

    if show_extra:
        print(f"\n=== rows we produced that didn't match ground truth (up to {show_extra} per paper) ===")
        for r in rows_out:
            if not r["extra_rows"]:
                continue
            print(f"\n{r['paper']}:")
            for o in r["extra_rows"][:show_extra]:
                flag = f" [{o.get('_status')}]" if o.get("_status") == "flagged" else ""
                print(f"  - {o.get('parameter')} = {o.get('value')} {o.get('unit','')} "
                      f"[{o.get('material','')}]{flag} | \"{(o.get('quote') or '')[:90]}\"")

    json.dump({"per_paper": [{k: v for k, v in r.items() if k not in ("missed_rows", "extra_rows")}
                             for r in rows_out],
              "totals": tot, "cost_totals": tot_cost, "skipped": skipped},
             open("ground_truth_comparison.json", "w"), indent=2, ensure_ascii=False)
    print("\nFull report written to ground_truth_comparison.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory containing one subfolder per paper")
    ap.add_argument("--include-flagged", action="store_true",
                    help="count lean's flagged rows too, not just verified (loosens 'ours')")
    ap.add_argument("--show-misses", type=int, default=0, help="print up to N missed gold rows per paper")
    ap.add_argument("--show-extra", type=int, default=0, help="print up to N unmatched our-rows per paper")
    args = ap.parse_args()
    run(args.root, args.include_flagged, args.show_misses, args.show_extra)


if __name__ == "__main__":
    main()
