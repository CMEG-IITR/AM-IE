"""
ab_compare_tiers.py -- isolate whether max_chunks, reasoning_effort, or both are
what's driving the recall improvement on large skills.

Runs run_lean() four times per paper, varying ONLY the size-tier scaling:
  baseline     - no scaling at all (your original flat defaults: max_chunks=40, low)
  chunks_only  - bigger skills get more chunks, effort stays at the base value
  effort_only  - bigger skills get higher effort, max_chunks stays at the base value
  both         - the DEFAULT_SIZE_TIERS shipped in agentic_pipeline.py (chunks + effort)

Each run's verified rows are matched against server.json (ground truth) using the
same logic as compare_vs_ground_truth.py, so you get one row per (paper, variant)
with recall/precision/cost -- cheap to scan for "did effort actually help, or was
it just the extra chunks doing the work".

COST WARNING: this calls the real API 4x per paper (once per variant). Run it on
one or two papers first, not the whole batch, especially if you've been rate limited.

Usage:
  python ab_compare_tiers.py paper.xml vocab_tree.json --gold server.json
  python ab_compare_tiers.py paper.xml vocab_tree.json --gold server.json --triage-json triage_result.json
  python ab_compare_tiers.py --papers-root ran/ --limit 2          # batch mode, mirrors ran.zip layout
"""
import argparse
import json
import sys
import time
from pathlib import Path

from agentic_pipeline import (build_skill_library, build_subtype_skill_library, run_lean,
                              DEFAULT_SIZE_TIERS)
from compare_vs_ground_truth import load_gold, load_ours, match
from triage_pipeline import strip_markup_sectioned, run_triage

# Same thresholds as DEFAULT_SIZE_TIERS (40 / 100 / big), isolating one knob at a time.
# (cap, max_chunks, per_param_k, min_effort) -- None means "leave that knob alone".
VARIANTS = {
    "baseline":    [(40, None, None, None), (100, None, None, None), (10**9, None, None, None)],
    "chunks_only": [(40, None, None, None), (100, 70, 4, None), (10**9, 110, 5, None)],
    "effort_only": [(40, None, None, None), (100, None, None, "medium"), (10**9, None, None, "medium")],
    "both":        DEFAULT_SIZE_TIERS,
}


def run_one_paper(paper_xml, vocab_path, gold_path, triage_json=None, model="gpt-5-mini",
                  base_effort="low", variants=None, out_dir="ab_results"):
    variants = variants or VARIANTS
    out_dir = Path(out_dir) / Path(paper_xml).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = open(paper_xml, encoding="utf-8").read()
    sectioned = strip_markup_sectioned(raw)
    if triage_json:
        triage = json.load(open(triage_json))
    else:
        print(f"  running triage for {paper_xml} ...")
        triage, sectioned = run_triage(raw)

    skills = build_skill_library(vocab_path)
    subskills = build_subtype_skill_library(vocab_path)
    gold_rows, _ = load_gold(gold_path)

    rows = []
    for name, tiers in variants.items():
        out_path = out_dir / f"{name}_results.json"
        stats_path = out_dir / f"{name}_stats.json"
        print(f"  [{name}] running lean ...")
        t0 = time.time()
        results, stats = run_lean(triage, sectioned, skills, subskills, model=model,
                                  reasoning_effort=base_effort, size_tiers=tiers,
                                  out_path=str(out_path), stats_path=str(stats_path))
        wall = time.time() - t0

        our_rows = load_ours(str(out_path))
        matched, missed, extra = match(gold_rows, our_rows)
        n_gold = len(gold_rows)
        n_our = len(matched) + len(extra)
        recall = len(matched) / n_gold if n_gold else float("nan")
        precision = len(matched) / n_our if n_our else float("nan")
        u = stats["usage"]
        rows.append(dict(variant=name, gold=n_gold, matched=len(matched), missed=len(missed),
                         extra=len(extra), recall=recall, precision=precision,
                         calls=u.get("calls", 0), input_tokens=u.get("input_tokens", 0),
                         output_tokens=u.get("output_tokens", 0),
                         reasoning_tokens=u.get("reasoning_tokens", 0), wall=round(wall, 1),
                         scaled_jobs=[j["skill"] for j in stats["jobs"]
                                     if j.get("reasoning_effort_used") != base_effort
                                     or j.get("max_chunks_used", 40) != 40]))
    return rows


def print_table(all_rows, paper_label=""):
    print(f"\n{'variant':<14}{'gold':>5}{'match':>6}{'miss':>5}{'extra':>6}{'recall':>8}{'precis':>8}"
          f"{'calls':>7}{'in_tok':>8}{'out_tok':>8}{'reason':>8}{'sec':>7}")
    for r in all_rows:
        print(f"{r['variant']:<14}{r['gold']:>5}{r['matched']:>6}{r['missed']:>5}{r['extra']:>6}"
              f"{r['recall']*100:>7.0f}%{r['precision']*100:>7.0f}%"
              f"{r['calls']:>7}{r['input_tokens']:>8}{r['output_tokens']:>8}{r['reasoning_tokens']:>8}{r['wall']:>7.1f}")
    base = next((r for r in all_rows if r["variant"] == "baseline"), None)
    if base:
        print(f"\ndelta vs baseline (recall / extra tokens):")
        for r in all_rows:
            if r["variant"] == "baseline":
                continue
            d_recall = (r["recall"] - base["recall"]) * 100
            d_tok = (r["input_tokens"] + r["output_tokens"]) - (base["input_tokens"] + base["output_tokens"])
            print(f"  {r['variant']:<14} recall {d_recall:+.0f} pts   tokens {d_tok:+d}   "
                  f"scaled: {', '.join(r['scaled_jobs']) or '(nothing scaled)'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper", nargs="?", help="path to paper.xml (single-paper mode)")
    ap.add_argument("vocab", nargs="?", default="vocab_tree.json")
    ap.add_argument("--gold", help="path to server.json ground truth (single-paper mode)")
    ap.add_argument("--triage-json", help="reuse a saved triage result instead of calling the LLM again")
    ap.add_argument("--papers-root", help="batch mode: a directory of <paper_id>/ folders, each with "
                                          "one .xml and a server.json (same layout as ran.zip)")
    ap.add_argument("--limit", type=int, default=2, help="batch mode: max papers to run (cost control)")
    ap.add_argument("--effort", default="low", help="base reasoning effort before any scaling")
    ap.add_argument("--batch-triage-json", help="batch mode: reuse this triage file for every paper "
                                                "instead of calling the LLM again (rarely correct across "
                                                "different papers; mainly for offline testing)")
    args = ap.parse_args()

    all_results = {}

    if args.papers_root:
        root = Path(args.papers_root)
        done = 0
        for p in sorted(root.iterdir()):
            if done >= args.limit:
                print(f"\n(stopped at --limit {args.limit}; pass a higher --limit to run more)")
                break
            xmls = list(p.glob("*.xml"))
            gold = p / "server.json"
            if not p.is_dir() or not xmls or not gold.exists() or gold.stat().st_size == 0:
                continue
            print(f"\n=== paper {p.name} ===")
            rows = run_one_paper(str(xmls[0]), args.vocab, str(gold), triage_json=args.batch_triage_json,
                                 base_effort=args.effort, out_dir=f"ab_results/{p.name}")
            print_table(rows)
            all_results[p.name] = rows
            done += 1
    else:
        if not args.paper or not args.gold:
            print("single-paper mode needs: paper.xml vocab_tree.json --gold server.json", file=sys.stderr)
            sys.exit(1)
        rows = run_one_paper(args.paper, args.vocab, args.gold, args.triage_json, base_effort=args.effort)
        print_table(rows)
        all_results[Path(args.paper).stem] = rows

    json.dump(all_results, open("ab_tier_comparison.json", "w"), indent=2, ensure_ascii=False)
    print("\nFull report written to ab_tier_comparison.json")


if __name__ == "__main__":
    main()
