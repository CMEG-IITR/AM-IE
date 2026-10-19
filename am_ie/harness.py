"""Reproducible run wrapper around the existing AM extraction pipeline.

Each invocation gets its own directory containing the inputs' fingerprints,
triage, extraction, usage stats, and a manifest of the settings used. The
pipeline remains the policy; this module owns run state and provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import agentic_pipeline, llm_pipeline, triage_pipeline
from .material_classifier import classify_material


@dataclass(frozen=True)
class HarnessConfig:
    model: str = "gpt-5-mini"
    triage_backend: str = "gpt5mini"
    reasoning_effort: str = "low"
    per_param_k: int = 3
    max_chunks: int = 40
    retry_rounds: int = 0
    max_workers: int = 4
    allow_skill_requests: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _code_fingerprints() -> dict:
    """Capture the editable policy surface, including uncommitted changes."""
    root = Path(__file__).resolve().parent.parent
    paths = [
        root / "am_ie" / "harness.py",
        root / "am_ie" / "agentic_pipeline.py",
        root / "am_ie" / "triage_pipeline.py",
        root / "am_ie" / "chunk_retrieval.py",
        root / "am_ie" / "llm_pipeline.py",
        root / "am_ie" / "material_classifier.py",
        root / "am_ie" / "si_units.py",
        root / "am_ie" / "param_category.py",
    ]
    paths.extend(sorted((root / "am_ie" / "skills").glob("**/SKILL.md")))
    return {str(path.relative_to(root)): _sha256(path) for path in paths if path.is_file()}


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(paper_path, vocab_path="vocab_tree.json", output_root="harness_runs",
        config: Optional[HarnessConfig] = None, triage_path=None):
    """Run triage and lean extraction, saving a self-contained run record."""
    config = config or HarnessConfig()
    paper = Path(paper_path).resolve()
    vocab = Path(vocab_path).resolve()
    if not paper.is_file():
        raise FileNotFoundError(f"Paper not found: {paper}")
    if not vocab.is_file():
        raise FileNotFoundError(f"Vocabulary not found: {vocab}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / f"{timestamp}_{paper.stem}"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = {
        "run_id": run_dir.name,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _revision(),
        "code_sha256": _code_fingerprints(),
        "config": asdict(config),
        "inputs": {
            "paper": {"path": str(paper), "sha256": _sha256(paper)},
            "vocabulary": {"path": str(vocab), "sha256": _sha256(vocab)},
        },
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    active_stage = "parse"
    try:
        # Start a fresh usage window before any model-backed stage. Triage
        # calls are recorded by call_gpt_mini_tool; extraction usage is
        # returned directly by run_lean and is combined below.
        llm_pipeline.USAGE_LOG.clear()
        raw_markup = paper.read_text(encoding="utf-8")
        active_stage = "triage"
        if triage_path:
            triage_file = Path(triage_path).resolve()
            triage = json.loads(triage_file.read_text(encoding="utf-8"))
            sectioned = triage_pipeline.strip_markup_sectioned(raw_markup)
            manifest["inputs"]["triage"] = {
                "path": str(triage_file), "sha256": _sha256(triage_file),
            }
        else:
            triage, sectioned = triage_pipeline.run_triage(
                raw_markup, model=config.model, backend=config.triage_backend,
                output_path=str(run_dir / "triage.json"),
            )
        _write_json(run_dir / "triage.json", triage)
        _write_json(run_dir / "sections.json", {
            "sections": dict(sectioned),
            "tables": getattr(sectioned, "tables", []),
        })

        active_stage = "material_classification"
        skill_library = agentic_pipeline.build_skill_library(str(vocab))
        subtype_library = agentic_pipeline.build_subtype_skill_library(str(vocab))
        material = classify_material(triage, scan_text="\n\n".join(sectioned.values()))
        _write_json(run_dir / "material_classification.json", material)

        triage_usage = llm_pipeline.summarize_usage(llm_pipeline.USAGE_LOG)
        active_stage = "retrieve_extract_verify"
        results, stats = agentic_pipeline.run_lean(
            triage, sectioned, skill_library, subtype_library,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            per_param_k=config.per_param_k,
            max_chunks=config.max_chunks,
            retry_rounds=config.retry_rounds,
            allow_skill_requests=config.allow_skill_requests,
            max_workers=config.max_workers,
            out_path=str(run_dir / "extraction.json"),
            stats_path=str(run_dir / "stats.json"),
        )
        extraction_usage = stats.get("usage", {})
        usage_fields = (
            "calls", "input_tokens", "output_tokens", "reasoning_tokens",
            "cached_tokens", "seconds",
        )
        stats["stage_usage"] = {
            "triage": triage_usage,
            "extraction": extraction_usage,
        }
        stats["usage"] = {
            field: triage_usage.get(field, 0) + extraction_usage.get(field, 0)
            for field in usage_fields
        }
        _write_json(run_dir / "stats.json", stats)
        _write_json(run_dir / "trajectory.json", {
            "schema_version": 1,
            "run_id": run_dir.name,
            "stages": [
                {
                    "stage": "parse",
                    "status": "complete",
                    "section_names": list(sectioned.keys()),
                    "section_count": len(sectioned),
                    "table_count": len(getattr(sectioned, "tables", [])),
                },
                {
                    "stage": "triage",
                    "status": "complete",
                    "backend": "saved_triage" if triage_path else config.triage_backend,
                    "process_categories": triage.get("process_categories", []),
                    "process_subtypes": triage.get("process_subtypes", []),
                    "printed_materials": triage.get("printed_materials", []),
                    "relevant_sections": triage.get("relevant_sections", []),
                    "usage": triage_usage,
                },
                {
                    "stage": "material_classification",
                    "status": "complete",
                    "result": material,
                },
                {
                    "stage": "retrieve_extract_verify",
                    "status": "complete",
                    "jobs": stats.get("jobs", []),
                    "usage": extraction_usage,
                    "result_counts": {
                        "verified_rows": sum(len(r.get("parameters", {}).get("verified", [])) for r in results.values()),
                        "flagged_rows": sum(len(r.get("parameters", {}).get("flagged", [])) for r in results.values()),
                    },
                },
            ],
        })
        manifest["status"] = "complete"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = round(time.time() - started, 2)
        manifest["llm_usage"] = stats["usage"]
        manifest["stage_usage"] = stats["stage_usage"]
        manifest["result_counts"] = {
            "skills": len(results),
            "verified_rows": sum(len(r.get("parameters", {}).get("verified", [])) for r in results.values()),
            "flagged_rows": sum(len(r.get("parameters", {}).get("flagged", [])) for r in results.values()),
        }
        _write_json(manifest_path, manifest)
        return run_dir, results, stats
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_stage"] = active_stage
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = round(time.time() - started, 2)
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(manifest_path, manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run AM extraction with isolated artifacts and provenance.")
    parser.add_argument("paper", help="Path to a paper XML/HTML file")
    parser.add_argument("--vocab", default="vocab_tree.json")
    parser.add_argument("--output-root", default="harness_runs")
    parser.add_argument("--triage-json", help="Reuse a saved triage result")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--triage-backend", choices=("gpt5mini", "ollama"), default="gpt5mini")
    parser.add_argument("--effort", choices=("minimal", "low", "medium", "high"), default="low")
    parser.add_argument("--per-param-k", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=40)
    parser.add_argument("--retry-rounds", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-skill-requests", action="store_true")
    args = parser.parse_args(argv)
    config = HarnessConfig(
        model=args.model, triage_backend=args.triage_backend,
        reasoning_effort=args.effort, per_param_k=args.per_param_k,
        max_chunks=args.max_chunks, retry_rounds=args.retry_rounds,
        max_workers=args.workers, allow_skill_requests=args.allow_skill_requests,
    )
    run_dir, results, stats = run(
        args.paper, args.vocab, args.output_root, config, args.triage_json,
    )
    agentic_pipeline._print_results(results)
    usage = stats.get("usage", {})
    print(f"Run artifacts: {run_dir}")
    print(f"LLM calls: {usage.get('calls', 0)} | input tokens: {usage.get('input_tokens', 0)} | "
          f"output tokens: {usage.get('output_tokens', 0)} | wall: {stats.get('wall_seconds', 0)}s")


if __name__ == "__main__":
    main()
