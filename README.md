# AM-IE

Tools for extracting additive manufacturing parameters from research papers.

## Project layout

- `am_ie/` contains the pipeline, retrieval, normalization, classification, and comparison modules.
- `am_ie/skills/<skill-name>/SKILL.md` contains one extraction skill per process category and configured subtype. Each file uses the standard `name` and `description` frontmatter; category parameters still come from `vocab_tree.json`.
- `scripts/compat/` contains legacy command wrappers for the package modules. Run them as `python scripts/compat/agentic_pipeline.py`.
- `legacy/` contains the earlier standalone extraction and vocabulary-building scripts.
- `vocab_flat.json` and `vocab_tree.json` are generated vocabulary data used by the pipelines.
- `papers/` is for local source papers and is excluded from Git.
- Generated extraction and comparison outputs are excluded from Git.

## Common commands

Run these from the repository root so the default vocabulary and data paths resolve:

```bash
python -m am_ie.agentic_pipeline
python -m am_ie.triage_pipeline
python -m am_ie.chunk_retrieval paper.xml
python -m am_ie.compare_vs_ground_truth path/to/papers
python -m am_ie.json_to_csv agentic_extraction_results.json
```

## Harness runs

Use the run harness to execute triage and extraction with per-run artifacts,
input fingerprints, settings, usage, and failure status:

```bash
python -m am_ie.harness papers/paper.xml --vocab vocab_tree.json
```

Each run is written under `harness_runs/<timestamp>_<paper>/`. The directory
contains the manifest, triage result, parsed sections and tables, symbolic
material classification, extraction, and run statistics. Use `--triage-json`
to replay extraction from a saved triage result. Existing pipeline commands
remain available for baseline comparisons.

Compare a completed run with a reference JSON using
`python -m am_ie.harness_eval harness_runs/<run-id> --reference path/to/server.json`.
Reports treat references as unreviewed candidates by default. The checked pilot
subsets under `evaluation/references/` can be compared with
`--reference-status human_reviewed_subset`; this reports row agreement without
claiming full-paper accuracy. Use `human_reviewed` only after checking reference
completeness as well as each row.

Set `OPENAI_API_KEY` in the environment before running commands that call the OpenAI API. Local credentials belong in `.env`, which is ignored by Git.

