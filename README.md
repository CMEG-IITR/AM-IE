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

Set `OPENAI_API_KEY` in the environment before running commands that call the OpenAI API. Local credentials belong in `.env`, which is ignored by Git.

