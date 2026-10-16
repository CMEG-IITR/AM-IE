# AM-IE

Tools for extracting additive manufacturing parameters from research papers.

## Project layout

- `am_ie/` contains the pipeline, retrieval, normalization, classification, and comparison modules.
- `am_ie/skills/<skill-name>/SKILL.md` contains one extraction skill per process category and configured subtype. Each file uses the standard `name` and `description` frontmatter; category parameters still come from `vocab_tree.json`.
- The Python files in the repository root are compatibility launchers for the package modules. Existing commands such as `python agentic_pipeline.py` continue to work.
- `old/` contains the earlier extraction and vocabulary-building scripts.
- `vocab_flat.json` and `vocab_tree.json` are generated vocabulary data used by the pipelines.
- `papers/` is for local source papers and is excluded from Git.
- Generated extraction and comparison outputs are excluded from Git.

## Common commands

Run these from the repository root so the default vocabulary and data paths resolve:

```bash
python agentic_pipeline.py
python triage_pipeline.py
python chunk_retrieval.py paper.xml
python compare_vs_ground_truth.py path/to/papers
python json_to_csv.py agentic_extraction_results.json
```

Set `OPENAI_API_KEY` in the environment before running commands that call the OpenAI API. Local credentials belong in `.env`, which is ignored by Git.

