# AM extraction harness

## Design

The existing triage, retrieval, skill dispatch, structured extraction, and
symbolic verification are the harness policy. `am_ie.harness` is the run
controller: it supplies the paper and taxonomy, records the policy settings,
and writes a durable artifact bundle for each run. Keeping that boundary
explicit lets us improve one piece at a time without making a new model or a
larger token budget look like a harness improvement.

The current run sequence is:

1. Parse paper markup and preserve section/table structure.
2. Triage for material/process and relevant sections, or load a saved triage.
3. Load only the relevant category/subtype skills and retrieve evidence chunks.
4. Make schema-locked extraction calls; run the existing code checks and
   targeted repair behavior.
5. Save results, stats, and a manifest with configuration and SHA-256
   fingerprints of the paper, vocabulary, code modules, and skill prompts.

The model remains the policy inside each LLM call. Parsing, retrieval,
taxonomy constraints, routing, validation, budgets, retries, and persistence
are harness decisions. Parallel category jobs stay bounded by `--workers`;
this flow does not need a multi-agent hierarchy to be a harness.

## Run artifacts

```bash
python -m am_ie.harness papers/paper.xml --vocab vocab_tree.json
```

Artifacts are written to an ignored `harness_runs/<timestamp>_<paper>/`
directory. The bundle contains:

- `manifest.json`: status, git revision, input/code fingerprints, settings,
  elapsed time, usage, and row counts.
- `triage.json`: process/material routing and relevant sections.
- `sections.json`: parsed section text and structured XML table grids.
- `material_classification.json`: symbolic material/property routing.
- `extraction.json` and `stats.json`: verified/flagged rows and per-job usage.
- `trajectory.json`: ordered stage outcomes, triage decisions, retrieval/extraction
  jobs, per-stage usage, and result counts. Failed runs identify their failing
  stage in `manifest.json`.
- Per-skill stats include retrieved chunk IDs by parameter, sent chunk metadata,
  prompt/schema fingerprints, model-call outcomes, and verifier decisions.

Failed runs retain a manifest with the error and any artifacts written before
failure. To replay extraction without another triage call, pass
`--triage-json path/to/triage.json`.

## Evaluation protocol

The first harness change establishes reproducible runs; it does not claim an
accuracy improvement. Before tuning prompts, retrieval, retries, or budgets,
create a fixed evaluation manifest that points to licensed/versioned papers
and adjudicated expected rows. Keep that set outside generated run folders.
For each paper, score extraction against adjudicated rows and report precision,
recall, quote/evidence validity, material attribution, and empty-result
correctness alongside calls, tokens, and elapsed time.

The local pilot set is listed in `evaluation/benchmark_manifest.json`. Its
existing `server.json` files are candidate references and remain marked
pending until a person checks their completeness, taxonomy mapping, units,
quotes, and material links against the paper. Compare a run without claiming
accuracy using:

```bash
python -m am_ie.harness_eval harness_runs/pilot/<run-id> \
  --reference papers/ibrahim/ran/1/server.json
```

The report is written to that run folder as `comparison.json`; it includes
reference audits, matched and differing rows, material-name differences, and
whether each cited value and quote appears in the paper. Only pass
`--reference-status human_reviewed` after a person has completed that review;
then the report also includes exact-match precision, recall, and F1.

Compare one harness change at a time. Hold the paper set, taxonomy, model,
reasoning effort, and call/token budget fixed; repeat stochastic runs and
report per-paper outcomes and variability, not only a pooled average. When
testing a model change, cross it with the same harness versions so model and
harness effects remain visible. Retrieval recall can be measured separately
from extraction quality using the existing `chunk_retrieval.evaluate` path.

The evaluation essay's central point applies directly here: capability comes
from the model, harness, environment, and inference budget together. It also
recommends fixing budgets, versioning the full stack, publishing trajectories,
and measuring variance. This repo has no checked-in adjudicated evaluation
corpus yet, so claims about end-to-end quality should wait until that corpus
and its scoring rules exist.

Reference: [On the Evaluation of Agent Harnesses](https://blog.debargha.com/essays/on-agent-harnesses/).

## Next steps

1. Move the current monolithic orchestration helpers behind stage interfaces
   (`parse`, `triage`, `retrieve`, `extract`, `verify`, `score`) while keeping
   the present CLI as a compatibility entry point.
2. Add a versioned, adjudicated paper set and a machine-readable scorer.
3. Add structured per-call trajectories (stage, skill, selected chunk IDs,
   prompt/config fingerprint, response metadata, verifier decisions), with
   paper text and secrets excluded from logs by default.
4. Run controlled ablations for retrieval, subtype skills, verification,
   retries, and concurrency before adopting more complex planning or agents.
