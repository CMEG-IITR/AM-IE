# Table extraction skill

Use this guidance whenever the excerpts include table rows. This skill is part of the current extraction request; it does not require a separate model call.

## Instructions

- Treat every supplied table row as evidence, even when its section is not among the paper's prose sections selected by triage.
- Read each row together with its table caption and column headers. Headers define what each cell means; do not interpret a numeric cell without its column label.
- Check every supplied row against every parameter in the loaded process skill. Extract every applicable value, including values in appendices and tables about measured properties, composition, and processing conditions.
- A table may report the parameter under a different phrase than the taxonomy uses. Map it to the closest exact parameter name allowed by the extraction schema; use the table's units and copy the cell value exactly.
- For a row with multiple conditions (for example, temperatures), emit a separate extraction row for every condition/value pair when the table reports them separately. Preserve the condition in the value only when needed to distinguish the reported value and supported by the cited excerpt.
- Cite the excerpt ID for the table row containing the value. Do not cite prose or another table row as a substitute.
- Do not skip a relevant value because a material/sample name appears in the caption or surrounding table context rather than in the row itself. Use the available material field rules and do not invent a sample identity.
- If no table row supports a parameter, omit it. Never infer missing cells, calculate values, or fabricate values.
