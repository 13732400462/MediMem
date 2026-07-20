# Table 1 Provenance Layout and Table 3 Ranking Design

Date: 2026-07-20

## Scope

This change revises the paper's table presentation only. It does not rerun experiments, alter models, change prompts or evaluators, or modify any reported metric value.

The affected paper artifacts are:

- `tables/main_medical_results.tex` (main-paper Table 1)
- `tables/pmoa_main_ablation.tex` (main-paper Table 3)
- table captions and the immediately related explanatory text
- bibliography entries required by the new compact reference aliases
- active single- and double-column table width declarations
- the compiled PDFs and Overleaf package

## Table 1 Structure

Table 1 follows the information hierarchy used by the downloaded PhysAgent paper:

1. `Type`
2. `Method`
3. `Reference`
4. dataset-specific metrics
5. pooled metrics

The rows are grouped with `\multirow` in this order:

- **Prompt-based:** Direct, CoT
- **Memory agent:** A-MEM, Mem0, Letta/MemGPT
- **Clinical agent:** DDO, ColaCare, EHRMaster, CliCARE, MedAgentBoard
- **Ours:** MediMem

The metric block retains the existing frozen five-source protocol:

- Per-source PDO: MedMCQA, MedQA, WikiDoc, PMOA, PMC
- Overall: Top-1, Diagnosis F1, PDO, CDR F1

No experimental values or rank markings are changed except where table structure requires moving the existing values into the grouped rows.

## Reference Convention

The `Reference` column uses the exact compact convention demonstrated by PhysAgent: `\defcitealias` defines a venue-year alias and `\citetalias` renders it in the table. It does not display author names.

The aliases are:

| Method | Reference label |
|---|---|
| Direct | Ours |
| CoT | NeurIPS'22 |
| A-MEM | arXiv'25 |
| DDO | EMNLP'25 |
| ColaCare | WWW'25 |
| EHRMaster / EHRStruct pipeline | AAAI'26 |
| CliCARE | AAAI'26 |
| MedAgentBoard | NeurIPS'25 |
| Mem0 | arXiv'25 |
| Letta/MemGPT | arXiv'23 |
| MediMem | Ours |

The bibliography gains entries for CoT and CliCARE if they are not already present. Existing bibliography records remain the source of all other aliases.

The caption explicitly states that `Reference` identifies the source publication or pipeline. The table does not claim that the adapter runs reproduce each source paper's original benchmark configuration.

## Base-Model Reporting

Table 1 does not add a `Base Model` column. Every row in the frozen comparison was actually evaluated through the same Qwen3-VL-8B service, so a full column would repeat the same value and reduce metric readability.

The caption states once that all rows use Qwen3-VL-8B. Source papers may have used different native backbones, but those are not the models that generated the values in this table. The `Reference` column identifies pipeline provenance, not the source paper's original backbone.

## Table 3 Ranking Marks

Table 3 adds the same best/second-best convention used by the other result tables. Higher is better for the four quality metrics.

- MediMem is bold for Top-1 `0.7064`, Diagnosis F1 `0.4305`, PDO `0.6098`, and CDR F1 `0.6220`.
- Fixed top-k is underlined for second-best Top-1 `0.7032`, PDO `0.6072`, and CDR F1 `0.6185`.
- No memory cleaning is underlined for second-best Diagnosis F1 `0.4291`.
- Delta PDO remains unranked because it is a relative change from the full system rather than an independent quality metric.

The Table 3 caption states that best and second-best quality results are bold and underlined.

## Width and Alignment

Table widths align with the manuscript text block:

- double-column `table*` content uses `tabular*{\textwidth}` with fillable intercolumn spacing;
- single-column `table` content uses `tabular*{\columnwidth}`;
- no active table uses `resizebox` merely to force width;
- table rules and outer content edges must visually align with the corresponding single- or double-column text boundary.

The implementation audits all active paper tables, not only Tables 1 and 3. It changes width declarations only where an active table violates these rules; it does not redesign unrelated table content.

## Text and Disclosure Updates

The Table 1 caption and nearby experimental text explain:

- all rows use the same frozen cases, evaluator, and Qwen3-VL-8B service;
- `Reference` denotes the source publication or source-defined pipeline;
- released-source clinical methods are evaluated through protocol adapters;
- Mem0 and Letta/MemGPT use their native memory APIs.

The description continues to identify EHRMaster, CliCARE, and MedAgentBoard as released-source adaptations rather than original benchmark reproductions.

## Validation

The final implementation must pass:

1. source checks confirming that each Table 1 method has the intended type and reference alias;
2. a rank check confirming every Table 3 bold/underline mark against the numeric values;
3. a width audit confirming `\textwidth` for active double-column tables and `\columnwidth` for active single-column tables;
4. a full PDFLaTeX--BibTeX--PDFLaTeX build with no LaTeX error, undefined citation/reference, or overfull box;
5. rendered visual review of Table 1, Table 3, and all pages affected by table movement;
6. identical `main.pdf` and `medimem.pdf`;
7. reconstruction of the Overleaf ZIP from the active source set;
8. updates to both project-context documents recording the presentation-only change and final validation.

## Non-Goals

- No experiment reruns.
- No value replacement or cross-run result splicing.
- No change to the frozen Qwen3-VL-8B protocol.
- No reassignment of a method to a more favorable category.
- No citation label based on an unverified venue.
