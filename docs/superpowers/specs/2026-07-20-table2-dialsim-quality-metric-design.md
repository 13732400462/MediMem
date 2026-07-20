# Table 2 DialSim Quality-Metric Replacement Design

## Objective

Restore the complete four-source non-medical comparison to the main paper as Table 2, retain the strongest single complete MediMem run, and replace DialSim latency with a quality metric that is comparable across all methods.

## Frozen result selection

- The MediMem row must use the complete hybrid-BGE formal run.
- No value may be selected independently per dataset.
- The retained MediMem values are:
  - LoCoMo Answer Acc. / Evidence R@5: `52.50 / 71.76`
  - LongMemEval Answer Acc. / Evidence R@5: `39.80 / 79.53`
  - DialSim Answer Acc.: `58.60`
  - RHELM Answer Acc. / Evidence R@5: `30.34 / 28.44`
  - Average Answer Acc. / Evidence R@5: `45.31 / 59.91`
- Mem0 remains in the shared table because its comparability audit passed.

## DialSim columns

DialSim will report:

1. Answer Accuracy
2. QA F1

Latency is removed because it is hardware- and deployment-dependent and is not a quality metric comparable to the neighboring evidence-recall columns.

QA F1 is the existing token-multiset F1 already produced by the frozen evaluator. Each method's value is the sample-weighted mean over the same three DialSim subsets and the same 1,000 frozen questions.

| Method | Answer Acc. | QA F1 |
|---|---:|---:|
| Direct | 28.90 | 32.02 |
| Static RAG | 59.60 | 61.21 |
| A-MEM | 32.30 | 33.70 |
| DDO | 60.60 | 61.65 |
| G-Memory | 60.40 | 61.47 |
| MemInsight | 60.50 | 61.57 |
| MemoryOS | 60.40 | 61.47 |
| Mem0 | 60.00 | 60.92 |
| Letta/MemGPT | 26.80 | 26.42 |
| MediMem | 58.60 | 59.41 |

DDO is bold for both DialSim columns. MemInsight is underlined as second-best for both columns.

## Aggregate metrics

- Average Answer Accuracy remains the arithmetic mean of the four source-level accuracy values.
- Average Evidence R@5 remains the arithmetic mean over LoCoMo, LongMemEval, and RHELM.
- DialSim QA F1 is not mixed into Average Evidence R@5.
- Existing aggregate values and rankings therefore remain unchanged.

## Paper placement and wording

- Restore `tables/nonmedical_timeline_main.tex` to `sections/experiments.tex`, making it main-text Table 2 again.
- Remove the duplicate table input and dedicated full-table section from the appendix.
- Keep an honest main-text statement that Mem0 leads average Answer Acc. and Average R@5 (`51.01 / 69.18`) over MediMem (`45.31 / 59.91`).
- Update the caption to define DialSim QA F1 and remove all latency language.
- Preserve protocol disclosures for Mem0 native-memory R@5, the unified wrappers, and Letta/MemGPT.
- Update abstract and conclusion only as needed to remain consistent with the restored main-text table; do not claim MediMem wins the aggregate.

## Derived artifacts and validation

- Update the merged CSV and Markdown table artifact with DialSim QA F1.
- Update both project context files with the metric change and its source.
- Recompile the paper with PDFLaTeX, BibTeX, and repeated PDFLaTeX passes.
- Reject the change if there are LaTeX errors, undefined citations/references, overfull boxes, clipped table cells, or unreadable text.
- Render and visually inspect the final Table 2 page and surrounding section.
- Rebuild `medimem.pdf`, `main.pdf`, and the Overleaf ZIP.
- No model, retrieval, judge, or evaluator experiment is rerun.

## Provenance

DialSim QA F1 values come from the existing frozen formal metrics:

- Core methods: `runs/nonmedical_timeline_formal_20260716_recovery_v4/core/.../dialsim_metrics.csv`
- Mem0: `runs/nonmedical_timeline_formal_20260716_recovery_v4/mem0/.../dialsim_metrics.csv`
- Letta/MemGPT: `runs/nonmedical_timeline_formal_20260719_extractive_v8/letta/.../dialsim_metrics.csv`
- MediMem: `runs/table2_hybrid_bge_20260719/formal/dialsim/.../dialsim_metrics.csv`

All entries use `n=1,000`, the frozen sample set, and the existing evaluator output.
