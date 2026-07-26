# Pairwise Ablation and Medical Backbone Evaluation Design

## Goal

Add two independent experiments without changing the accepted results in the
current main medical table or the archived non-medical timeline table:

1. extend the PMOA-TTS component analysis with the three pairwise removals of
   the positive components already shown in the paper; and
2. add a Section 4.2 medical-backbone comparison covering five pipelines and
   five backbones under the frozen five-source protocol.

The existing Qwen3-VL scale comparison remains the Section 4.1 Table 2. The
non-medical LoCoMo, LongMemEval, DialSim, and RHELM table retains its accepted
values and moves to the appendix. It is not rerun or combined with the new
medical-backbone table.

## Frozen Experiment Matrix

### Pairwise component ablation

The ablation uses the same 5,000 frozen PMOA-TTS cases, Qwen3-VL-8B model,
prompt, decoding configuration, evaluator, request seed, and strict
no-fallback gates as the existing complete ablation. The three new variants
are:

- no source-aligned evidence notes and no applied memory cleaning (`-E-C`);
- no source-aligned evidence notes and fixed top-k retrieval (`-E-K`); and
- no applied memory cleaning and fixed top-k retrieval (`-C-K`).

Here `E` denotes source-aligned evidence notes, `C` applied memory cleaning,
and `K` dynamic top-k retrieval. Full MediMem and the three existing
single-factor rows remain the comparison anchors. The implementation must
compose existing feature flags; it must not introduce a different prompt,
evaluator, sample subset, or scoring rule for the pairwise rows.

### Medical backbone comparison

The new Section 4.2 table uses the frozen five-source, 500-cases-per-source
protocol already used by the accepted Qwen3-VL scale table:

- MedMCQA: 500;
- MedQA: 500;
- Medical Meadow WikiDoc: 500;
- PMOA-TTS: 500; and
- PMC-Patients: 500.

The five pipelines are:

- Direct;
- CoT;
- A-MEM;
- CliCARE; and
- MediMem (Ours).

The five backbones are:

- Qwen3-VL-30B;
- Llama-3.1-70B (`llama-3.1-70b`);
- DeepSeek-V3 (`deepseek-v3`);
- GPT-4.1 (`gpt-4.1`); and
- Gemini-2.5-Flash (`gemini-2.5-flash`).

The accepted Qwen3-VL-30B cells are reused from their original complete frozen
runs after manifest and sample-ID verification. They are not rerun. Each of
the four remote backbones runs all five pipelines over all 2,500 frozen cases,
for 20 new model-pipeline units and 50,000 pipeline-case units. Every unit uses
the same visible inputs, method adapter, prompt family, maximum completion
budget, request seed, evaluator, metric implementation, and strict
no-fallback policy. Results from different runs or sample subsets may not be
spliced.

## API and Model Audit

The WLAI OpenAI-compatible endpoint is `https://api.wlai.vip/v1`. The API key
is read only inside the experiment process from
`/root/.config/medimem/wlai.key`; it must never appear in commands, logs,
manifests, predictions, Git, or paper files.

Before formal launch, the supervisor records:

- the UTC query time and SHA-256 of the sanitized `/v1/models` response;
- a minimal live chat-completion probe for each exact target model ID;
- the returned model identity and non-empty usage fields;
- frozen sample-manifest hashes;
- code commit, prompt/config hashes, evaluator version, and method-adapter
  identities; and
- a redacted API configuration containing the base URL but no secret.

The incomplete nature of the `/v1/models` catalogue is respected: an exact
model is accepted based on a successful live probe, not catalogue membership
alone.

## Execution Architecture

GPU 0 is reserved for the pairwise PMOA-TTS ablation. It hosts the local
Qwen3-VL-8B service and runs the three pairwise variants with enough genuine
worker concurrency to keep the service saturated.

GPU 1 is reserved for the medical-backbone experiment. Remote APIs perform
the target-model generation. GPU 1 runs the existing CUDA-capable local
embedding and retrieval work used by the A-MEM and MediMem adapters; Direct,
CoT, and the released-source CliCARE adapter remain on their frozen execution
paths. Medical Top-1 and Diagnosis F1 are computed deterministically and do
not add an LLM judge. No artificial workload may be added to create a
misleading GPU-Util value.

The medical supervisor interleaves model and dataset queues so API waiting
from one queue does not idle all workers. Concurrency begins with bounded
probes and increases only while the endpoint returns stable success without
rate-limit or transport failures. Each target model has an independent run
root, progress journal, prediction files, metrics, and audit. Resume is
allowed only for missing sample IDs under byte-identical frozen
configuration. End-to-end throughput, API concurrency, GPU utilization, and
GPU memory are reported separately.

Because target generation is remote, sustained 100% GPU utilization on GPU 1
is not a validity requirement and cannot be guaranteed. The optimization
target is maximum valid end-to-end throughput, not synthetic GPU load.

## Output Table

The current Section 4.1 Table 2 remains a separate Qwen3-VL scale comparison.
The new Section 4.2 table has columns:

`Paradigm | Method | Reference | Backbone | MedMCQA | MedQA | WikiDoc |
PMOA-TTS | PMC-Patients`

Each dataset cell reports `Top-1 / Diagnosis F1`. Rows are grouped first by
pipeline and then by the five backbones, matching the visual organization of
the approved reference table. Within each backbone, the best pipeline value
for a dataset metric is bold and the second-best is underlined. The table
reports one complete frozen-run point estimate per cell; it does not report
mean or standard deviation.

The appendix non-medical table keeps its accepted values and protocol text.
Moving it must not silently renumber or break references.

## Validation and Admission Gates

Every new formal unit must satisfy all of the following:

- exact frozen sample-ID set and expected count;
- unique prediction IDs and complete metric coverage;
- live API and real-data requirements enabled;
- zero fallback, blocked method, API error, worker failure, missing
  prediction, and progress failure;
- zero critical leakage and zero needs-review findings;
- manifest and validation guards passed;
- exact target model ID and method-adapter identity recorded; and
- no reference diagnosis, gold answer, evaluator field, or adversarial target
  visible during inference.

The pairwise ablation additionally verifies that each row differs from Full
MediMem in exactly the declared two feature flags. The medical-backbone audit
checks a complete 4-by-5 grid of new model-pipeline units and separately
verifies the provenance of the five reused Qwen3-VL-30B units.

Failed or incomplete units remain archived as failed runs and do not enter the
paper. Numeric cells remain `Pending` until their complete unit passes every
gate.

## Paper and Archive Updates

After all accepted units pass:

1. generate the pairwise ablation rows from audited metrics;
2. generate the new Section 4.2 table from the audited medical grid;
3. retain the existing Section 4.1 Table 2 values unchanged;
4. move the accepted non-medical table and its analysis to the appendix;
5. update claims only to match the completed results, including negative or
   mixed findings;
6. archive manifests, predictions, metrics, validations, probes, sanitized
   service records, commands, and utilization traces locally; and
7. follow the established LaTeX build, visual verification, PDF, ZIP, and
   Overleaf synchronization workflow.

No result is inferred, fabricated, or copied from a different model, dataset
version, sample count, evaluator, or run.
