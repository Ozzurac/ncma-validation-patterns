# Public model-evaluation patterns

**Purpose:** small Python examples for transparent LLM and ASR benchmarking,
train/dev/holdout isolation and checkpoint selection. This is an **educational
reference**, not the privately operated NCMA evaluation harness.

No models, external APIs, network calls, secrets, model weights, private data,
real transcripts or training implementation are included. Every example input
in this repository is **synthetic**.

## Run

In the existing `ncma-validation-patterns` repository, add the accompanying
Python modules, tests and example script, then execute:

```bash
python -m unittest discover -s tests -v
python -m examples.model_evaluation_demo
python -m compileall -q validation_patterns examples tests
```

Requires Python 3.11+; the standalone evaluation modules use only Python's
standard library. The public example does not call any LLM or ASR service.

## ASR benchmark accounting

`validation_patterns/asr_benchmark.py` demonstrates:

1. Freeze a set of unique clip IDs with immutable references.
2. Normalize references and hypotheses under **one declared policy**.
3. Calculate aggregate WER = (S + D + I) / reference words and character
   error rate = (S + D + I) / reference characters.
4. Count whole-utterance exact-normalized matches separately.
5. Record both clip latency distributions and aggregate real-time factor
   (RTF = cumulative inference seconds / cumulative audio seconds).
6. Report warm median excluding the first clip separately from total latency.

The example applies Unicode NFKC, case folding, punctuation-to-space, whitespace
collapse and whitespace word segmentation. CER excludes whitespace. **That is
an illustrative documented policy, not an assertion that the historical private
runner used exactly this normalization.** WER can exceed 100% when there are
many insertions. The example p95 uses nearest-rank quantiles.

A rigorous live ASR experiment also needs pinned model versions, fixed audio
sampling conditions, provider validation, model-load timing, cache-warming
accounting, independent data, and live-microphone acceptance if applicable.

## LLM inference accounting

`validation_patterns/llm_benchmark.py` demonstrates:

- Exact case IDs and token-count validation.
- Separate first-token and post-first-token decode time.
- Decode throughput measured as (generated tokens - 1) / decode seconds,
  **not** including the first token in the decode numerator.
- Explicit completed, truncated and failed counts.
- Median timing and tokens/s calculated on completed cases, with the
  resulting **completion-rate selection bias** visible.
- Empty evaluations or invalid timing are rejected.

For historical model comparisons, model format, runtime backend, quantization,
context size, generation cap, sampler, model-revision identity and hardware
capacity must all be recorded. A token/s leaderboard without correctness and
completion metrics is not sufficient for candidate promotion.

Latency and time-to-first-token measurements from different inference backends
may not be directly comparable. GPU memory on Windows can include allocations
outside the candidate process. Sampling GPU totals does not prove exclusive
model residency. Truncated outputs are not semantically successful outputs.

## ML dataset and checkpoint selection

`validation_patterns/split_audit.py` demonstrates:

- Unique record IDs and explicit **TRAIN**, **DEV** and **HOLDOUT** boundaries.
- Detection of repeated normalized prompts across splits.
- Detection of a shared provenance-group ID spanning splits.
- A dataset with no holdout returns **INCONCLUSIVE**, never PASS.
- Selection of the best *eligible development* checkpoint subject to a
  general-language regression threshold, without accepting holdout scores as
  part of the selection interface.

Important: exact string fingerprints and manually assigned group IDs **cannot
exclude semantic paraphrases, synthetic generator leakage, common task ancestry
or memorized benchmark questions**. Real ML experiments require additional
independent provenance and adversarial contamination checks. The demo intentionally
omits any model training, optimizer settings, adaptation algorithms or private
family registry.

## Historical NCMA studies and their limits

- [Local LLM benchmark](https://github.com/Ozzurac/ncma-engineering/blob/main/projects/local-llm-benchmarking.md)
- [Brazilian Portuguese ASR benchmark](https://github.com/Ozzurac/ncma-engineering/blob/main/projects/asr-benchmarking.md).
- [Fine-tuning and LoRA evaluation](https://github.com/Ozzurac/ncma-engineering/blob/main/projects/fine-tuning-evaluation.md).

The figures in those studies come from bounded historical private observations.
**Running this public example will not reproduce those figures.** Without the
original frozen evaluation manifests, exact model artifacts and execution stack,
that would be an unjustified claim.

## Publication boundary

These files were independently authored for publication. Do not import the
private Jarvis/ModForge/MCP original source, operational endpoints, credentials,
local filenames or paths, prompts, audio, weights, teacher output, private
experiments or original Git history into this repository.
