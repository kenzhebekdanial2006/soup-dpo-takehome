# Soup DPO take-home

An evidence-first DPO run for Russian support preferences: actual Soup layer
streaming, an independent memory estimate, saved-adapter verification, a real
`lr=0` negative control, and streamed-vs-resident gradient comparison.

**Current status:** completed on a real Colab Tesla T4, run
[`t4-20261002T203632Z`](evidence/t4-20261002T203632Z/status.json).
100 actual optimizer steps; all 112 adapter tensors changed; ordinary PEFT reload,
adapter-disabled restoration and the real `lr=0` control passed their checks.
Streamed/resident gradients were bit-identical on three batches with deterministic
math attention. All 18 engineering tests passed on T4.

Synthetic heldout preference accuracy: 87% to 100% (100 rows, ten independent
groups; paired delta 0.13, group-bootstrap 95% CI [0.00, 0.36]). PyTorch training
peak: 2.017 GiB allocated / 6.439 GiB reserved; sampled main-command device peak:
6.651 GiB. The original 3.815 GiB memory plan underestimated reserved/device
footprint; the report explains the assumptions and gap without changing it.

Soup's numeric gate returned SHIP; final deployment call: **DON'T SHIP**, because
synthetic templates and four general questions cannot validate deployment quality.
See [two-page report](reports/report.pdf),
[report source](reports/report.md), and [AI disclosure](docs/AI_USAGE.md).

## Run on free Colab T4

1. Upload `notebooks/takehome_t4.ipynb` to Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**. The notebook creates its
   own Python 3.11 environment, including on Colab's Python 3.13 runtime.
3. Run all cells. When prompted, upload the supplied `soup-dpo-takehome.zip`.
4. The notebook installs the reviewed Soup release, runs all stages, and
   downloads a result ZIP even if a stage fails. Keep that ZIP and the completed
   notebook **with cell outputs**. A full run downloads roughly 3 GB of model
   weights and can take a substantial part of a free Colab session.
5. Extract the result ZIP into this project, retaining every evidence directory.
   The result includes the adapter and its initial snapshot. These weights are
   excluded from Git; share the result ZIP privately with the reviewer.

The input ZIP must contain this project at its root. No Hugging Face token or
paid GPU is needed for the chosen non-gated base model. Colab T4 availability is
not guaranteed; an unavailable GPU is a blocked run, not a passing result.

Equivalent Linux/T4 commands:

```bash
python -m pip install torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt -c config/t4-constraints.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -m scripts.run_t4
```

For a rerun, use `python -m scripts.run_t4 --revision <resolved-model-SHA>` from
the original `model_manifest.json`. Every attempt gets a new timestamped evidence
directory. Do not merge logs from different attempts or remove failures.

## What is measured

| Question | Evidence |
|---|---|
| What should fit before training? | `memory_budget.json`: components and formulas; assumptions explicitly separate from measurements |
| What actually used VRAM? | Same-process CUDA high-water allocated/reserved stats; raw `nvidia-smi` before, after, 1 s samples |
| Did optimization receive gradients? | Finite/nonzero LoRA gradient event for each optimizer step, actual step count |
| Did trained tensors change? | Exact pre-training adapter snapshot vs final tensor deltas, not nonzero count |
| Does the saved adapter work? | Ordinary PEFT reload, full tensor round-trip, response-token likelihood changes, adapter-disabled restoration |
| Can this catch a successful no-op? | Two actual Soup DPO steps with `lr=0`; verification must reject changed-weight and functional checks |
| Is streaming backward plausible? | Same Soup/TRL DPO batches, same weights, nonzero B, all adapter gradients compared to resident control on T4 |
| Did the DPO reference stay fixed? | Before/after ref log-probs and reference-instance/ref-adapter metadata |
| Is there heldout improvement? | Preference accuracy, length-normalized diagnostic, scenario-group bootstrap, raw per-row scores |
| What does Soup think? | `doctor`, DPO `lint`, CLI preflight and `ship` outputs, arguments and exit codes |

The main training calls `DPOTrainerWrapper.setup` and the underlying actual TRL
trainer to add evidence hooks around training. It does not implement a substitute
DPO optimizer. CLI `soup train --dry-run` is also recorded. Raw outputs remain
unchanged in `*.raw.log`; timestamped copies are separate files. `soup ship`
uses transparent **measured offline scores**, not an external paid judge or
invented general-suite scores. Its four-question regression leg is labeled as a
tiny forced-choice sanity check and cannot justify production deployment.

## Data and limitations

The task permits synthetic data. `scripts/build_data.py` deterministically makes
500 fictional Russian preference pairs with explicit fictional support policies.
There are ten intents, five contexts per intent and ten noisy variants per
context. One context per intent is held out, keeping paraphrases in the same
split: 400 training rows / 100 heldout rows, only ten independent heldout groups.
Numbers and identifiers are fictional. No source email or real customer data is
published. See `data/provenance.json` for hashes and exact generation rules.

Shared response/policy templates across splits create severe shortcut risk.
The synthetic heldout test is an engineering diagnostic, not real-domain quality
validation. The verification script cannot detect bad preference labels,
memorization, every wrong-gradient failure, or production-only regressions.
The parity probe covers three DPO batches at this specific unquantized
model/config/stack/T4; it does not validate NF4 or other hardware. See
[source review](docs/SOURCE_REVIEW.md) for what Soup checks actually cover.

## Deliverables

- `reports/report.pdf`: maximum two pages, generated only from recorded facts.
- `evidence/<run-id>/`: raw timestamped logs, raw GPU output, versions, hashes,
  errors, measurements, verification output and final verdict.
- `scripts/verify_training.py`: standalone Part 2 verification (exit 2 for failed checks).
- `notebooks/takehome_t4.ipynb`: clean runnable notebook.
- `notebooks/takehome_t4_executed.ipynb`: downloaded Colab notebook with actual outputs and repair cells.
- [Final release](https://github.com/kenzhebekdanial2006/soup-dpo-takehome/releases/tag/t4-evidence-v1): full result archive including adapter weights and initialization snapshots.
- `config/soup.yaml` and `scripts/`: complete reproducible code.
- Report includes one paragraph on surprises/remaining concerns and AI use.

Failed attempts and precision diagnostics remain in `evidence/`; see
[experiment history](docs/EXPERIMENTS.md). The final report is regenerated from
the completed run, with a separately labeled post-run memory analysis.
Give the company access to this private repository before submitting its link.

## Local verification

```bash
python -m pytest -q
python -m ruff check scripts tests
```

Tests include no-op detection, changed outputs, wrong adapter key coverage,
nonfinite weights, group leakage, memory formulas, a tiny CPU PEFT model's actual
adapter update, and fail-closed reporting. They do not substitute for the T4 run.
