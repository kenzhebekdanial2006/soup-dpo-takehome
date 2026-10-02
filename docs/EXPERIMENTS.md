# Recorded experiments

All IDs use UTC. Failed raw logs and command records remain unchanged. The
executed Colab notebook contains repair cells and their actual output;
the clean notebook is the entry point for a fresh rerun.

| Evidence directory | Outcome and correction |
|---|---|
| `colab-bootstrap-20261003`, `bootstrap-1790971298188378023` | Colab dependency mismatch, then resolving a venv symlink launched an externally managed base Python. Use isolated Python 3.11 and preserve the venv entry point. |
| `t4-20261002T200642Z`, `t4-20261002T201429Z` | Actual backward probe ran, but recording used obsolete TRL field names. Adopt the verified TRL 0.29 keys `prompt_ids`, `chosen_ids`, `rejected_ids`. |
| `t4-20261002T201713Z` | Default attention: matching losses, disagreeing gradients. A failed parity result. |
| `diagnostic-20261002T202330Z` | Disabling reduced-precision fp16 GEMM reductions alone did not fix parity; identical resident repeats also varied. |
| `diagnostic-20261002T202625Z` | Deterministic SDPA math and GEMM: zero gradient error and repeat noise on all 112 tensors across three batches, at unchanged tolerances. Numerical variability was isolated; a streaming defect was not established. |
| `t4-20261002T203145Z` | Parity passed; real zero-LR control retained identical weights. Reference check incorrectly compared pre-AMP and post-AMP outputs. Initial probe moved after Accelerate preparation, before updates; threshold unchanged. |
| `t4-20261002T203632Z` | Full T4 pipeline completed. 18 tests passed. Zero-LR control correctly rejected tensor and functional change; main training performed 100 optimizer steps. All 112 saved tensors changed; ordinary PEFT reload matched exactly. All 100 heldout probes changed with zero repeat noise and adapter-disabled restoration. Reference delta zero. |

Final synthetic preference accuracy: sum-based 0.87 to 1.00; length-normalized
0.90 to 1.00. Paired delta +0.13, group-bootstrap 95% CI [0.00, 0.36], ten
independent heldout groups. Four general forced-choice questions remained 4/4.
Soup's numeric gate returned SHIP. Final call remains DON'T SHIP: shared synthetic
templates and four questions do not establish deployment quality.

The original pre-run memory plan is preserved. `memory_observed.json` is explicitly
post-run analysis, added locally after downloading the original result archive.
It restricts samples to the main training command, excluding the two-model parity
probe: 2.017 GiB live allocated, 6.439 GiB reserved, 6.651 GiB sampled device
peak, against a 3.815 GiB plan. Length-only sensitivity at 235 instead of 512
tokens gives 2.632 GiB; it is not a replacement prediction. The fixed 1 GiB
context/workspace/cache allowance underestimated physical footprint. Separate
peak counters and one-second samples do not prove a unique causal breakdown.

Soup's optional near-duplicate check skipped because datasketch was absent;
group and exact-prompt checks were performed. Chat doctor flagged missing
generation markers (MINOR); actual DPO collator masking has a separate passing
test. Real-ticket annotation, generated-response review, broader regressions
and more seeds are still required before deployment.

Final source adds automatic post-run memory analysis for future runs; the recorded
final run preceded that orchestration addition. Source changes and derived report
are distinguished from unedited raw logs. Codex operated Colab; the applicant's
independent review is not claimed.
