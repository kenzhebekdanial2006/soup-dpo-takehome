# AI assistance disclosure

Codex authored this project at the applicant's request, including the synthetic
ticket templates, Python scripts, Colab notebook, tests and report generation.
It inspected the task and the pinned Soup release rather than relying on CLI
names or remembered behavior. There is no hidden LLM labeler or external judge.

Accepted suggestions: compare against exact adapter initialization; verify
ordinary PEFT reload and adapter-disabled outputs; run an actual zero-learning-rate
DPO control; compare streamed and resident backward paths with nonzero LoRA B;
record raw command output, dependency versions and an immutable model revision.

Source verification changed the initial approach: doctor is a chat/SFT tool and
rejects DPO input, so preference lint and a clearly labeled chat projection are
both used. The GPU recipe uses fp16, and the reference has no second base-model
copy. A SHIP result is not treated as proof of learning or deployment quality.

Local checks and real T4 results are recorded in separate evidence directories.
No successful GPU training, measured VRAM, or applicant review is claimed before
it happens. The applicant should personally read the code and inspect the actual
T4 evidence, then add a short factual note below about what they accepted,
changed, reran or disagreed with. Do not pre-write answers for the live interview.

Applicant's independent review: **not yet recorded**.

Colab bootstrap corrections: the first recipe assumed a Python <=3.12 notebook
kernel. It was replaced with an isolated Python 3.11 environment. The subsequent
recipe incorrectly resolved the Linux venv interpreter symlink, launching uv's
managed base Python and triggering pip's externally-managed-environment error.
The venv entry point is now preserved, checked before package installation, and
covered by a Linux regression test. Retry logs are retained separately. These
corrections do not establish a successful GPU training run.

The first real T4 pipeline reached streamed/resident backward probing but failed
while recording token lengths: the script expected older TRL field names.
Both parity metadata and training guards now use the pinned TRL 0.29 schema,
verified against its actual preference collator. The failed attempt's raw logs
are retained. Colab pipeline output is now explicitly piped to the notebook.

Actual T4 diagnostics then found gradient disagreement under default attention,
with similar variation even between two resident repeats at identical loss.
Disabling fp16 GEMM reduced-precision reductions alone did not resolve it.
Deterministic SDPA math plus deterministic GEMM produced bit-identical gradients
for all 112 trainable tensors on three real DPO batches and zero resident-repeat
noise, using the original tolerances. The pipeline now uses those settings for
parity, training and scoring. This does not establish a streaming defect, and
the failed/default-kernel probes are retained alongside the passing diagnostic.

The lr=0 control completed two real optimizer steps with byte-identical adapter
weights, exposing another invalid comparison: reference probabilities were
captured before Accelerate's AMP forward wrapper and compared to its FP32-converted
outputs after training. The initial reference probe now runs at on_train_begin,
after preparation and before updates; both probes record their phase and dtype.
The reference-stability threshold remains unchanged.

Codex completed the real T4 run `t4-20261002T203632Z` in the applicant's open
Colab, downloaded its executed notebook and result ZIP, verified raw-log hashes,
and generated the final two-page report. Training, saved-adapter verification,
unchanged reference and the deliberately rejected zero-LR control are recorded;
all 18 engineering tests passed on T4. Post-run memory sensitivity was added
locally and clearly labeled; the original estimate, raw logs and failed attempts
were preserved. Soup's numeric SHIP was rejected as a deployment decision because
quality evidence is synthetic and the general regression set has four questions.
