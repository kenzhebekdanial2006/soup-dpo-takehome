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
