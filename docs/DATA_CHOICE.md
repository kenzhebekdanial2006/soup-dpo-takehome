# Why a mock dataset

The task explicitly permits a synthetic mock when a suitable dataset is unavailable.
These public alternatives were checked on 2026-10-02:

- [customer-support-dpo-100k](https://huggingface.co/datasets/stindardlogic/customer-support-dpo-100k)
  is itself synthetic and its displayed examples are English. Using it would
  introduce translation and potentially unknown-company policy assumptions.
- [Russian detox DPO](https://huggingface.co/datasets/r1char9/detox-dpo-dataset)
  targets detoxification rather than support-ticket handling.
- [support-json-ru](https://huggingface.co/datasets/A11Sunday/support-json-ru)
  is relevant Russian support data, but its target is a structured JSON decision,
  not supplied chosen/rejected DPO pairs.

We did not establish a suitable openly licensed **real Russian support-preference**
corpus within this search. This is not a claim that none exists. A transparent
small mock was chosen for the engineering experiment, with the task's permission.
We did not translate, copy, or silently relabel any of these datasets.

The mock's labels and templates were authored with Codex. Positive replies follow
the fictional policy supplied in each prompt; negative replies make an
unsupported promise, ask for a credential, skip an investigation, or take an
unauthorized action. Identifiers begin `TEST-`; all tickets are fictional.
The generation source and data hashes are included, and template leakage is
explicitly a reason not to deploy this model.
