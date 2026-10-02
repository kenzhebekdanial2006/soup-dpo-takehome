# Local checks — NOT a T4 run

Python 3.11.9 on Windows, Soup core 0.75.2. Local pytest: **4 passed, 5 skipped**
(Torch/PEFT unavailable in this local lightweight environment). Ruff passes;
the real Soup schema accepts `config/soup.yaml`. The prepared report is two pages.
No training was performed and no adapter learning is claimed from these checks.

The first dependency-freeze attempt failed because uv-created virtual environments
do not automatically include pip. Its original failed command log remains here.
`uv pip freeze --python <environment>` then succeeded; both outputs are retained.
This local setup failure is not attributed to Soup or CUDA.

The Colab pipeline runs the ML-dependent tests with the actual installed training
stack before the real T4 parity, no-op control, and DPO run.
