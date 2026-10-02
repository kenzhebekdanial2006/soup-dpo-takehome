"""Create a small Colab notebook; checked in as an ordinary ipynb artifact."""
import hashlib
from pathlib import Path

from scripts.common import write_json


def cell(kind, source):
    result = {"cell_type": kind, "id": hashlib.sha256((kind + source).encode()).hexdigest()[:12],
              "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    cells = [
        cell("markdown", """# Soup DPO: evidence on a free Colab T4
Select **Runtime → Change runtime type → T4 GPU**, then Run all.
The notebook automatically creates Python 3.11 for Soup, even on Colab Python 3.13.
Upload `soup-dpo-takehome.zip` when asked. No paid services or secret tokens.
The run records errors and downloads a result ZIP even after failure.
Save the executed notebook with outputs too. A successful run does not imply SHIP.
The source ZIP is excluded from the returned archive; model cache is never exported.
"""),
        cell("code", """import os, sys, subprocess, zipfile
from pathlib import Path
from google.colab import files
print('Colab notebook kernel:', sys.version)
uploaded = files.upload()
assert len(uploaded) == 1, 'Upload only soup-dpo-takehome.zip'
input_zip = next(iter(uploaded))
root = Path('/content/soup-dpo-takehome')
root.mkdir(exist_ok=True)
with zipfile.ZipFile(input_zip) as archive:
    for member in archive.infolist():
        destination = (root / member.filename).resolve()
        assert destination.is_relative_to(root.resolve()), 'Unsafe archive path'
    archive.extractall(root)
os.chdir(root)
assert Path('scripts/run_t4.py').exists()
print('Project:', root)
"""),
        cell("code", """# Colab's notebook kernel can be Python 3.13; Soup runs in isolated Python 3.11.
Path('evidence').mkdir(exist_ok=True)
def bootstrap(command, name):
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    Path(f'evidence/{name}.raw.log').write_bytes(completed.stdout)
    print(completed.stdout.decode('utf-8', errors='replace'), flush=True)
    if completed.returncode:
        raise RuntimeError(f'{name} failed (exit {completed.returncode}); raw log retained.')
bootstrap([sys.executable, '-m', 'pip', 'install', 'uv==0.9.3'], 'colab-install-uv')
bootstrap([sys.executable, '-m', 'uv', 'python', 'install', '3.11'], 'colab-install-python')
bootstrap([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', '--seed', '.colab-venv'], 'colab-create-venv')
PIPELINE_PYTHON = str((root / '.colab-venv/bin/python').resolve())
bootstrap([PIPELINE_PYTHON, '-m', 'pip', 'install', 'torch==2.6.0+cu124',
           '--extra-index-url', 'https://download.pytorch.org/whl/cu124'], 'colab-install-torch')
bootstrap([PIPELINE_PYTHON, '-m', 'pip', 'install', '-r', 'requirements.txt',
           '-c', 'config/t4-constraints.txt', '--extra-index-url',
           'https://download.pytorch.org/whl/cu124'], 'colab-install-soup')
bootstrap([PIPELINE_PYTHON, '-c', 'import sys,torch; print(sys.version); print(torch.__version__,torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'], 'colab-check-runtime')
"""),
        cell("code", """# Each stage runs in a fresh process; no need to import the newly installed ML stack here.
# Streaming parity, lr=0 control and the 400-row DPO run all execute on this T4.
result = subprocess.run([PIPELINE_PYTHON, '-m', 'scripts.run_t4'])
print('Pipeline exit code:', result.returncode)
print('All available logs and failure evidence are retained.')
"""),
        cell("code", """import json
latest_path = Path('evidence/latest_run.json')
if latest_path.exists():
    latest = json.loads(latest_path.read_text())
    print(latest)
    verdict = Path(latest['path']) / 'verdict.json'
    if verdict.exists():
        print(verdict.read_text())
print('Read reports/report.pdf and raw logs; do not infer learning from loss alone.')
"""),
        cell("code", """# Export even a failed attempt. Includes adapter + exact initialization when available.
export_path = '/content/soup-dpo-results.zip'
export_python = globals().get('PIPELINE_PYTHON', sys.executable)
subprocess.run([export_python, '-m', 'scripts.package', '--include-weights', '--output', export_path], check=True)
files.download(export_path)
"""),
        cell("markdown", """## Before submission
Download this notebook via **File → Download → Download .ipynb**, retaining outputs.
Keep every failed attempt. Extract `soup-dpo-results.zip` into the local project,
commit the evidence/report, and share the adapter archive privately with the reviewer.
Record your own review in `docs/AI_USAGE.md`; no personal review is pre-claimed.
The main report is exactly two pages. The final deployment call remains DON'T SHIP
for synthetic-only quality evidence, even if Soup's numeric ship gate passes.
"""),
    ]
    write_json(Path("notebooks/takehome_t4.ipynb"), {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"colab": {"name": "Soup DPO T4 take-home"}, "accelerator": "GPU",
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "cells": cells,
    })


if __name__ == "__main__":
    main()
