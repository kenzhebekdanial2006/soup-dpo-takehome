import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_notebook import main


def source_cells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main()
    notebook = json.loads(Path("notebooks/takehome_t4.ipynb").read_text())
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_python_313_kernel_bootstraps_311_training(tmp_path, monkeypatch):
    cells = source_cells(tmp_path, monkeypatch)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=b"test bootstrap output\n", returncode=0)

    scope = {"Path": Path, "root": tmp_path, "sys": SimpleNamespace(executable="/colab/python313"),
             "subprocess": SimpleNamespace(run=run, PIPE=-1, STDOUT=-2)}
    assert "assert (3, 10)" not in cells[0]
    exec(compile(cells[1], "bootstrap-cell", "exec"), scope)
    assert scope["PIPELINE_PYTHON"] == str((tmp_path / ".colab-venv/bin/python").resolve())
    assert calls[1][-3:] == ["python", "install", "3.11"]
    assert all(command[0] == scope["PIPELINE_PYTHON"] for command in calls[3:])
    exec(compile(cells[2], "pipeline-cell", "exec"), scope)
    assert calls[-1] == [scope["PIPELINE_PYTHON"], "-m", "scripts.run_t4"]
    assert len(list(Path("evidence").glob("*.raw.log"))) == 6


def test_failed_bootstrap_keeps_raw_failure_and_stops(tmp_path, monkeypatch):
    cells = source_cells(tmp_path, monkeypatch)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=b"original failure\n", returncode=7)

    scope = {"Path": Path, "root": tmp_path, "sys": SimpleNamespace(executable="/colab/python313"),
             "subprocess": SimpleNamespace(run=run, PIPE=-1, STDOUT=-2)}
    with pytest.raises(RuntimeError, match="colab-install-uv failed"):
        exec(compile(cells[1], "bootstrap-cell", "exec"), scope)
    assert len(calls) == 1
    assert Path("evidence/colab-install-uv.raw.log").read_bytes() == b"original failure\n"
