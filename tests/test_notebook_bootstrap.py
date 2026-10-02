import json
import os
import subprocess
import sys
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

    class Process:
        stdout = ["live pipeline log\n"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self):
            return 0

    def popen(command, **kwargs):
        calls.append(command)
        return Process()

    scope = {"Path": Path, "root": tmp_path, "sys": SimpleNamespace(executable="/colab/python313"),
             "subprocess": SimpleNamespace(run=run, Popen=popen, PIPE=-1, STDOUT=-2)}
    assert "assert (3, 10)" not in cells[0]
    exec(compile(cells[1], "bootstrap-cell", "exec"), scope)
    assert scope["PIPELINE_PYTHON"] == str(tmp_path / ".colab-venv/bin/python")
    assert calls[1][-3:] == ["python", "install", "3.11"]
    assert all(command[0] == scope["PIPELINE_PYTHON"] for command in calls[3:])
    exec(compile(cells[2], "pipeline-cell", "exec"), scope)
    assert calls[-1] == [scope["PIPELINE_PYTHON"], "-m", "scripts.run_t4"]
    assert len(list(Path("evidence").rglob("*.raw.log"))) == 7


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
    assert next(Path("evidence").rglob("colab-install-uv.raw.log")).read_bytes() == b"original failure\n"


def test_venv_symlink_is_not_dereferenced(tmp_path, monkeypatch):
    cells = source_cells(tmp_path, monkeypatch)
    venv_python = tmp_path / ".colab-venv/bin/python"
    base_python = tmp_path / "managed-python/bin/python3.11"
    original_resolve = Path.resolve
    calls = []

    def resolve(path, *args, **kwargs):
        # Simulate the Linux symlink even on Windows without symlink privileges.
        if path == venv_python:
            return base_python
        return original_resolve(path, *args, **kwargs)

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == str(base_python):
            return SimpleNamespace(stdout=b"error: externally-managed-environment\n", returncode=1)
        return SimpleNamespace(stdout=b"venv active\n", returncode=0)

    monkeypatch.setattr(Path, "resolve", resolve)
    scope = {"Path": Path, "root": tmp_path, "sys": SimpleNamespace(executable="/colab/python313"),
             "subprocess": SimpleNamespace(run=run, PIPE=-1, STDOUT=-2)}
    exec(compile(cells[1], "bootstrap-cell", "exec"), scope)
    assert all(command[0] == str(venv_python) for command in calls[3:])
    assert "sys.prefix != sys.base_prefix" in calls[3][-1]


def test_retry_keeps_previous_raw_failure(tmp_path, monkeypatch):
    cells = source_cells(tmp_path, monkeypatch)
    outputs = iter([b"first failure\n", b"second failure\n"])

    def run(command, **kwargs):
        return SimpleNamespace(stdout=next(outputs), returncode=1)

    scope = {"Path": Path, "root": tmp_path, "sys": SimpleNamespace(executable="/colab/python313"),
             "subprocess": SimpleNamespace(run=run, PIPE=-1, STDOUT=-2)}
    for _ in range(2):
        with pytest.raises(RuntimeError, match="colab-install-uv failed"):
            exec(compile(cells[1], "bootstrap-cell", "exec"), scope)
    assert {path.read_bytes() for path in Path("evidence").rglob("*.raw.log")} == {
        b"first failure\n", b"second failure\n"}


@pytest.mark.skipif(os.name != "posix" or sys.version_info[:2] != (3, 11),
                    reason="Exercises the real Linux Python 3.11 symlink used by Colab")
def test_bootstrap_check_runs_inside_real_linux_venv(tmp_path, monkeypatch):
    import venv

    cells = source_cells(tmp_path, monkeypatch)
    venv.EnvBuilder(with_pip=False, symlinks=True).create(tmp_path / ".colab-venv")
    assert (tmp_path / ".colab-venv/bin/python").is_symlink()

    def run(command, **kwargs):
        if command[1] == "-c" and "sys.prefix != sys.base_prefix" in command[2]:
            return subprocess.run(command, **kwargs)
        return SimpleNamespace(stdout=b"dependency installs skipped in regression test\n", returncode=0)

    scope = {"Path": Path, "root": tmp_path, "sys": SimpleNamespace(executable=sys.executable),
             "subprocess": SimpleNamespace(run=run, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)}
    exec(compile(cells[1], "bootstrap-cell", "exec"), scope)
