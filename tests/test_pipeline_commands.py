import os
import sys

import pytest

from scripts import run_t4


@pytest.mark.skipif(os.name != "posix", reason="Colab uses Linux CLI lookup semantics")
def test_pipeline_finds_cli_in_venv_without_activation(tmp_path, monkeypatch):
    bin_dir = tmp_path / "isolated-venv" / "bin"
    bin_dir.mkdir(parents=True)
    cli = bin_dir / "soup"
    cli.write_text(f'#!{sys.executable}\nprint("isolated Soup CLI found")\n', encoding="utf-8")
    cli.chmod(0o755)
    monkeypatch.setattr(run_t4.sys, "executable", str(bin_dir / "python"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    evidence = tmp_path / "evidence"
    assert run_t4.run_command(["soup"], evidence, "cli-lookup") == 0
    assert (evidence / "cli-lookup.raw.log").read_bytes() == b"isolated Soup CLI found\n"
