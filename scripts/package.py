"""Export a reviewable source ZIP or a run result ZIP without credentials/caches."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def package(output, include_weights=False):
    root = Path.cwd().resolve()
    output = Path(output).resolve()
    include = [root / name for name in ["README.md", "requirements.txt", "pyproject.toml", ".gitignore", ".github",
                                        "config", "data", "docs", "scripts", "tests", "notebooks", "reports", "evidence"]]
    if include_weights:
        include += [root / "outputs" / "adapter", root / "outputs" / "control_adapter"]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry in include:
            paths = [entry] if entry.is_file() else sorted(entry.rglob("*"))
            for path in paths:
                if not path.is_file() or path.resolve() == output:
                    continue
                relative = path.relative_to(root)
                if any(p in {"__pycache__", ".ipynb_checkpoints"} or p.startswith("checkpoint-") for p in relative.parts):
                    continue
                if path.suffix in {".pyc", ".zip", ".pt", ".pth", ".bin"}:
                    continue
                if not include_weights and path.suffix == ".safetensors":
                    continue
                archive.write(path, str(relative))
    print(f"Saved {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../soup-dpo-takehome.zip")
    parser.add_argument("--include-weights", action="store_true")
    args = parser.parse_args()
    package(args.output, args.include_weights)
