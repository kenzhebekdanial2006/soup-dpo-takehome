"""Resolve one immutable base revision, then use ONLY that local snapshot."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import sha256, utc_now, write_json


def main():
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    snapshot_download(
        args.model, revision=revision, local_dir="models/base",
        allow_patterns=["*.json", "*.safetensors", "*.model", "merges.txt", "vocab.*"],
    )
    files = sorted(Path("models/base").glob("*.safetensors"))
    if not files:
        raise RuntimeError("No model weights downloaded")
    manifest = {
        "timestamp": utc_now(), "model_id": args.model, "requested_revision": args.revision,
        "resolved_revision": revision, "weights": {p.name: sha256(p) for p in files},
        "config_sha256": sha256("models/base/config.json"),
        "tokenizer_config_sha256": sha256("models/base/tokenizer_config.json"),
    }
    write_json(Path(args.evidence) / "model_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
