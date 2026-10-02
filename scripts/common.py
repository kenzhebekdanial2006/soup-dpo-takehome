from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n"
    )


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_prompt(row):
    return json.dumps(row["prompt"], ensure_ascii=False, sort_keys=True)


def bootstrap_mean_ci(values, seed=731, repeats=2000):
    """Resample independent units, NOT paraphrase rows within the same unit."""
    import random

    if not values:
        raise ValueError("No independent units")
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(repeats)
    )
    return [samples[int(0.025 * repeats)], samples[int(0.975 * repeats)]]
