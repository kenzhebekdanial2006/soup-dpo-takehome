"""Post-run memory analysis. Preserve the original estimate and raw GPU samples."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from scripts.common import read_jsonl, utc_now, write_json
from scripts.make_report import read
from scripts.memory_budget import estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    directory = Path(args.evidence)
    trained = read(directory, "training_result.json")
    if not trained:
        raise ValueError("A measured main training result is required")
    budget = read(directory, "memory_budget.json")
    audit = read(directory, "data_audit.json")
    training_ids = {row["id"] for row in read_jsonl("data/train.jsonl")}
    lengths = [row[side]["full_tokens"] for row in audit["row_stats"]
               if row["id"] in training_ids for side in ("chosen", "rejected")]
    # A labeled post-hoc sensitivity calculation; never replace the pre-run estimate.
    config = read(Path("config"), "model_shape_expected.json")
    sensitivity = estimate(config, sequence=max(lengths), batch=budget["shape"]["pair_batch"])
    command = read(directory, "train.command.json")
    start, end = [datetime.fromisoformat(command[key]) for key in ("started", "finished")]
    samples = []
    with (directory / "nvidia-smi.samples.raw.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            timestamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            if start <= timestamp <= end:
                samples.append(int(row[3].strip()))
    gi = 1024**3
    result = {
        "kind": "POST-RUN ANALYSIS; original memory_budget.json remains unchanged",
        "timestamp": utc_now(), "training_command_window": {"started": command["started"], "finished": command["finished"]},
        "pre_run_plan_GiB": budget["planned_peak_upper_GiB"],
        "training_allocated_peak_GiB": trained["training_peak_allocated_bytes"] / gi,
        "training_reserved_peak_GiB": trained["training_peak_reserved_bytes"] / gi,
        "training_window_nvidia_smi_peak_GiB": max(samples) / 1024 if samples else None,
        "training_window_gpu_samples": len(samples),
        "audited_training_full_tokens": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
        "post_hoc_length_sensitivity_plan_GiB": sensitivity["planned_peak_upper_GiB"],
        "post_hoc_length_sensitivity_delta_GiB": sensitivity["planned_peak_upper_GiB"] - budget["planned_peak_upper_GiB"],
        "interpretation": [
            "The pre-run formula assumed padded length 512; audited real sequences are substantially shorter.",
            "Length sensitivity is calculated after the run, not a revised pre-run prediction or measured allocation.",
            "Allocated, reserved and driver-visible peaks measure different things and need not occur simultaneously.",
            "CUDA context and cached blocks contribute to nvidia-smi/reserved, but not to torch live allocated bytes.",
            "The fixed 1 GiB workspace/fragmentation allowance is a planning assumption, not a hard upper bound.",
            "Only the main training command window is sampled here; the two-model parity probe is excluded.",
            "Audited full token lengths are a tokenizer diagnostic; exact TRL runtime padding may differ by EOS handling.",
            "One-second GPU sampling can miss brief peaks; no single causal attribution is proved.",
        ],
    }
    write_json(directory / "memory_observed.json", result)
    print(result)


if __name__ == "__main__":
    main()
