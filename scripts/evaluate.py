"""Heldout preference diagnostics and transparent Soup offline ship evidence."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.common import bootstrap_mean_ci, read_jsonl, sha256, utc_now, write_json
from scripts.scoring import accuracy, score_pairs


def summarize_groups(base, tuned):
    groups = defaultdict(list)
    for b, t in zip(base, tuned, strict=True):
        if b["id"] != t["id"]:
            raise ValueError("Evaluation rows changed")
        groups[b["group"]].append(float(t["margin_sum"] > 0) - float(b["margin_sum"] > 0))
    means = [sum(values) / len(values) for values in groups.values()]
    return {"independent_groups": len(means), "paired_accuracy_delta": sum(means) / len(means),
            "cluster_bootstrap_95_ci": bootstrap_mean_ci(means),
            "per_group_delta": {g: sum(v) / len(v) for g, v in groups.items()}}


def main():
    from scripts.precision import configure_deterministic_math

    precision = configure_deterministic_math()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--verification", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    evidence = Path(args.evidence)
    verification = json.loads(Path(args.verification).read_text(encoding="utf-8"))
    summary = summarize_groups(verification["base_scores"], verification["tuned_scores"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        "models/base", device_map=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa", trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained("models/base", trust_remote_code=False)
    general = read_jsonl("data/general.jsonl")
    general_base = score_pairs(base, tokenizer, general)
    model = PeftModel.from_pretrained(base, "outputs/adapter")
    general_tuned = score_pairs(model, tokenizer, general)
    # This measures forced-choice likelihood, NOT generated answers or lm-eval.
    ship_evidence = {
        "task": {"mode": "metric", "base": verification["base_accuracy_sum"],
                 "tuned": verification["tuned_accuracy_sum"]},
        "benchmarks": {"tiny_general_forced_choice_4_NOT_mmlu": {
            "base": accuracy(general_base), "tuned": accuracy(general_tuned)}},
        "provenance": {"timestamp": utc_now(), "config_sha": sha256("config/soup.yaml"),
                       "verification_sha": sha256(args.verification),
                       "task_metric": "Response-only sum-logp preference accuracy on synthetic heldout",
                       "general_metric": "4-question response likelihood, not broad forgetting evidence"},
    }
    write_json(evidence / "ship_evidence.json", ship_evidence)
    write_json(evidence / "evaluation.json", {
        "timestamp": utc_now(), **summary, "general_base": general_base,
        "precision": precision,
        "general_tuned": general_tuned, "length_normalized_accuracy": {
            "base": verification["base_accuracy_mean"], "tuned": verification["tuned_accuracy_mean"]},
        "limitation": "Synthetic shared templates and 10 independent groups cannot establish real-ticket quality",
    })
    print(summary)


if __name__ == "__main__":
    main()
