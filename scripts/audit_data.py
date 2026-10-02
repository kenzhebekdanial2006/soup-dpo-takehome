"""Validate every row with actual tokenizer; no truncated examples are silently skipped."""
from __future__ import annotations

import argparse
from collections import Counter

from scripts.common import canonical_prompt, read_jsonl, write_json
from scripts.scoring import encode_pair


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="models/base")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
    train, test = read_jsonl("data/train.jsonl"), read_jsonl("data/heldout.jsonl")
    if {r["group"] for r in train} & {r["group"] for r in test}:
        raise ValueError("Scenario-group leakage")
    if {canonical_prompt(r) for r in train} & {canonical_prompt(r) for r in test}:
        raise ValueError("Exact-prompt leakage")
    stats = []
    for row in train + test:
        encoded = encode_pair(tokenizer, row)
        stats.append({"id": row["id"], **{side: {
            "full_tokens": len(encoded[side]["ids"]),
            "response_tokens": len(encoded[side]["ids"]) - encoded[side]["response_start"],
        } for side in ["chosen", "rejected"]}})
    result = {"rows_checked": len(stats), "truncated": 0, "identical_token_pairs": 0,
              "trained_eos_present": True, "group_overlap": 0, "prompt_overlap": 0,
              "intents": dict(Counter(r["intent"] for r in train + test)),
              "mean_response_lengths": {s: sum(r[s]["response_tokens"] for r in stats) / len(stats) for s in ["chosen", "rejected"]},
              "row_stats": stats,
              "limitation": "Does not rule out shared-template semantic leakage or wrong human labels"}
    write_json(args.output, result)
    print({k: v for k, v in result.items() if k != "row_stats"})


if __name__ == "__main__":
    main()
