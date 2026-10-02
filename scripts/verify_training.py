"""Part 2: require changed tensors AND a working adapter after ordinary PEFT reload.

Exit 0 = learning/activation check passed; 2 = a substantive failure; 1 = runtime error.
This is NOT a deployment-quality or gradient-correctness proof.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from scripts.common import read_jsonl, sha256, utc_now, write_json
from scripts.scoring import accuracy, score_pairs


def compare_states(initial, trained):
    import torch

    if set(initial) != set(trained):
        return {"passed": False, "reason": "Adapter key sets differ",
                "missing": sorted(set(initial) - set(trained)),
                "unexpected": sorted(set(trained) - set(initial))}
    rows = []
    for key in sorted(initial):
        a, b = initial[key].float(), trained[key].float()
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            return {"passed": False, "reason": f"Shape/nonfinite failure: {key}"}
        delta = (b - a).norm().item()
        rows.append({"key": key, "delta_l2": delta,
                     "relative_delta": delta / max(a.norm().item(), 1e-12),
                     "changed": delta > 1e-6})
    b_rows = [r for r in rows if "lora_B" in r["key"]]
    fraction = sum(r["changed"] for r in b_rows) / len(b_rows) if b_rows else 0
    return {"passed": fraction >= 0.9, "changed_B_fraction": fraction,
            "changed_tensors": sum(r["changed"] for r in rows), "total_tensors": len(rows),
            "threshold_l2": 1e-6, "tensors": rows}


def functional_change(base, repeated_base, tuned):
    # Max changes in token-level logps, measured on IDENTICAL completion tokens.
    def deltas(first, second):
        if len(first) != len(second):
            raise ValueError("Probe count mismatch")
        result = []
        for a, b in zip(first, second, strict=True):
            if a["id"] != b["id"]:
                raise ValueError("Probe identity mismatch")
            changes = []
            for side in ["chosen", "rejected"]:
                x, y = a[side]["token_logps"], b[side]["token_logps"]
                if len(x) != len(y):
                    raise ValueError("Token boundary mismatch")
                changes.extend(abs(u - v) for u, v in zip(x, y, strict=True))
            result.append(max(changes))
        return result

    noise = max(deltas(base[:len(repeated_base)], repeated_base))
    threshold = max(1e-4, 20 * noise)
    changes = deltas(base, tuned)
    return {"passed": sum(d > threshold for d in changes) >= min(3, len(changes)),
            "baseline_repeat_noise": noise, "threshold": threshold,
            "changed_probes": sum(d > threshold for d in changes),
            "max_token_logp_delta": max(changes), "per_probe_max_delta": changes}


def main():
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="models/base")
    parser.add_argument("--initial", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default="data/heldout.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    adapter_file = Path(args.adapter) / "adapter_model.safetensors"
    initial_file = Path(args.initial) / "adapter_model.safetensors"
    trained, initial = load_file(str(adapter_file)), load_file(str(initial_file))
    if any(".inner." in key for key in trained):
        raise ValueError("Nonportable .inner. adapter keys detected")
    weight_check = compare_states(initial, trained)
    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=device, trust_remote_code=False,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    base_scores = score_pairs(base, tokenizer, rows)
    repeat_scores = score_pairs(base, tokenizer, rows[:4])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    loaded = get_peft_model_state_dict(model, adapter_name="default")
    coverage = set(loaded) == set(trained) and all(
        torch.equal(loaded[k].cpu().to(trained[k].dtype), trained[k]) for k in trained
    )
    if not coverage:
        raise ValueError("Saved adapter tensors did not round-trip into ordinary PEFT")
    warning_texts = [str(w.message) for w in caught]
    if any("missing" in w.lower() or "unexpected" in w.lower() for w in warning_texts):
        raise ValueError(f"Adapter loading warning: {warning_texts}")
    tuned_scores = score_pairs(model, tokenizer, rows)
    with model.disable_adapter():
        disabled = score_pairs(model, tokenizer, rows[:4])
    disabled_check = functional_change(base_scores[:len(disabled)], repeat_scores, disabled)
    activation = functional_change(base_scores, repeat_scores, tuned_scores)
    # Disabled adapter MUST reproduce base within baseline noise/1e-4.
    disabled_restores_base = disabled_check["max_token_logp_delta"] <= max(
        1e-4, 20 * disabled_check["baseline_repeat_noise"]
    )
    result = {
        "timestamp": utc_now(), "passed": weight_check["passed"] and activation["passed"] and disabled_restores_base,
        "base": args.base, "adapter_sha256": sha256(adapter_file),
        "initial_sha256": sha256(initial_file), "data_sha256": sha256(args.data),
        "tensor_delta": weight_check, "ordinary_peft_reload_exact": coverage,
        "warnings": warning_texts, "functional_change": activation,
        "adapter_disabled_restores_base": disabled_restores_base,
        "base_accuracy_sum": accuracy(base_scores), "tuned_accuracy_sum": accuracy(tuned_scores),
        "base_accuracy_mean": accuracy(base_scores, "margin_mean"),
        "tuned_accuracy_mean": accuracy(tuned_scores, "margin_mean"),
        "base_scores": base_scores, "tuned_scores": tuned_scores,
        "misses": ["Wrong preference labels", "Memorization and production regressions",
                   "Wrong-but-nonzero streamed gradients (separate parity test required)",
                   "Rare prompts or deployments not represented by these probes"],
    }
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ["base_scores", "tuned_scores", "tensor_delta"]}, indent=2))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
