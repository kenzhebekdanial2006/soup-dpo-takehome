"""Response-only log probabilities; deterministic fp16 base and FP32 reductions."""
from __future__ import annotations


def encode_pair(tokenizer, row, max_length=512):
    prefix = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    result = {}
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    for side in ["chosen", "rejected"]:
        full = tokenizer.apply_chat_template(row["prompt"] + row[side], tokenize=False)
        if not full.startswith(prefix):
            raise ValueError(f"Chat template prefix mismatch: {row.get('id')}")
        ids = tokenizer.encode(full, add_special_tokens=False)
        if ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError(f"BPE boundary changed at response: {row.get('id')}")
        if len(prefix_ids) > max_length // 2 or len(ids) > max_length:
            raise ValueError(f"Would truncate or mask response: {row.get('id')}")
        response_ids = ids[len(prefix_ids):]
        if not response_ids or tokenizer.eos_token_id not in response_ids:
            raise ValueError(f"Response has no trained EOS: {row.get('id')}")
        result[side] = {"ids": ids, "response_start": len(prefix_ids)}
    if result["chosen"]["ids"] == result["rejected"]["ids"]:
        raise ValueError(f"Identical tokenized pair: {row.get('id')}")
    return result


def response_score(model, encoding):
    import torch

    device = next(p.device for p in model.parameters() if p.device.type != "meta")
    ids = torch.tensor([encoding["ids"]], device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        start = encoding["response_start"]
        # First completion token is predicted from the final prompt token.
        selected = logits[0, start - 1:-1].float().log_softmax(-1)
        targets = ids[0, start:]
        token_lp = selected.gather(-1, targets[:, None]).squeeze(-1)
        if not torch.isfinite(token_lp).all():
            raise ValueError("Non-finite response probabilities")
        return {"sum": token_lp.sum().item(), "mean": token_lp.mean().item(),
                "tokens": len(targets), "token_logps": token_lp.cpu().tolist()}


def score_pairs(model, tokenizer, rows):
    model.eval()
    scores = []
    for row in rows:
        encoded = encode_pair(tokenizer, row)
        c = response_score(model, encoded["chosen"])
        r = response_score(model, encoded["rejected"])
        scores.append({"id": row["id"], "group": row["group"],
                       "chosen": c, "rejected": r,
                       "margin_sum": c["sum"] - r["sum"],
                       "margin_mean": c["mean"] - r["mean"]})
    return scores


def accuracy(scores, axis="margin_sum"):
    return sum(row[axis] > 0 for row in scores) / len(scores)
