"""Independent first-order estimate; not Soup's empirical formula or a measurement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.common import sha256, utc_now, write_json


def estimate(c, sequence=512, batch=1, rank=8, buffers=2):
    h, i, n, v = (c[k] for k in ["hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size"])
    heads, kv_heads = c["num_attention_heads"], c["num_key_value_heads"]
    kv = kv_heads * (h // heads)
    # q,k,v biases, two RMSNorm weights, no o/MLP bias: Qwen2-specific.
    layer_params = 2 * h * h + 2 * h * kv + 3 * h * i + 3 * h + 2 * kv
    tied = c["tie_word_embeddings"]
    params = n * layer_params + h * v * (1 if tied else 2) + h
    adapter_params = n * rank * ((h + h) + (h + kv))  # q_proj and v_proj
    rows = 2 * batch  # concatenated chosen + rejected, NOT gradient accumulation
    components = {
        "decoder_pool": buffers * layer_params * 2,
        "embedding_or_large_vocab_slot": h * v * 2,
        "final_norm": h * 2,
        "adapter_weights_fp32": adapter_params * 4,
        "adapter_gradients_fp32": adapter_params * 4,
        "adam_m_and_v_fp32": adapter_params * 8,
        "frozen_reference_adapter_fp32_allowance": adapter_params * 4,
        "checkpoint_boundary_activations": rows * sequence * h * n * 2,
        "one_recomputed_layer_intermediates": rows * sequence * (8 * h + 3 * i) * 2,
        "policy_logits_fp16": rows * sequence * v * 2,
        "policy_logsoftmax_and_backward_fp32_planning_allowance": rows * sequence * v * 12,
        "reference_logit_peak_allowance": rows * sequence * v * 6,
        "cuda_context_workspace_fragmentation_allowance": 1024**3,
    }
    # Reference runs sequentially without grads. Charge max, never the sum of two models.
    common = sum(val for key, val in components.items()
                 if key not in {"policy_logits_fp16", "policy_logsoftmax_and_backward_fp32_planning_allowance", "reference_logit_peak_allowance"})
    policy = components["policy_logits_fp16"] + components["policy_logsoftmax_and_backward_fp32_planning_allowance"]
    reference = components["reference_logit_peak_allowance"]
    upper = common + max(policy, reference)
    return {
        "kind": "PRE-RUN ESTIMATE; not measured", "units": "bytes; GiB = bytes / 2**30",
        "shape": {"S": sequence, "pair_batch": batch, "forward_rows": rows, "h": h, "i": i, "layers": n, "vocab": v},
        "base_parameters": params, "lora_parameters": adapter_params,
        "base_resident_fp16_bytes": params * 2,
        "host_decoder_store_fp16_bytes": n * layer_params * 2,
        "components": components, "planned_peak_upper_bytes": upper,
        "planned_peak_upper_GiB": upper / 1024**3,
        "assumptions": [
            "Unquantized fp16 frozen base; T4 sm75 has no native bf16; adapters FP32",
            "28 layers, tied embedding/head for chosen Qwen; exactly q_proj/v_proj rank 8",
            "Two decoder buffers and one tied resident embedding matrix; no full base on GPU",
            "Reference uses adapter disabled or frozen initial ref adapter, no second base model",
            "Budget includes one extra frozen FP32 reference adapter (no gradient/Adam states)",
            "Reference passes may overlap policy activations: counted activation common plus max logits",
            "Adam states only for LoRA; no frozen-weight optimizer states; accumulation does not multiply batch",
            "14 bytes/logit is a conservative allowance, not an independently measured DPO constant",
            "SDPA, use_cache=false; a fallback to eager attention adds quadratic storage",
            "Host memory also needs download/sharding staging and Python overhead; store size is not host peak",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="models/base/config.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = estimate(json.loads(Path(args.model_config).read_text(encoding="utf-8")))
    result.update(timestamp=utc_now(), model_config_path=args.model_config,
                  model_config_sha256=sha256(args.model_config))
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
