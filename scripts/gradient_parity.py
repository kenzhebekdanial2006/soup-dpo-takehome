"""Compare actual Soup/TRL DPO loss + every trainable gradient to a resident control.

Nonzero LoRA B is intentional: a zero-initialized B would make every A gradient zero,
leaving half the backward path untested. This probe does NOT update the training adapter.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import read_jsonl, utc_now, write_json


def main():
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from soup_cli.config.loader import load_config
    from soup_cli.trainer.dpo import DPOTrainerWrapper
    from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/soup.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This parity evidence must be collected on T4")
    cfg = load_config(args.config)
    streamed = DPOTrainerWrapper(cfg, device="cuda", report_to="none")
    resident_cfg = cfg.model_copy(deep=True)
    resident_cfg.training.stream_layers = False
    resident_cfg.output = "outputs/parity_resident"
    resident = DPOTrainerWrapper(resident_cfg, device="cuda", report_to="none")
    records = read_jsonl("data/train.jsonl")[:3]
    try:
        streamed.setup({"train": records, "val": []})
        resident.setup({"train": records, "val": []})
        state = {k: v.detach().clone().cpu() for k, v in get_peft_model_state_dict(streamed.model).items()}
        generator = torch.Generator().manual_seed(917)
        for key, value in state.items():
            if "lora_B" in key:
                state[key] = torch.randn(value.shape, generator=generator, dtype=torch.float32) * 0.005
        for wrapper in [streamed, resident]:
            set_peft_model_state_dict(wrapper.model, state, adapter_name="default")
            align_trainable_dtype_for_fp16(wrapper.model, fp16=True, bf16=False)
            wrapper.model.config.use_cache = False
            wrapper.model.train()
        resident.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Same prepared TRL input for both paths, not two independent tokenizations.
        probes = []
        for index in range(3):
            item = streamed.trainer.train_dataset[index]
            batch = streamed.trainer.data_collator([item])
            batch = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in batch.items()}
            outputs = []
            for wrapper in [streamed, resident]:
                wrapper.model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = wrapper.trainer.compute_loss(wrapper.model, dict(batch))
                loss.backward()
                grads = {}
                for name, parameter in wrapper.model.named_parameters():
                    if parameter.requires_grad:
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise RuntimeError(f"Missing/nonfinite gradient {name}")
                        grads[name] = parameter.grad.detach().float().cpu().clone()
                outputs.append((loss.detach().float().item(), grads))
            (stream_loss, sg), (resident_loss, rg) = outputs
            if set(sg) != set(rg):
                raise RuntimeError("Streamed/resident trainable key mismatch")
            comparisons = []
            for key in sorted(sg):
                a, b = sg[key], rg[key]
                abs_error = (a - b).abs().max().item()
                relative_l2 = (a - b).norm().item() / max(b.norm().item(), 1e-12)
                close = torch.allclose(a, b, rtol=1e-2, atol=1e-6)
                if b.norm().item() > 1e-7:
                    close = close and relative_l2 <= 0.01
                comparisons.append({"key": key, "passed": bool(close),
                                    "max_abs_error": abs_error, "relative_l2": relative_l2,
                                    "resident_norm": b.norm().item()})
            loss_close = abs(stream_loss - resident_loss) <= 2e-4 + 2e-3 * abs(resident_loss)
            probes.append({"index": index, "prompt_tokens": len(item["prompt_input_ids"]),
                           "chosen_tokens": len(item["chosen_input_ids"]),
                           "rejected_tokens": len(item["rejected_input_ids"]),
                           "stream_loss": stream_loss, "resident_loss": resident_loss,
                           "loss_close": loss_close, "gradients": comparisons,
                           "passed": loss_close and all(c["passed"] for c in comparisons)})
        result = {"timestamp": utc_now(), "passed": all(p["passed"] for p in probes),
                  "gpu": torch.cuda.get_device_name(0), "probes": probes,
                  "quantization": cfg.training.quantization,
                  "tolerances": {"gradient_rtol": 0.01, "gradient_atol": 1e-6,
                                 "loss_rtol": 0.002, "loss_atol": 2e-4},
                  "scope": "Three real DPO batches, nonzero B, this base/config/stack/T4 only",
                  "misses": "Later-step buffer faults, NF4 path, other shapes/models/hardware"}
        write_json(args.output, result)
        print({"passed": result["passed"], "probes": len(probes), "output": str(Path(args.output))})
        raise SystemExit(0 if result["passed"] else 2)
    finally:
        streamed._close_stream_runtime()
        resident._close_stream_runtime()


if __name__ == "__main__":
    main()
