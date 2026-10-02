"""Use Soup's actual DPO wrapper; add evidence hooks, not a replacement trainer."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common import read_jsonl, sha256, utc_now, write_json, write_jsonl
from scripts.dpo_data import prepared_lengths
from scripts.precision import configure_deterministic_math


def main():
    import torch
    from soup_cli.config.loader import load_config
    from soup_cli.trainer.dpo import DPOTrainerWrapper
    from transformers import TrainerCallback

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/soup.yaml")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--control", action="store_true", help="Two real DPO optimizer steps with lr=0")
    args = parser.parse_args()
    precision = configure_deterministic_math()
    evidence = Path(args.evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; no CPU training can satisfy the T4 task")
    if "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"Expected T4, got {torch.cuda.get_device_name(0)}")
    cfg = load_config(args.config)
    if args.control:
        cfg.training.lr = 0.0
        cfg.training.warmup_ratio = 0.0
        cfg.output = "outputs/control_adapter"
    write_json(evidence / "effective_config.json", cfg.model_dump(mode="json"))
    write_json(evidence / "hardware.json", {
        "timestamp": utc_now(), "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "precision": precision,
        "config_sha256": sha256(args.config), "train_sha256": sha256("data/train.jsonl"),
    })
    torch.cuda.reset_peak_memory_stats()
    wrapper = DPOTrainerWrapper(cfg, device="cuda", report_to="none")
    training_rows = read_jsonl("data/train.jsonl")
    wrapper.setup({"train": training_rows, "val": []})
    model, trainer = wrapper.model, wrapper.trainer
    model.config.use_cache = False
    # Fix tiny negative control's duration without pretending it was the full run.
    if args.control:
        trainer.args.max_steps = 2
        trainer.args.save_steps = 2
    initial_dir = evidence / "initial_adapter"
    model.save_pretrained(str(initial_dir), safe_serialization=True, selected_adapters=["default"])
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Unexpected trainable parameters")
    if any(param.device.type != "cuda" for _, param in trainable):
        raise RuntimeError("LoRA parameter is not materialized on CUDA")
    if not trainer.args.fp16 or trainer.args.bf16:
        raise RuntimeError("T4 must use fp16 AMP and not bf16")
    if not cfg.training.stream_layers or wrapper._stream_runtime is None:
        raise RuntimeError("No active layer-streaming runtime")
    prepared = trainer.train_dataset
    for row in prepared:
        lengths = prepared_lengths(row)
        if lengths["prompt_tokens"] + max(lengths["chosen_tokens"], lengths["rejected_tokens"]) > cfg.data.max_length:
            raise RuntimeError("Prepared TRL dataset exceeds max_length")
    write_json(evidence / "pre_training.json", {
        "timestamp": utc_now(), "initial_adapter_sha256": sha256(initial_dir / "adapter_model.safetensors"),
        "trainable_count": sum(p.numel() for _, p in trainable),
        "trainable_dtypes": sorted({str(p.dtype) for _, p in trainable}),
        "stream_runtime": wrapper._stream_runtime.stats(),
        "fp16": trainer.args.fp16, "bf16": trainer.args.bf16,
        "prepared_rows": len(prepared), "reference_instance": trainer.ref_model is not None,
        "reference_adapter": getattr(trainer, "ref_adapter_name", None),
    })
    first_batch = trainer.data_collator([prepared[0]])
    first_batch = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in first_batch.items()}
    reference_before = [v.detach().cpu() for v in trainer.compute_ref_log_probs(first_batch)]
    gradient_events = []
    actual_optimizer_steps = []
    optimizer_hook = []

    class EvidenceCallback(TrainerCallback):
        def on_train_begin(self, callback_args, state, control, **kwargs):
            optimizer = kwargs["optimizer"]
            optimizer = getattr(optimizer, "optimizer", optimizer)
            if not hasattr(optimizer, "register_step_post_hook"):
                raise RuntimeError("Cannot record actual optimizer steps")

            def after_step(opt, step_args, step_kwargs):
                actual_optimizer_steps.append(utc_now())

            optimizer_hook.append(optimizer.register_step_post_hook(after_step))

        def on_pre_optimizer_step(self, callback_args, state, control, **kwargs):
            norms = {}
            for name, param in trainable:
                grad = param.grad
                if grad is None:
                    raise RuntimeError(f"Missing gradient: {name}")
                if not torch.isfinite(grad).all():
                    raise RuntimeError(f"Nonfinite gradient: {name}")
                norms[name] = grad.float().norm().item()
            if not any(norms.values()):
                raise RuntimeError("All LoRA gradients are zero")
            gradient_events.append({"timestamp": utc_now(), "step": state.global_step + 1,
                                    "finite": True, "norms": norms})
            write_jsonl(evidence / "gradient_events.jsonl", gradient_events)

        def on_log(self, callback_args, state, control, logs=None, **kwargs):
            print({"timestamp": utc_now(), "step": state.global_step, "metrics": logs}, flush=True)

    trainer.add_callback(EvidenceCallback())
    setup_peak = torch.cuda.max_memory_allocated()
    setup_reserved_peak = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    # Training context closes runtime only after our reference and snapshot checks.
    try:
        from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16

        align_trainable_dtype_for_fp16(model, fp16=trainer.args.fp16, bf16=trainer.args.bf16)
        metrics = trainer.train()
        torch.cuda.synchronize()
        train_peak, train_reserved_peak = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        reference_after = [v.detach().cpu() for v in trainer.compute_ref_log_probs(first_batch)]
        reference_deltas = [(a - b).abs().max().item() for a, b in zip(reference_before, reference_after, strict=True)]
        reference_stable = max(reference_deltas) <= 1e-4
        trainer.save_model(cfg.output)
        wrapper.tokenizer.save_pretrained(cfg.output)
        trainer.state.save_to_json(str(evidence / "trainer_state.json"))
        result = {
            "timestamp": utc_now(), "global_steps": trainer.state.global_step,
            "gradient_events": len(gradient_events), "reference_stable": reference_stable,
            "actual_optimizer_steps": len(actual_optimizer_steps),
            "reference_max_delta": max(reference_deltas),
            "setup_peak_allocated_bytes": setup_peak, "setup_peak_reserved_bytes": setup_reserved_peak,
            "training_peak_allocated_bytes": train_peak, "training_peak_reserved_bytes": train_reserved_peak,
            "metrics": metrics.metrics, "control_lr_zero": args.control,
            "initial_adapter_sha256": sha256(initial_dir / "adapter_model.safetensors"),
            "final_adapter_sha256": sha256(Path(cfg.output) / "adapter_model.safetensors"),
        }
        write_json(evidence / "training_result.json", result)
        print(result)
        if not reference_stable:
            raise RuntimeError("DPO reference changed during training")
        if len(gradient_events) != trainer.state.global_step:
            raise RuntimeError("Missing optimizer-step evidence")
        if len(actual_optimizer_steps) != trainer.state.global_step:
            raise RuntimeError("AMP skipped optimizer updates despite advancing trainer steps")
    finally:
        for hook in optimizer_hook:
            hook.remove()
        wrapper._close_stream_runtime()


if __name__ == "__main__":
    main()
