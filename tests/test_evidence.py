import json

import pytest

from scripts.build_data import build
from scripts.common import canonical_prompt, read_jsonl
from scripts.make_report import create_report
from scripts.memory_budget import estimate
from scripts.verify_training import compare_states, functional_change


def state(value):
    torch = pytest.importorskip("torch", reason="ML dependency checks run in Colab")
    return {"module.lora_A.weight": torch.ones(2, 3),
            "module.lora_B.weight": torch.full((3, 2), float(value))}


def scores(value):
    return [{"id": str(i), "chosen": {"token_logps": [-float(value), -1.]},
             "rejected": {"token_logps": [-3., -2.]}} for i in range(4)]


def test_no_op_is_rejected_even_with_nonzero_A():
    result = compare_states(state(0), state(0))
    assert not result["passed"]
    assert result["changed_tensors"] == 0
    assert not functional_change(scores(1), scores(1), scores(1))["passed"]


def test_change_must_exceed_measured_noise():
    assert compare_states(state(0), state(0.01))["passed"]
    assert functional_change(scores(1), scores(1), scores(1.1))["passed"]
    assert not functional_change(scores(1), scores(1.001), scores(1.01))["passed"]


def test_missing_or_nonfinite_tensor_fails():
    assert not compare_states(state(0), {})["passed"]
    assert not compare_states(state(0), state(float("nan")))["passed"]


def test_weight_change_does_not_replace_functional_proof():
    # Saved weights change, but a loader that ignored them still serves identical outputs.
    assert compare_states(state(0), state(0.1))["passed"]
    assert not functional_change(scores(1), scores(1), scores(1))["passed"]


def test_probe_identity_is_not_silently_ignored():
    other = scores(1.1)
    other[0]["id"] = "different"
    with pytest.raises(ValueError):
        functional_change(scores(1), scores(1), other)


def test_group_split_and_reproducible_data(tmp_path):
    build(tmp_path / "first")
    build(tmp_path / "second")
    assert (tmp_path / "first/train.jsonl").read_bytes() == (tmp_path / "second/train.jsonl").read_bytes()
    train = read_jsonl(tmp_path / "first/train.jsonl")
    heldout = read_jsonl(tmp_path / "first/heldout.jsonl")
    assert len(train) == 400 and len(heldout) == 100
    assert len({r["group"] for r in heldout}) == 10
    assert not {r["group"] for r in train} & {r["group"] for r in heldout}
    assert not {canonical_prompt(r) for r in train} & {canonical_prompt(r) for r in heldout}


def test_budget_counts_paired_rows_tied_head_and_no_full_reference():
    cfg = {"hidden_size": 1536, "intermediate_size": 8960, "num_hidden_layers": 28,
           "vocab_size": 151936, "num_attention_heads": 12, "num_key_value_heads": 2,
           "tie_word_embeddings": True}
    result = estimate(cfg)
    assert result["lora_parameters"] == 1089536
    assert result["base_parameters"] == 1543714304
    assert result["shape"]["forward_rows"] == 2
    components = result["components"]
    assert components["embedding_or_large_vocab_slot"] == 1536 * 151936 * 2
    assert result["planned_peak_upper_bytes"] == sum(components.values()) - components["reference_logit_peak_allowance"]
    # Pair batch increases activation/logit memory but does not multiply weights.
    larger = estimate(cfg, batch=2)
    assert larger["components"]["policy_logits_fp16"] == 2 * components["policy_logits_fp16"]
    assert larger["components"]["decoder_pool"] == components["decoder_pool"]


def test_missing_evidence_is_unrun_and_dont_ship(tmp_path):
    report = create_report(tmp_path)
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["verdict"] == "DON'T SHIP"
    assert not any(verdict["gates"].values())
    assert "UNRUN" in " ".join(body for _, body in report)


def test_actual_tiny_peft_update_and_disabled_model_control():
    # Genuine CPU tensor computation; no downloaded checkpoint and no mocked PEFT loader.
    torch = pytest.importorskip("torch", reason="ML dependency checks run in Colab")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(17)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=True, attention_dropout=0.0)
    model = get_peft_model(Qwen2ForCausalLM(config), LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"))
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    initial = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    with torch.no_grad(), model.disable_adapter():
        base_logits = model(ids).logits.clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(4):
        optimizer.zero_grad()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
    trained = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    assert compare_states(initial, trained)["passed"]
    with torch.no_grad():
        active = model(ids).logits
        with model.disable_adapter():
            disabled = model(ids).logits
    assert (active - base_logits).abs().max().item() > 1e-4
    assert torch.equal(disabled, base_logits)
