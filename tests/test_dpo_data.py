import pytest

from scripts.dpo_data import prepared_lengths


def test_empty_completion_is_rejected():
    with pytest.raises(ValueError, match="completion is empty"):
        prepared_lengths({"prompt_ids": [1], "chosen_ids": [], "rejected_ids": [2]})


def test_old_schema_reports_missing_fields():
    with pytest.raises(ValueError, match="TRL 0.29 prepared fields"):
        prepared_lengths({"prompt_input_ids": [1], "chosen_input_ids": [2], "rejected_input_ids": [3]})


def test_recorded_lengths_match_actual_trl_completion_masks():
    pytest.importorskip("torch")
    pytest.importorskip("trl")
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    row = {"prompt_ids": [1, 2], "chosen_ids": [3, 4, 5], "rejected_ids": [6]}
    batch = DataCollatorForPreference(pad_token_id=0)([row])
    lengths = prepared_lengths(row)
    assert lengths["chosen_tokens"] == batch["completion_mask"][0].sum().item()
    assert lengths["rejected_tokens"] == batch["completion_mask"][1].sum().item()
    assert lengths["prompt_tokens"] + lengths["chosen_tokens"] == batch["attention_mask"][0].sum().item()
    assert lengths["prompt_tokens"] + lengths["rejected_tokens"] == batch["attention_mask"][1].sum().item()
