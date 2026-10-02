"""Inspect the prepared text preference schema of the pinned TRL 0.29 stack."""


def prepared_lengths(row):
    fields = {"prompt_tokens": "prompt_ids", "chosen_tokens": "chosen_ids", "rejected_tokens": "rejected_ids"}
    missing = set(fields.values()) - row.keys()
    if missing:
        raise ValueError(f"Missing TRL 0.29 prepared fields: {sorted(missing)}; available: {sorted(row)}")
    lengths = {name: len(row[field]) for name, field in fields.items()}
    if not lengths["chosen_tokens"] or not lengths["rejected_tokens"]:
        raise ValueError("Prepared TRL completion is empty")
    return lengths
