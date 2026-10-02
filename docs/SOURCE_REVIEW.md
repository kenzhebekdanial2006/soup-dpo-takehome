# Source-grounded checks (reviewed 2026-10-02)

The reviewed release is `soup-cli==0.75.2`, upstream tag `v0.75.2`, commit
`3966ef95cead56f500a67b1f68cbc84d22b11cfb`. These are source observations;
they are **not** results of our T4 run.

| Surface | What it actually establishes | What it does not establish |
|---|---|---|
| `soup data doctor` | Chat template compatibility, EOS/loss-mask diagnostics, truncation heuristics | Rejects preference formats; chat projection is not DPO mask validation |
| `soup data lint` | Preference length bias, identical pairs, near duplicates, prompt leakage | Preference label correctness, all semantic duplicates, independent evaluation |
| `soup train --dry-run` | CLI config/data/environment and hardware-fit path | Does not execute a DPO backward or prove adapters update |
| Streaming setup panel | Architecture guards, layer/RAM pool sizing, estimated paired VRAM | A fit prediction is not measured peak or gradient correctness |
| `soup ship --evidence` | Strict task-score improvement plus supplied regression scores | Does not independently verify the scores, their metric, sampling uncertainty, or training correctness |

Implementation links pinned to the reviewed commit:

- [DPO wrapper, paired batch and reference handling](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/trainer/dpo.py)
- [Streaming setup and actual runtime budget](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/trainer/stream_setup.py)
- [Streaming state serialization and backward implementation](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/utils/layer_stream_runtime.py)
- [Doctor preference-format refusal](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/commands/data_doctor.py)
- [Preference lint engine](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/utils/data_lint.py)
- [Pure ship decision engine](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/utils/ship_verdict.py)
- [Config schema](https://github.com/MakazhanAlpamys/Soup/blob/3966ef95cead56f500a67b1f68cbc84d22b11cfb/src/soup_cli/config/schema.py)

The upstream [streaming documentation](https://trysoup.dev/docs/layer-streaming)
describes the historical `.inner.` serialization bug and NF4 buffer aliasing
gradient bug. Our unquantized 1.5B recipe does not exercise NF4; passing it says
nothing about NF4. The independent gradient comparison exists because loss
alone cannot validate backward.

[Qwen's model config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)
specifies 28 layers, hidden size 1536, intermediate size 8960, 12 attention heads,
2 KV heads, vocabulary 151936 and tied embeddings. The local downloaded config
and snapshot manifest are the run's authoritative evidence.
