from __future__ import annotations

import argparse
import platform
import sys

from scripts.common import utc_now, write_json


def main():
    import importlib.metadata as metadata

    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    versions = {p: metadata.version(p) for p in ["soup-cli", "torch", "transformers", "peft", "trl", "datasets", "accelerate"]}
    result = {"timestamp": utc_now(), "python": sys.version, "platform": platform.platform(),
              "versions": versions, "cuda_available": torch.cuda.is_available(),
              "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        result.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                      bf16_supported=torch.cuda.is_bf16_supported())
    write_json(args.output, result)
    print(result)
    if not result["cuda_available"] or "T4" not in result.get("gpu", ""):
        raise RuntimeError("Choose a Colab NVIDIA T4 GPU runtime, then restart")


if __name__ == "__main__":
    main()
