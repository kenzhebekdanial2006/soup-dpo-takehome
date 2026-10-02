"""A reproducible T4 attention path for meaningful backward comparisons."""


def configure_deterministic_math():
    import os

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True)
    return {"attention": "SDPA math only", "deterministic_algorithms": True,
            "fp16_reduced_precision_reduction": False, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
