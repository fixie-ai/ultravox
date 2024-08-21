import torch


def default_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def default_dtype():
    # MPS got bfloat16 support in macOS Sonoma 14.
    return (
        torch.bfloat16
        if torch.cuda.is_available() or torch.backends.mps.is_available()
        else torch.float32
    )


def get_dtype(data_type: str):
    return (
        torch.bfloat16
        if data_type == "bfloat16"
        else torch.float16 if data_type == "float16" else torch.float32
    )
