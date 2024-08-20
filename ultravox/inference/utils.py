import torch


def default_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def default_dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def get_dtype(data_type: str):
    return (
        torch.bfloat16
        if data_type == "bfloat16"
        else torch.float16 if data_type == "float16" else torch.float32
    )
