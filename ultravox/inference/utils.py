import torch


def default_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        # until https://github.com/pytorch/pytorch/issues/77764 is resolved
        # else "mps" if torch.backends.mps.is_available() else "cpu"
        else "cpu"
    )


def default_dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def get_dtype(data_type: str):
    return (
        torch.bfloat16
        if data_type == "bfloat16"
        else torch.float16 if data_type == "float16" else torch.float32
    )
