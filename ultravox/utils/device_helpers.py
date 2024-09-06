import os
from typing import Optional

import torch

def default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def default_dtype():
    # macOS Sonoma 14 enabled bfloat16 on MPS.
    return (
        torch.bfloat16
        if torch.cuda.is_available() or torch.backends.mps.is_available()
        else torch.float16
    )

def get_dtype(data_type: Optional[str] = None):
    if data_type is None:
        return default_dtype()
    else:
        return torch.bfloat16 if data_type == "bfloat16" else torch.float16 if data_type == "float16" else torch.float32

def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))
    
def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0)) 