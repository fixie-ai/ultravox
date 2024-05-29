import os
import typing as t

import peft
import torch
import transformers


def get_llm(model_name: str, cpu=False, dtype=torch.float32):
    pipe = transformers.pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=dtype,
        device_map="cpu" if cpu else "cuda",
        token=os.environ.get("HF_ACCESS_TOKEN", None),
    )

    if "pad_token" not in pipe.tokenizer.special_tokens_map:
        pipe.tokenizer.add_special_tokens({"pad_token": "</s>"})

    return pipe.model, pipe.tokenizer


def apply_lora(
    model: torch.nn.Module,
    lora_config: t.Optional[peft.LoraConfig] = None,
):
    if lora_config.r == 0:
        freeze_parameters(model)
    else:
        model = peft.get_peft_model(model, lora_config)

    return model


def freeze_parameters(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False
