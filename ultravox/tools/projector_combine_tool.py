"""
This script combines two projectors into a single projector.

The projectors are combined by multiplying the weights of the two projectors and then normalizing them.

PTAL at projector_regression_tool.py for more details on how the projectors are created.

Example usage:

    just python -m ultravox.tools.projector_combine_tool -p fixie-ai/llama-3.2-1b-8b-projection \
        -m wandb://fixie/ultravox/model-llama-3.2-1b-whisper-small:v1 \
        -t meta-llama/Llama-3.1-8B-Instruct -u fixie-ai/test-projected-nolayernorm
"""

import dataclasses
import logging
import os

import safetensors.torch
import simple_parsing
import torch
import torch.nn as nn

from ultravox.model import file_utils
from ultravox.model import hf_hub_utils
from ultravox.model import ultravox_config
from ultravox.model import ultravox_model

PREFIXES_TO_REMOVE = ["projector.", "0."]  # NOTE: the order matters


@dataclasses.dataclass
class CombineProjectionArgs:
    # Path to the source model. It can be a local path or a HF Hub model.
    source_model_path: str = simple_parsing.field(alias="-s")
    # Where to save the new model. If it starts with hf://, it will be pushed to the hub, otherwise it will be saved locally.
    target_model_path: str = simple_parsing.field(alias="-t")
    # The text model to use for the target model. Note that this cannot be a local path.
    target_text_model_id: str = simple_parsing.field(alias="-lm")
    # The projector to use for the target model. It can be local, a HF Hub model, or a W&B model.
    projector_path: str = simple_parsing.field(alias="-p")
    # Whether to merge the LoRA layers (for audio encoder) into the target model.
    merge_lora: bool = simple_parsing.field(alias="-merge", default=False)
    # Whether to print verbose output.
    verbose: bool = simple_parsing.field(alias="-v", default=True)

    def __post_init__(self):
        self.source_model_path = file_utils.download_dir_if_needed(
            self.source_model_path
        )


def main(args: CombineProjectionArgs):
    projector = load_projector(args.projector_path)

    source_model = ultravox_model.UltravoxModel.from_pretrained(
        args.source_model_path, trust_remote_code=True
    )

    assert source_model.config.text_model_id is not None and not any(
        k.startswith("language_model.") for k in source_model.keep_params
    ), "This process will likely not work if the base text model has been modified. Panicking!"
    assert (
        source_model.config.projector_ln_mid == True
    ), "In order to combine the projectors, the model must use layer norm after the first linear layer instead of the after the second linear layer (v0.4.1 and below)."
    assert (
        source_model.multi_modal_projector.linear_2.bias is None
    ), "Expected bias to be None."

    # reduce memory usage by releasing unused parts of the model
    source_model.language_model = None

    config_params = {
        **source_model.config.to_dict(),
        "text_model_id": args.target_text_model_id,
    }

    config = ultravox_config.UltravoxConfig(**config_params)

    target_model = ultravox_model.UltravoxModel(config=config)

    # reduce memory usage by releasing unused parts of the model. TODO: load in meta device instead.
    target_model.language_model = None

    # compute the merged projector weights
    combined_weights = source_model.diff_state_dict()
    combined_weights["multi_modal_projector.linear_2.weight"] = combine_linear_layers(
        source_model.multi_modal_projector.linear_2, projector
    ).weight

    mismatch = target_model.load_state_dict(combined_weights, strict=False)
    missing_trainable_keys = [
        k for k in mismatch.missing_keys if k in target_model.keep_params
    ]
    assert (
        not mismatch.unexpected_keys
    ), f"Found unexpected keys when loading weights: {mismatch.unexpected_keys}"
    assert (
        not missing_trainable_keys
    ), f"Found keys not initialized from model: {missing_trainable_keys}"

    if args.merge_lora:
        target_model.merge_and_unload()

    if hf_hub_utils.is_hf_model(args.target_model_path):
        target_model.push_to_hub(hf_hub_utils.get_hf_model_id(args.target_model_path))
    else:
        target_model.save_pretrained(args.target_model_path)

    if args.verbose:
        print("-------------")
        print(f"Model with merged projector saved to {args.target_model_path}")
        print(target_model.config)
        print(target_model)
        model_path = os.path.join(args.target_model_path, "model.safetensors")
        if os.path.exists(model_path):
            data = safetensors.torch.load_file(model_path)
            print("saved keys", data.keys())
        print("-------------")


def load_weights(path: str, file_name: str):
    local_path = file_utils.download_file_if_needed(path, file_name)
    if os.path.isdir(local_path):
        local_path = path
    else:
        raise ValueError(f"Invalid path: {path}")

    print(f"Loading weights from {local_path}")
    weights = safetensors.torch.load_file(local_path)
    return weights


def load_projector(path: str, file_name: str = "model.safetensors"):
    weights = load_weights(path, file_name)

    assert (
        len(weights) <= 2
    ), "Expected projector to have a single linear layer and no bias"

    for prefix in PREFIXES_TO_REMOVE:
        weights = {k.replace(prefix, ""): v for k, v in weights.items()}

    if "bias" in weights:
        logging.warning(
            "Found bias in projector weights. This is not expected. Ignoring bias."
        )

    return linear_from_weights(weights["weight"])


def linear_from_weights(weights: torch.Tensor):
    layer = nn.Linear(weights.shape[1], weights.shape[0], bias=False)
    layer.load_state_dict({"weight": weights.float()})
    return layer


def combine_linear_layers(linear_1: nn.Linear, linear_2: nn.Linear):
    # convert weights to full precision to avoid numerical instability
    # NOTE: this changes the underlying layers, but in this case it's fine
    linear_1.float()
    linear_2.float()
    combined = linear_from_weights(linear_2.weight @ linear_1.weight)
    test_inp = torch.randn(1, combined.in_features)
    assert torch.allclose(linear_2(linear_1(test_inp)), combined(test_inp), atol=1e-4)
    return combined


if __name__ == "__main__":
    main(simple_parsing.parse(CombineProjectionArgs))
