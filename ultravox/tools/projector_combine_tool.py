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

import huggingface_hub
import safetensors.torch
import simple_parsing
import transformers

from ultravox.model import ultravox_config
from ultravox.model import ultravox_model
from ultravox.model import wandb_utils


@dataclasses.dataclass
class CombineProjectionArgs:
    orig_model_id: str = simple_parsing.field(alias="-m")
    new_text_model_id: str = simple_parsing.field(alias="-t")
    new_model_id: str = simple_parsing.field(alias="-u")
    projector_repo: str = simple_parsing.field(alias="-p")
    projector_file: str = simple_parsing.field(
        alias="-f", default="projection_s2b.safetensors"
    )

    def __post_init__(self):
        if wandb_utils.is_wandb_url(self.orig_model_id):
            self.orig_model_id = wandb_utils.download_model_from_wandb(
                self.orig_model_id
            )


def load_hf_weights(repo: str, file_path: str):
    local_path = huggingface_hub.hf_hub_download(repo, file_path)
    weights = safetensors.torch.load_file(local_path)
    return weights


def main():
    weights = load_hf_weights(args.projector_repo, args.projector_file)
    assert "0.weight" in weights, "Expected projector to have '0.weight'"
    assert set(weights.keys()).issubset(
        {"0.weight", "0.bias"}
    ), "The projector should only have a single linear layer"

    uv: ultravox_model.UltravoxModel = transformers.AutoModel.from_pretrained(
        args.orig_model_id, trust_remote_code=True
    )

    assert (
        uv.config.text_model_id is not None
    ), "This process will likely not work if the base text model has been modified. Panicking!"
    assert (
        uv.config.audio_model_id is not None
    ), "Audio model has been modified ... this is not supported ... YET!"
    assert (
        uv.config.projector_ln_mid == True
    ), "In order to combine the projectors, the model must use layer norm after the first linear layer instead of the after the second linear layer (v0.4.1 and below)."

    # reduce memory usage by releasing unused parts of the model
    uv.language_model = None
    uv.audio_tower = None

    config_params = {**uv.config.to_dict(), "text_model_id": args.new_text_model_id}

    config = ultravox_config.UltravoxConfig(**config_params)

    new_model = ultravox_model.UltravoxModel(config=config)

    # reduce memory usage by releasing unused parts of the model. TODO: load in meta device instead.
    new_model.audio_tower = None
    new_model.language_model = None

    # compute the merged projector weights
    combined_weights = uv.multi_modal_projector.state_dict()
    combined_weights["linear_2.weight"] = (
        weights["0.weight"].float() @ uv.multi_modal_projector.linear_2.weight.float()
    )

    new_model.multi_modal_projector.load_state_dict(combined_weights)

    new_model.push_to_hub(args.new_model_id)


if __name__ == "__main__":
    args = simple_parsing.parse(CombineProjectionArgs)
    main()
