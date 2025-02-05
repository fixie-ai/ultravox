"""
This script is used to average multiple models together (aka model soup).

Paper: https://arxiv.org/abs/2203.05482

Rough idea and why this can work:
Fine-tuned models optimized independently from the same pre-trained initialization lie in
the same basin of the error landscape, but none of them are usually optimal. Using averaging
usually leads to a better solution due to the basin-shape of the error landscape.

Usage:
    python -m ultravox.tools.model_averaging --model_paths <model_1> <model_2> <model_3> ... --output_path updated_model
"""

import dataclasses
import os
import shutil

import safetensors.torch
import simple_parsing
import torch

from ultravox.model import file_utils


@dataclasses.dataclass
class Config:
    model_paths: list[str]
    output_path: str = "updated_model"
    model_file_name: str = "model.safetensors"

    def __post_init__(self):
        self.model_paths = [
            file_utils.download_dir_if_needed(x) for x in self.model_paths
        ]


def main():
    args = simple_parsing.parse(Config)

    combined_model: dict[str, torch.Tensor] = {}
    num_models = len(args.model_paths)
    dtype = torch.bfloat16

    for i, model_path in enumerate(args.model_paths):
        weights_path = os.path.join(model_path, args.model_file_name)
        model = safetensors.torch.load_file(weights_path)
        if i == 0:
            # We assume that all models have the same structure
            # TODO: assert that the structure is the same
            shutil.copytree(model_path, args.output_path)
            combined_model = {k: v.float() / num_models for k, v in model.items()}
            dtype = list(model.values())[0].dtype
        else:
            assert combined_model.keys() == model.keys()
            for key in combined_model.keys():
                combined_model[key] += model[key].float() / num_models

    combined_model = {k: v.to(dtype=dtype) for k, v in combined_model.items()}

    weights_path = os.path.join(args.output_path, args.model_file_name)
    safetensors.torch.save_file(combined_model, weights_path)
    print(f"Saved averaged model to {weights_path}")


if __name__ == "__main__":
    main()
