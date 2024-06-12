#!/usr/bin/env python

import dataclasses
from typing import Optional

import simple_parsing

from ultravox.inference import ultravox_infer


# This script is used to upload a model to the HuggingFace Hub, for either internal or external consumption.
# Ex: python -m ultravox.tools.push_to_hub -m wandb://fixie/ultravox/<model_path> -u fixie-ai/ultravox-vXYZ
@dataclasses.dataclass
class UploadToHubArgs:
    # Model ID to use for the model
    model: str = simple_parsing.field(alias="-m")
    # HuggingFace Hub model_id to push to
    hf_upload_model: str = simple_parsing.field(alias="-u")
    # Tokenizer ID to use: usually the same as the model ID
    tokenizer: Optional[str] = None
    # Device to use for the model
    device: Optional[str] = simple_parsing.field(default=None, alias="-D")
    # Data type to use for the model
    data_type: Optional[str] = None
    # Public or private (default)
    private: bool = True


def main(args: UploadToHubArgs):
    # Load the model and tokenizer, then merge LoRA weights if they exist
    inference = ultravox_infer.UltravoxInference(
        args.model,
        tokenizer_id=args.tokenizer,
        device=args.device,
        data_type=args.data_type,
    )
    print("Uploading model to HuggingFace Hub...")
    inference.model.push_to_hub(args.hf_upload_model, private=args.private)
    # It's not necessary to upload the tokenizer, but it can be useful for consistency
    print("Uploading tokenizer to HuggingFace Hub...")
    inference.tokenizer.push_to_hub(args.hf_upload_model, private=args.private)


if __name__ == "__main__":
    main(simple_parsing.parse(UploadToHubArgs))
