#!/usr/bin/env python

import dataclasses
from typing import Optional

import simple_parsing
import transformers

from ultravox import data as datasets
from ultravox.inference import ultravox_infer
from ultravox.model import ultravox_pipeline


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
    # Verify the model after uploading
    verify: bool = True


def main(args: UploadToHubArgs):
    # Load the model and tokenizer, then merge LoRA weights if they exist
    inference = ultravox_infer.UltravoxInference(
        args.model,
        tokenizer_id=args.tokenizer,
        device=args.device,
        data_type=args.data_type,
    )
    pipe = ultravox_pipeline.UltravoxPipeline(
        model=inference.model,
        tokenizer=inference.tokenizer,
        audio_processor=inference.processor.audio_processor,
        device=args.device,
    )
    print("Uploading model to HuggingFace Hub...")
    pipe.push_to_hub(args.hf_upload_model, private=args.private)

    if args.verify:
        print("Model uploaded. Testing model...")
        loaded_pipe = transformers.pipeline(
            model=args.hf_upload_model, trust_remote_code=True
        )
        ds = datasets.create_dataset("boolq", datasets.VoiceDatasetArgs())
        sample = next(iter(ds))
        generated = loaded_pipe(
            {"audio": sample.audio, "turns": sample.messages[:-1]}, max_new_tokens=10
        )
        print(f"Generated (max 10 tokens): {generated}")


if __name__ == "__main__":
    main(simple_parsing.parse(UploadToHubArgs))
