#!/usr/bin/env python

import dataclasses
from typing import Optional

import simple_parsing

from ultravox.inference import ultravox_infer


@dataclasses.dataclass
class UploadToHubArgs:
    # Model ID to use for the model
    model: str = simple_parsing.field(alias="-m")
    # Audio processor ID to use
    audio_processor: Optional[str] = None
    # Tokenizer ID to use
    tokenizer: Optional[str] = None
    # Data sets to use for inference
    device: Optional[str] = simple_parsing.field(default=None, alias="-D")
    # Data type to use for the model
    data_type: Optional[str] = None
    # HuggingFace Hub URL to push to
    hf_upload_url: Optional[str] = None


def main(args: UploadToHubArgs):
    inference = ultravox_infer.UltravoxInference(
        args.model,
        tokenizer_id=args.tokenizer,
        audio_processor_id=args.audio_processor,
        device=args.device,
        data_type=args.data_type,
    )
    print("Uploading model to HuggingFace Hub...")
    inference.model.push_to_hub(args.hf_upload_url)
    print("Uploading tokenizer to HuggingFace Hub...")
    inference.tokenizer.push_to_hub(args.hf_upload_url)


if __name__ == "__main__":
    main(simple_parsing.parse(UploadToHubArgs))
