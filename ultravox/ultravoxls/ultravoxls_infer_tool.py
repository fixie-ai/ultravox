#!/usr/bin/env python
import dataclasses
import os
from typing import Optional

import simple_parsing
import torch
import torchaudio

from ultravox import data as datasets
from ultravox.inference import base
from ultravox.ultravoxls import ultravoxls_infer


@dataclasses.dataclass
class InferArgs:
    # Model ID to use for the model
    model: str = simple_parsing.field(
        default="wandb://fixie/ultravox/model-lsm_multilingual_ls_8bs_lrdiv6:v5",
        alias="-m",
    )
    # Path to the audio file
    audio_file: str = simple_parsing.field(alias="-f", default="")
    # dataset name. Either audio_file or ds_name must be provided.
    ds_name: str = simple_parsing.field(default="", alias="-d")
    # Number of samples to generate when using a dataset
    num_samples: int = simple_parsing.field(default=1, alias="-n")
    # Temperature for sampling
    temperature: float = simple_parsing.field(default=1.0, alias="-t")
    # How many tokens to generate
    max_tokens: int = simple_parsing.field(default=160, alias="-T")
    # Use the first N seconds of the audio file as input. The output will be the continuation of the input audio.
    crop_input_audio_secs: Optional[float] = simple_parsing.field(default=2, alias="-c")
    # Output path for the generated audio
    output_path: str = simple_parsing.field(default="./data/infer", alias="-o")

    def __post_init__(self):
        if not self.audio_file and not self.ds_name:
            raise ValueError("Either audio_file or ds_name must be provided")
        if self.audio_file and self.ds_name:
            raise ValueError("Only one of audio_file or ds_name must be provided")


def run_infer(
    inference: base.VoiceInference,
    sample: datasets.VoiceSample,
    temperature: float = 1.0,
    max_tokens: int = 160,
    crop_input_audio_secs: Optional[float] = None,
    partial_path: str = "./data/infer",
):
    os.makedirs(os.path.dirname(partial_path), exist_ok=True)
    input_path = partial_path + "-in.wav"
    output_path = partial_path + "-out.wav"

    if crop_input_audio_secs and sample.audio is not None:
        sample.audio = sample.audio[: int(crop_input_audio_secs * sample.sample_rate)]

    torchaudio.save(
        input_path, torch.Tensor(sample.audio).reshape(1, -1), sample.sample_rate
    )
    audio_out = inference.infer(
        sample, max_tokens=max_tokens, temperature=temperature
    ).text
    torchaudio.save(output_path, audio_out, 24000)
    print("Audio saved to", output_path)


def oneshot_infer(
    inference: base.VoiceInference,
    audio_file: str,
    **kwargs,
):
    with open(audio_file, "rb") as file:
        sample = datasets.VoiceSample.from_prompt_and_buf("", file.read())

    run_infer(inference, sample, **kwargs)


def run_inference(args: InferArgs):
    inference = ultravoxls_infer.UltravoxLSInference(args.model, device="cpu")

    if args.ds_name:
        voice_data_args = datasets.VoiceDatasetArgs(
            split=datasets.DatasetSplit.VALIDATION, shuffle=True
        )
        ds: datasets.GenericDataset | datasets.Range = datasets.create_dataset(
            args.ds_name, voice_data_args
        )
        ds = datasets.Range(ds, num_samples=args.num_samples)
        for i, sample in enumerate(ds):
            run_infer(
                inference,
                sample,
                crop_input_audio_secs=args.crop_input_audio_secs,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                partial_path=f"{args.output_path}/infer-{args.ds_name}-{i}",
            )
    else:
        oneshot_infer(
            inference,
            args.audio_file,
            crop_input_audio_secs=args.crop_input_audio_secs,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            partial_path=f"{args.output_path}/infer",
        )


if __name__ == "__main__":
    run_inference(simple_parsing.parse(InferArgs))
