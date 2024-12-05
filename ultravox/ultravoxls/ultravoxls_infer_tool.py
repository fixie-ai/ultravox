#!/usr/bin/env python
import dataclasses
import os
from typing import Optional

import simple_parsing
import torch
import torchaudio

from ultravox.data import data_sample
from ultravox.inference import base
from ultravox.ultravoxls import ultravoxls_infer


@dataclasses.dataclass
class InferArgs:
    # Path to the audio file
    audio_file: str = simple_parsing.field(alias="-f")
    # Model ID to use for the model
    model: str = simple_parsing.field(
        default="wandb://fixie/ultravox/model-lsm_multilingual_ls_8bs_lrdiv6:v5",
        alias="-m",
    )
    # Temperature for sampling
    temperature: float = simple_parsing.field(default=1.0, alias="-t")
    # How many tokens to generate
    max_tokens: int = simple_parsing.field(default=160, alias="-T")
    # Use the first N seconds of the audio file as input. The output will be the continuation of the input audio.
    crop_input_audio_secs: Optional[float] = simple_parsing.field(default=2, alias="-c")


def run_infer(
    inference: base.VoiceInference,
    sample: data_sample.VoiceSample,
    temperature: float = 1.0,
    max_tokens: int = 160,
):
    os.makedirs("./data", exist_ok=True)
    input_path = "./data/infer-in.wav"
    torchaudio.save(
        input_path, torch.Tensor(sample.audio).reshape(1, -1), sample.sample_rate
    )
    audio_out = inference.infer(
        sample, max_tokens=max_tokens, temperature=temperature
    ).text
    output_path = "./data/infer-out.wav"
    torchaudio.save(output_path, audio_out, 24000)
    print("Audio saved to", output_path)


def oneshot_infer(
    inference: base.VoiceInference,
    audio_file: str,
    crop_input_audio_secs: Optional[float] = None,
    **kwargs,
):
    with open(audio_file, "rb") as file:
        sample = data_sample.VoiceSample.from_prompt_and_buf("", file.read())

    if crop_input_audio_secs and sample.audio is not None:
        sample.audio = sample.audio[: int(crop_input_audio_secs * sample.sample_rate)]

    run_infer(inference, sample, **kwargs)


args = simple_parsing.parse(InferArgs)
inference = ultravoxls_infer.UltravoxLSInference(args.model, device="cpu")
oneshot_infer(
    inference,
    args.audio_file,
    crop_input_audio_secs=args.crop_input_audio_secs,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
)
