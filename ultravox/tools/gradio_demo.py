from dataclasses import dataclass
from typing import Tuple

import gradio as gr
import numpy as np
import simple_parsing

from ultravox.data import datasets
from ultravox.inference import ultravox_infer


@dataclass
class DemoConfig:
    # model_path can refer to a HF hub model_id, a local path, or a Weights & Biases artifact
    #    fixie-ai/ultravox
    #    runs/llama2_asr_gigaspeech/checkpoint-1000/
    #    wandb://fixie/ultravox/model-llama2_asr_gigaspeech:v0
    model_path: str = "fixie-ai/ultravox"
    default_prompt: str = "Transcribe <|audio|>"


def main():
    args = simple_parsing.parse(config_class=DemoConfig)
    inference = ultravox_infer.UltravoxInference(args.model_path)

    def wrapper(text: str, audio: Tuple[int, np.ndarray]) -> str:
        sample = datasets.VoiceSample.from_prompt_and_raw(text, audio[1], audio[0])
        return inference.infer(sample, max_tokens=64).text

    inputs = [
        gr.Textbox(label="Prompt", value=args.default_prompt),
        gr.Audio(label="Audio", show_download_button=True),
    ]
    outputs = [gr.Textbox(label="Output")]
    examples = [["Transcribe <|audio|>", "examples/test16.wav"]]

    gr.Interface(fn=wrapper, inputs=inputs, outputs=outputs, examples=examples).launch(
        share=True
    )


if __name__ == "__main__":
    main()
