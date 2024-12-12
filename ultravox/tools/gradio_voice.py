from typing import cast

import gradio as gr
import numpy as np
from gradio.utils import get_space
from gradio_webrtc import AdditionalOutputs
from gradio_webrtc import ReplyOnPause
from gradio_webrtc import WebRTC
from gradio_webrtc import get_turn_credentials

from ultravox import data as datasets
from ultravox.inference import base as infer_base
from ultravox.inference import ultravox_infer

rtc_configuration = None
if get_space():
    rtc_configuration = get_turn_credentials(method="twilio")


def make_demo(args, inference):

    def transcribe(
        audio: tuple[int, np.ndarray],
        conversation: list[dict],
        max_new_tokens: int = 200,
        temperature: float = 0,
    ):

        sampling_rate, audio_array = audio

        conversation.append(
            {
                "role": "user",
                "content": gr.Audio(value=(sampling_rate, audio_array.squeeze())),
            }
        )
        yield AdditionalOutputs(conversation)

        sample = datasets.VoiceSample.from_prompt_and_raw(
            "<|audio|>", audio_array.squeeze(), sampling_rate
        )

        output = cast(ultravox_infer.UltravoxInference, inference).infer_stream(
            sample,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        conversation.append({"role": "assistant", "content": ""})
        for chunk in output:
            if isinstance(chunk, infer_base.InferenceChunk):
                conversation[-1]["content"] += chunk.text
                yield AdditionalOutputs(conversation)

    with gr.Blocks() as voice_demo:

        placeholder = """
<h1 style='text-align: center'>
    Talk to Ultravox Llama 3.1 8b (Powered by WebRTC ⚡️)
</h1>
<p style='text-align: center'>
    Once you grant access to your microphone, you can talk naturally to Ultravox.
    When you stop talking, the audio will be sent for processing.
</p>
<p style='text-align: center'>
    Each conversation is limited to 90 seconds. Once the time limit is up you can rejoin the conversation.
</p>
"""
        with gr.Row():
            conversation = gr.Chatbot(
                label="transcript", placeholder=placeholder, type="messages"
            )
        with gr.Row():
            with gr.Column(scale=4):
                audio = WebRTC(
                    rtc_configuration=rtc_configuration,
                    label="Stream",
                    mode="send",
                    modality="audio",
                )
            with gr.Column(scale=1):
                max_new_tokens = gr.Slider(
                    minimum=50,
                    maximum=2000,
                    value=args.max_new_tokens,
                    step=10,
                    interactive=True,
                    label="max_new_tokens",
                )
                temperature = gr.Slider(
                    minimum=0,
                    maximum=5.0,
                    value=args.temperature,
                    step=0.1,
                    interactive=True,
                    label="temperature",
                )

        audio.stream(
            ReplyOnPause(transcribe, input_sample_rate=16000),
            inputs=[audio, conversation, max_new_tokens, temperature],
            outputs=[audio],
            time_limit=90,
        )
        audio.on_additional_outputs(
            lambda g: g,
            outputs=[conversation],
            queue=False,
            show_progress="hidden",
        )

    return voice_demo
