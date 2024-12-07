from dataclasses import dataclass
from typing import Optional

import gradio as gr
import simple_parsing

from ultravox import data as datasets
from ultravox.inference import base as infer_base
from ultravox.tools import gradio_helper
from ultravox.tools.gradio_voice import make_demo

DEMO_INSTRUCTION: str = """Enter your prompt here (audio will be inserted at the end or at <|audio|>).

Text mode: Shift+Enter to submit.
Voice mode: Click the recording button to start, then click again to stop and submit.
"""


@dataclass
class DemoConfig:
    # model_path can refer to a HF hub model_id, a local path, or a Weights & Biases artifact
    #    fixie-ai/ultravox
    #    runs/llama2_asr_gigaspeech/checkpoint-1000/
    #    wandb://fixie/ultravox/model-llama2_asr_gigaspeech:v0
    model_path: str = "fixie-ai/ultravox-v0_3"
    device: Optional[str] = None
    data_type: Optional[str] = None
    default_prompt: str = ""
    max_new_tokens: int = 200
    temperature: float = 0
    voice_mode: bool = False


args = simple_parsing.parse(config_class=DemoConfig)
# This script will get loaded from both the python runtime as well as gradio's reloader,
# even when gr.NO_RELOAD is used. To avoid model re-init, we use a helper singleton
# to manage the inference object.
inference = gradio_helper.get_inference(args)


def add_text(history, text: str):
    # We set the prompt to "" in anticipation of the next prompt in text mode.
    return history + [[text, None]], ""


def add_audio(history, audio: str, text: str):
    # We want to keep the prompt (mixed audio/text instruction) as is in voice mode.
    return history + ([[text, None]] if text else []) + [[(audio,), None]]


def process_turn(
    history,
    prompt: str,
    audio: Optional[str] = None,
    max_new_tokens: int = 200,
    temperature: float = 0,
):
    if audio:
        if "<|audio|>" not in prompt:
            prompt += "<|audio|>"
        sample = datasets.VoiceSample.from_prompt_and_file(prompt, audio)
    else:
        # Note that prompt will be "" here, since we cleared it in add_text.
        # Instead, we can just get it from the chat history.
        sample = datasets.VoiceSample.from_prompt(history[-1][0])

    if len(sample.messages) != 1:
        raise ValueError(
            f"Expected exactly 1 message in sample but got {len(sample.messages)}"
        )

    output = inference.infer_stream(
        sample,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    history[-1][1] = ""
    for chunk in output:
        if isinstance(chunk, infer_base.InferenceChunk):
            history[-1][1] += chunk.text
            yield history


def process_text(history, prompt, max_new_tokens, temperature):
    yield from process_turn(
        history, prompt, max_new_tokens=max_new_tokens, temperature=temperature
    )


def process_audio(history, prompt, audio, max_new_tokens, temperature):
    yield from process_turn(
        history,
        prompt,
        audio=audio,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def gradio_reset():
    inference.update_conversation()
    return [], "", None


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(scale=10, height=1000)

    with gr.Row():
        with gr.Column(scale=1):
            reset = gr.Button("Reset")
            audio = gr.Audio(
                label="🎤",
                sources=["microphone"],
                type="filepath",
                visible=True,
            )
        with gr.Column(scale=8):
            prompt = gr.Textbox(
                show_label=False,
                lines=5,
                placeholder=DEMO_INSTRUCTION,
                value=args.default_prompt,
                container=True,
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
    prompt.submit(add_text, [chatbot, prompt], [chatbot, prompt], queue=False).then(
        process_text,
        [chatbot, prompt, max_new_tokens, temperature],
        [chatbot],
    )
    audio.stop_recording(
        add_audio, [chatbot, audio, prompt], [chatbot], queue=False
    ).then(
        process_audio,
        [chatbot, prompt, audio, max_new_tokens, temperature],
        [chatbot],
    )
    reset.click(gradio_reset, [], [chatbot, prompt, audio], queue=False)
    demo.load(gradio_reset, [], [chatbot, prompt, audio], queue=False)


if args.voice_mode:
    demo = make_demo(args, inference)

if __name__ == "__main__":
    demo.launch(share=True)
