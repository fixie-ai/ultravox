from dataclasses import dataclass
from typing import Optional

import gradio as gr
import simple_parsing

from ultravox.data import datasets
from ultravox.inference import ultravox_infer

demo_instruction: str = """Enter your prompt here (audio will be inserted at the end or at <|audio|>).

Text mode: Shift+Enter to submit.
Voice mode: Click the recording button to start, then click again to stop and submit.
"""


@dataclass
class DemoConfig:
    # model_path can refer to a HF hub model_id, a local path, or a Weights & Biases artifact
    #    fixie-ai/ultravox
    #    runs/llama2_asr_gigaspeech/checkpoint-1000/
    #    wandb://fixie/ultravox/model-llama2_asr_gigaspeech:v0
    model_path: str = "fixie-ai/ultravox"
    # Use <|audio|> to specify where to insert audio, otherwise, audio is inserted at the end in voice mode.
    default_prompt: str = ""
    max_new_tokens: int = 256
    device: str = "mps"
    data_type: str = "float16"


def main():
    args = simple_parsing.parse(config_class=DemoConfig)
    inference = ultravox_infer.UltravoxInference(
        args.model_path, device=args.device, data_type=args.data_type
    )

    def add_text(chatbot: gr.Chatbot, text: str) -> gr.Chatbot:
        return chatbot + [(text, None)]

    def add_audio(chatbot: gr.Chatbot, audio: str) -> gr.Chatbot:
        return chatbot + [((audio,), None)]

    def process_turn(
        chatbot: gr.Chatbot,
        prompt: str,
        audio: Optional[str] = None,
        temperature: float = 0,
    ):
        # We want to keep the prompt (mixed audio/text instruction) as is in voice mode, but set it to "" in anticipation of new prompt in text mode.
        prompt_to_return = prompt
        if audio:
            if "<|audio|>" not in prompt:
                prompt += "<|audio|>"
            sample = datasets.VoiceSample.from_prompt_and_file(prompt, audio)
        else:
            sample = datasets.VoiceSample.from_prompt(prompt)
            prompt_to_return = ""

        if len(sample.messages) != 1:
            raise ValueError(
                f"Expected exactly 1 message in sample but got {len(sample.messages)}"
            )

        output = inference.infer(
            sample,
            max_tokens=args.max_new_tokens,
            temperature=temperature,
        )

        chatbot = chatbot + [(None, output.text)]
        return chatbot, gr.update(value=prompt_to_return)

    def process_text(chatbot, prompt, temperature):
        return process_turn(chatbot, prompt, None, temperature)

    def process_audio(chatbot, prompt, audio, temperature):
        return process_turn(chatbot, prompt, audio, temperature)

    def gradio_reset():
        inference.reset_history()
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
                    placeholder=demo_instruction,
                    value=args.default_prompt,
                    container=True,
                )
            with gr.Column(scale=1):
                temperature = gr.Slider(
                    minimum=0,
                    maximum=5.0,
                    value=0,
                    step=0.1,
                    interactive=True,
                    label="temperature",
                )

        prompt.submit(add_text, [chatbot, prompt], [chatbot], queue=False).then(
            process_text,
            [chatbot, prompt, temperature],
            [chatbot, prompt],
            queue=False,
        )
        audio.stop_recording(add_audio, [chatbot, audio], [chatbot], queue=False).then(
            process_audio,
            [chatbot, prompt, audio, temperature],
            [chatbot, prompt],
            queue=False,
        )
        reset.click(gradio_reset, [], [chatbot, prompt, audio], queue=False)

    demo.launch(share=True)


if __name__ == "__main__":
    main()
