from dataclasses import dataclass
from typing import Tuple, List, Optional, Union, Dict

import gradio as gr
import numpy as np
import simple_parsing
import transformers

from ultravox.data import datasets
from ultravox.inference import ultravox_infer


@dataclass
class DemoConfig:
    model_path: str = "wandb://fixie/ultravox/model-zhuang.2024-07-31-ultravox.blsp-kd-2-tinyllama:v5"
    default_prompt: str = ""
    max_new_tokens: int = 256


class History:
    def __init__(self, audio_token_replacement: str, audio_placeholder: str = "<|audio|>"):
        self.past_messages: List[Dict[str, str]] = []
        self.past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None
        self.audio_token_replacement = audio_token_replacement
        self.audio_placeholder = audio_placeholder

    def update_messages(self, messages: List[Dict[str, str]], audio_token_len: int):
        self.past_messages = messages.copy()
        if audio_token_len > 0:
            assert self.audio_placeholder in self.past_messages[-1]['content'], "Audio placeholder not found in the last message"
            self.past_messages[-1]['content'] = self.past_messages[-1]['content'].replace(
                self.audio_placeholder, self.audio_token_replacement * audio_token_len
            )
        else:
            assert self.audio_placeholder not in self.past_messages[-1]['content'], "Unexpected audio placeholder found"

    def add_assistant_response(self, content: str):
        self.past_messages.append({'role': "assistant", 'content': content})
    
    @property
    def messages(self):
        return self.past_messages


def main():
    global args, history, inference

    args = simple_parsing.parse(config_class=DemoConfig)
    inference = ultravox_infer.UltravoxInference(args.model_path)
    history = History(audio_token_replacement=inference.tokenizer.eos_token)

    def add_text(chatbot: gr.Chatbot, text: str) -> gr.Chatbot:
        return chatbot + [(text, None)]
    
    def add_audio(chatbot: gr.Chatbot, audio: str) -> gr.Chatbot:
        return chatbot + [((audio,), None)]

    def disable_interaction():
        return gr.update(interactive=False), gr.update(interactive=False)
    
    def enable_interaction():
        return gr.update(interactive=True), gr.update(interactive=True)

    def process_turn(chatbot: gr.Chatbot, prompt: str, audio: Optional[Tuple[int, np.ndarray]] = None, num_beams: int = 1, temperature: float = 1.0):
        updated_prompt = prompt
        if audio:
            if "<|audio|>" not in prompt:
                prompt += "<|audio|>"
            sample = datasets.VoiceSample.from_prompt_and_file(prompt, audio)
        else:
            sample = datasets.VoiceSample.from_prompt(prompt)
            updated_prompt = ""

        sample.add_past_messages(history.messages)

        output = inference.infer(
            sample,
            max_new_tokens=args.max_new_tokens,
            past_key_values=history.past_key_values,
            num_beams=num_beams,
            temperature=temperature
        )
        history.past_key_values = output.past_key_values

        history.update_messages(sample.messages, output.audio_token_len)
        history.add_assistant_response(output.text)

        chatbot = chatbot + [(None, output.text)]
        return chatbot, gr.update(value=updated_prompt)

    def gradio_reset():
        global history
        history = History(audio_token_replacement=inference.tokenizer.eos_token)
        return [], "", None

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        
        with gr.Row():
            with gr.Column(scale=0.2, min_width=0):
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    interactive=True,
                    label="beam",
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    label="temperature",
                )
            with gr.Column(scale=0.08, min_width=0):
                clear = gr.Button("Reset")
            with gr.Column(scale=0.85):
                prompt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text (include <|audio|> to insert audio) and press enter",
                    value=args.default_prompt,
                    container=True)
            with gr.Column(scale=0.2, min_width=0):
                audio = gr.Audio(
                    label="🎤",
                    sources=["microphone"],
                    type="filepath",
                    visible=True,
                )

        def process_text(chatbot, prompt, num_beams, temperature):
            return process_turn(chatbot, prompt, None, num_beams, temperature)

        def process_audio(chatbot, prompt, audio, num_beams, temperature):
            return process_turn(chatbot, prompt, audio, num_beams, temperature)

        prompt.submit(disable_interaction, [], [prompt, audio], queue=False).then(
            add_text, [chatbot, prompt], [chatbot], queue=False
        ).then(
            process_text, [chatbot, prompt, num_beams, temperature], [chatbot, prompt], queue=False
        ).then(
            enable_interaction, [], [prompt, audio], queue=False
        )

        audio.stop_recording(disable_interaction, [], [prompt, audio], queue=False).then(
            add_audio, [chatbot, audio], [chatbot], queue=False
        ).then(
            process_audio, [chatbot, prompt, audio, num_beams, temperature], [chatbot, prompt], queue=False
        ).then(
            enable_interaction, [], [prompt, audio], queue=False
        )

        clear.click(gradio_reset, [], [chatbot, prompt, audio], queue=False)

    demo.launch(share=True)


if __name__ == "__main__":
    main()