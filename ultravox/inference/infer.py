import copy
import threading
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import transformers

from ultravox.data import datasets
from ultravox.inference import base
from ultravox.model import ultravox_processing

SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 1024
# Without this penalty, the model tends to repeat itself.
REPETITION_PENALTY = 1.1


class LocalInference(base.VoiceInference):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        processor: ultravox_processing.UltravoxProcessor,
        tokenizer: transformers.PreTrainedTokenizer,
        device: str,
        dtype: torch.dtype,
        conversation_mode: bool = False,
    ):
        self.model = model.to(device).to(dtype).eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.dtype = dtype

        self.conversation_mode = conversation_mode
        self.past_messages: List[Dict[str, str]] = []
        self.past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = (
            None
        )

    def update_conversation(
        self,
        past_messages: List[Dict[str, str]] = [],
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
    ):
        self.past_messages = past_messages
        self.past_key_values = past_key_values

    def _get_sample_with_past(
        self, sample: datasets.VoiceSample
    ) -> datasets.VoiceSample:
        sample = copy.copy(sample)
        sample.add_past_messages(self.past_messages)
        return sample

    def infer(
        self,
        sample: datasets.VoiceSample,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> base.VoiceOutput:
        extended_sample = self._get_sample_with_past(sample)
        inputs = self._dataproc(extended_sample)
        input_len = inputs["input_ids"].shape[1]
        output = self._generate(
            inputs, max_tokens, temperature, past_key_values=self.past_key_values
        )
        output_tokens = output.sequences[0][input_len:]
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        output_len = len(output_tokens)

        if self.conversation_mode:
            past_messages = copy.deepcopy(extended_sample.messages)
            audio_token_len = (
                0 if "audio_token_len" not in inputs else inputs["audio_token_len"][0]
            )
            if audio_token_len > 0:
                user_content = past_messages[-1]["content"]
                if user_content.count("<|audio|>") != 1:
                    raise ValueError(
                        f"Expected 1 audio placeholder, found {user_content.count('<|audio|>')}"
                    )
                past_messages[-1]["content"] = user_content.replace(
                    "<|audio|>", self.tokenizer.eos_token * audio_token_len
                )
            past_messages.append({"role": "assistant", "content": output_text})
            self.update_conversation(past_messages, output.past_key_values)

        return base.VoiceOutput(output_text, input_len, output_len)

    # streaming is not supported in conversation mode yet, to be implemented
    def infer_stream(
        self,
        sample: datasets.VoiceSample,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> base.InferenceGenerator:
        inputs = self._dataproc(sample)
        input_tokens = inputs["input_ids"].shape[1]
        decode_kwargs = {"skip_special_tokens": True}
        streamer = transformers.TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, decode_kwargs=decode_kwargs
        )

        thread_args = (
            inputs,
            max_tokens,
            temperature,
            streamer,
        )
        thread = threading.Thread(target=self._generate, args=thread_args)
        thread.start()
        output_tokens = 0
        for chunk in streamer:
            if chunk:
                yield base.InferenceChunk(chunk)
                output_tokens += 1
        yield base.InferenceStats(input_tokens, output_tokens)
        thread.join()

    def _dataproc(self, sample: datasets.VoiceSample):
        text_input = self.tokenizer.apply_chat_template(
            sample.messages, add_generation_prompt=True, tokenize=False
        )
        if sample.audio is not None:
            audio = sample.audio
            sample_rate = sample.sample_rate
            # Normalize audio to float32.
            if audio.dtype == np.int16:
                audio = audio / np.float32(32768.0)
            if audio.dtype not in [np.float64, np.float32]:
                raise ValueError("Audio must be float64 or float32 or int16")

            # Convert to tensor, resampling to 16kHz if needed.
            if sample_rate != SAMPLE_RATE:
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE
                )
            audio_input = torch.from_numpy(audio)
            # Squeeze from [1, T] to [T] if needed.
            if sample.audio.ndim == 2:
                audio_input = audio_input.squeeze(0)
        else:
            audio_input = None

        inputs = self.processor(
            audio=audio_input,
            text=text_input,
            return_tensors="pt",
            sampling_rate=SAMPLE_RATE,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        if "audio_values" in inputs:
            inputs["audio_values"] = inputs["audio_values"].to(dtype=self.dtype)
        return inputs

    @torch.inference_mode()
    def _generate(
        self,
        inputs: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        streamer: Optional[transformers.TextStreamer] = None,
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
    ):
        temperature = temperature or None
        do_sample = temperature is not None

        terminators = [self.tokenizer.eos_token_id]
        if "<|eot_id|>" in self.tokenizer.added_tokens_encoder:
            terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        return self.model.generate(
            **inputs,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
            temperature=temperature,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=terminators,
            streamer=streamer,
            past_key_values=past_key_values,
            return_dict_in_generate=True,
        )
