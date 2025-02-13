import copy
import threading
from concurrent import futures
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import transformers

from ultravox import data as datasets
from ultravox.inference import base
from ultravox.model import ultravox_processing

SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 1024


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
        self.data_collator = ultravox_processing.DataCollatorForSeq2SeqWithAudio(
            tokenizer=self.tokenizer,
            include_alt_fields=False,
        )

        assert self.tokenizer.padding_side == "left"

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

    def _build_past_messages(
        self,
        query_messages: List[Dict[str, str]],
        audio_token_len: int,
        response_content: str,
    ) -> List[Dict[str, str]]:
        messages = copy.copy(query_messages)
        if audio_token_len > 0:
            user_content = messages[-1]["content"]
            if user_content.count("<|audio|>") != 1:
                raise ValueError(
                    f"Expected 1 audio placeholder, found {user_content.count('<|audio|>')}"
                )
            messages[-1]["content"] = user_content.replace(
                "<|audio|>", self.tokenizer.eos_token * audio_token_len
            )
        messages.append({"role": "assistant", "content": response_content})
        return messages

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
            audio_token_len = inputs.get("audio_token_len", [0])[0]
            past_messages = self._build_past_messages(
                extended_sample.messages, audio_token_len, output_text
            )
            self.update_conversation(past_messages, output.past_key_values)
        return base.VoiceOutput(output_text, input_len, output_len)

    # Note: infer_batch doesn't support conversation mode or caching yet.
    def infer_batch(
        self,
        samples: List[datasets.VoiceSample],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[base.VoiceOutput]:
        assert not self.conversation_mode
        inputs = [self._dataproc(s) for s in samples]
        for input in inputs:
            for key, val in input.items():
                if not key.startswith("audio"):
                    input[key] = val.squeeze(0)

        tensors = self.data_collator(inputs)
        # Move non-None tensors to the same device as the model
        tensors = {
            k: v.to(self.model.device) if v is not None else v
            for k, v in tensors.items()
        }
        input_len = tensors["input_ids"].shape[1]
        output_batch = self._generate(
            tensors, max_tokens, temperature, return_dict_in_generate=False
        )
        output_texts = []
        for output in output_batch:
            output_tokens = output[input_len:]
            output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            output_len = len(output_tokens)
            output_text = base.VoiceOutput(output_text, input_len, output_len)
            output_texts.append(output_text)
        return output_texts

    def infer_stream(
        self,
        sample: datasets.VoiceSample,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> base.InferenceGenerator:
        extended_sample = self._get_sample_with_past(sample)
        inputs = self._dataproc(extended_sample)
        input_tokens = inputs["input_ids"].shape[1]
        streamer = transformers.TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        def thunk(f: futures.Future):
            result = self._generate(
                inputs, max_tokens, temperature, streamer, self.past_key_values
            )
            f.set_result(result)

        future: futures.Future[transformers.GenerateDecoderOnlyOutput] = (
            futures.Future()
        )
        thread = threading.Thread(target=thunk, args=(future,))
        thread.start()
        output_text = ""
        output_token_len = 0
        for chunk in streamer:
            if chunk:
                output_text += chunk
                output_token_len += 1
                yield base.InferenceChunk(chunk)
        thread.join()
        output = future.result()
        if self.conversation_mode:
            audio_token_len = inputs.get("audio_token_len", [0])[0]
            past_messages = self._build_past_messages(
                extended_sample.messages, audio_token_len, output_text
            )
            self.update_conversation(past_messages, output.past_key_values)
        yield base.InferenceStats(input_tokens, output_token_len)

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
        return_dict_in_generate: Optional[bool] = True,
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
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=terminators,
            streamer=streamer,
            past_key_values=past_key_values,
            return_dict_in_generate=return_dict_in_generate,
        )
