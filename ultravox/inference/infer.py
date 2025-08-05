import copy
import re
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
        dtype: torch.dtype,
        conversation_mode: bool = False,
        chat_template: Optional[str] = None,
        enable_thinking: bool = False,
        thinking_regex: Optional[str] = None,
    ):
        self.model = model.to(dtype).eval()
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
        self.chat_template = chat_template
        self.enable_thinking = enable_thinking
        self.thinking_regex = thinking_regex
        assert self.tokenizer.padding_side == "left"

    def update_conversation(
        self,
        past_messages: List[Dict[str, str]] = [],
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
    ):
        self.past_messages = past_messages
        self.past_key_values = past_key_values

    def _get_sample_with_past(
        self, sample: Optional[datasets.VoiceSample] = None
    ) -> datasets.VoiceSample:
        # Workaround for if we want to generate an assistant response without a user query
        if sample is None:
            if len(self.past_messages) == 0:
                raise ValueError("No past messages available to generate a response.")
            sample = datasets.VoiceSample(
                self.past_messages,
            )
        else:
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

    def _postprocess_response(self, text: str) -> Tuple[str, Optional[str]]:
        """Post-process the model's response text.

        This method handles post-processing of the model's response by separating the response text
        from the thinking content and returns them as a tuple.

        Args:
            text: The model's response text to process.

        Returns:
            Tuple of (response_text, thinking_content). If thinking is disabled, thinking_content will be None.

        Raises:
            ValueError: If thinking is enabled but thinking_regex is not set, or if thinking content
            is not found in the response when thinking is enabled.
        """
        if not self.enable_thinking:
            return text, None

        if not self.thinking_regex:
            raise ValueError("thinking_regex is not set while enable_thinking is True")

        match = re.search(self.thinking_regex, text, re.DOTALL)
        if not match:
            raise ValueError(
                f"{self.thinking_regex} not matched in the response while thinking is enabled: {text}"
            )

        thinking_content = match.group(1).strip()
        response_text = re.sub(self.thinking_regex, "", text, flags=re.DOTALL).strip()
        return response_text, thinking_content

    def infer(
        self,
        sample: Optional[datasets.VoiceSample] = None,
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

        response_text, thinking_content = self._postprocess_response(output_text)
        output_len = len(output_tokens)

        if self.conversation_mode:
            audio_token_len = inputs.get("audio_token_len", [0])[0]
            past_messages = self._build_past_messages(
                extended_sample.messages, audio_token_len, response_text
            )
            self.update_conversation(past_messages, output.past_key_values)

        return base.VoiceOutput(
            response_text, input_len, output_len, thinking_content=thinking_content
        )

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

            response_text, thinking_content = self._postprocess_response(output_text)
            output_len = len(output_tokens)
            output_texts.append(
                base.VoiceOutput(
                    response_text,
                    input_len,
                    output_len,
                    thinking_content=thinking_content,
                )
            )

        return output_texts

    def infer_stream(
        self,
        sample: Optional[datasets.VoiceSample] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> base.InferenceGenerator:
        extended_sample = self._get_sample_with_past(sample)

        # First pass: Process input without generation prompt to build KV cache
        inputs = self._dataproc(extended_sample, add_generation_prompt=False)

        # Build KV cache for input
        output = self._generate(
            inputs,
            max_new_tokens=1,  # Don't generate any new tokens
            temperature=temperature,
            past_key_values=self.past_key_values,
        )
        # Make a deep copy of past_key_values to preserve it
        preserved_past_key_values = copy.deepcopy(output.past_key_values)
        past_key_values = output.past_key_values
        del output

        # Second pass: Process with generation prompt but don't expand cache
        inputs = self._dataproc(extended_sample, add_generation_prompt=True)
        input_tokens = inputs["input_ids"].shape[1]

        streamer = transformers.TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        def thunk(f: futures.Future):
            result = self._generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                streamer=streamer,
                past_key_values=past_key_values,  # Use separate copy for generation
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
        future.result()  # Wait for generation to complete

        response_text, _ = self._postprocess_response(output_text)

        if self.conversation_mode:
            audio_token_len = inputs.get("audio_token_len", [0])[0]
            past_messages = self._build_past_messages(
                extended_sample.messages, audio_token_len, response_text
            )
            self.update_conversation(
                past_messages, preserved_past_key_values
            )  # Use preserved past_key_values from first step

        yield base.InferenceStats(input_tokens, output_token_len)

    def _dataproc(
        self, sample: datasets.VoiceSample, add_generation_prompt: bool = True
    ):
        text_input = self.tokenizer.apply_chat_template(
            sample.messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            chat_template=self.chat_template,
            enable_thinking=self.enable_thinking,
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
        sampling_args = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens or MAX_NEW_TOKENS,
        }
        if temperature is not None and temperature > 0:
            sampling_args["do_sample"] = True
        else:
            sampling_args["do_sample"] = False
            sampling_args["top_p"] = None
            sampling_args["top_k"] = None

        terminators = [self.tokenizer.eos_token_id]
        if "<|eot_id|>" in self.tokenizer.added_tokens_encoder:
            terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        return self.model.generate(
            **inputs,
            **sampling_args,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=terminators,
            streamer=streamer,
            past_key_values=past_key_values,
            return_dict_in_generate=return_dict_in_generate,
        )
