import threading
from typing import List, Optional

import librosa
import numpy as np
import torch
import transformers

from ultravox.data import datasets
from ultravox.inference import base
from ultravox.model import ultravox_processing

SAMPLE_RATE = 16000
MAX_TOKENS = 1024
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
    ):
        self.model = model.to(device).to(dtype).eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.dtype = dtype

    def infer(
        self,
        samples: List[datasets.VoiceSample],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> base.VoiceOutput:
        inputs = self._dataproc(samples)
        input_len = inputs["input_ids"].shape[1]
        outputs = self._generate(inputs, max_tokens, temperature)
        print("actual raw output", outputs)
        print("raw output shape", outputs.shape)

        return_vals = []
        for output in outputs:
            output_tokens = output[input_len:]
            output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            output_len = len(output_tokens)
            return_vals.append(base.VoiceOutput(output_text, input_len, output_len))
        return return_vals

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

        thread_args = (inputs, max_tokens, temperature, streamer)
        thread = threading.Thread(target=self._generate, args=thread_args)
        thread.start()
        output_tokens = 0
        for chunk in streamer:
            if chunk:
                yield base.InferenceChunk(chunk)
                output_tokens += 1
        yield base.InferenceStats(input_tokens, output_tokens)
        thread.join()

    def _dataproc(self, samples: List[datasets.VoiceSample]):
        text_inputs = [
            self.tokenizer.apply_chat_template(
                sample.messages, add_generation_prompt=True, tokenize=False
            )
            for sample in samples
        ]

        audio_inputs = []
        for sample in samples:
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
            audio_inputs.append(audio_input)

        # Check if audio inputs are uniform and match text inputs
        if audio_inputs and len(audio_inputs) != len(text_inputs):
            raise ValueError(
                f"Mismatch between number of text inputs ({len(text_inputs)}) and audio inputs ({len(audio_inputs)})"
            )

        # check if audio inputs are uniform
        if not all(x is None for x in audio_inputs) and not all(
            x is not None for x in audio_inputs
        ):
            raise ValueError(
                "Audio batch must be uniform. All elements in the audio batch must either be None or all must contain audio data."
            )
        audio_inputs = audio_inputs if audio_inputs[0] is not None else None

        # check if text inputs are uniform
        if not all(x is None for x in text_inputs) and not all(
            x is not None for x in text_inputs
        ):
            raise ValueError(
                "Text batch must be uniform. All elements in the text batch must either be None or all must contain text data."
            )
        text_inputs = text_inputs if text_inputs[0] is not None else None

        inputs = self.processor(
            audios=audio_inputs,
            texts=text_inputs,
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
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        streamer: Optional[transformers.TextStreamer] = None,
    ):
        print("generate shape", inputs.shape)
        temperature = temperature or None
        do_sample = temperature is not None

        terminators = [self.tokenizer.eos_token_id]
        if "<|eot_id|>" in self.tokenizer.added_tokens_encoder:
            terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        return self.model.generate(
            **inputs,
            do_sample=do_sample,
            max_new_tokens=max_tokens or MAX_TOKENS,
            temperature=temperature,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=terminators,
            streamer=streamer,
        )
