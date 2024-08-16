import threading
from typing import Any, List, Optional

import librosa
import numpy as np
import torch
import transformers
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader

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

    def batch_infer(
        self,
        all_samples: List[datasets.VoiceSample],
        batch_size: int = 1,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> base.InferenceGenerator:
        data_collator = datasets.DataCollatorForSeq2SeqWithAudio(
            tokenizer=self.tokenizer,
            include_alt_fields=False,
        )
        sample_additional_info = []
        for i, sample in enumerate(all_samples):
            question_text = sample.audio_transcript
            expected_answer = sample.messages[-1]["content"]
            sample_additional_info.append(
                {
                    "index": i,
                    "question_text": question_text,
                    "expected_answer": expected_answer,
                }
            )
            sample.messages = sample.messages[:-1]

        dataset: Dataset[Any] = Dataset.from_list(
            [self._dataproc(s, batch=True) for s in all_samples]
        )
        dataloader: DataLoader = DataLoader(
            dataset, collate_fn=data_collator, batch_size=batch_size
        )
        sample_index = 0
        for batch in dataloader:
            input_len = batch["input_ids"].shape[1]
            output_batch = self._generate(batch, max_tokens, temperature)
            for _, output in enumerate(output_batch):
                output_tokens = output[input_len:]
                output_text = self.tokenizer.decode(
                    output_tokens, skip_special_tokens=True
                )
                output_len = len(output_tokens)
                output_text = base.VoiceOutput(output_text, input_len, output_len)
                output = {
                    "output_text": output_text,
                    "question_text": sample_additional_info[sample_index][
                        "question_text"
                    ],
                    "expected_answer": sample_additional_info[sample_index][
                        "expected_answer"
                    ],
                    "index": sample_additional_info[sample_index]["index"],
                }
                sample_index += 1
                yield output

    def infer(
        self,
        sample: datasets.VoiceSample,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> base.VoiceOutput:
        inputs = self._dataproc(sample)
        input_len = inputs["input_ids"].shape[1]
        output = self._generate(inputs, max_tokens, temperature)
        output_tokens = output[0][input_len:]
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        output_len = len(output_tokens)
        return base.VoiceOutput(output_text, input_len, output_len)

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

    def _dataproc(self, sample: datasets.VoiceSample, batch=False):
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
        if batch:
            for key, val in inputs.items():
                inputs[key] = val.squeeze(0)
        return inputs

    @torch.inference_mode()
    def _generate(
        self,
        inputs: torch.Tensor,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        streamer: Optional[transformers.TextStreamer] = None,
    ):
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
