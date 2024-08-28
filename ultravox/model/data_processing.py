from typing import Any, Dict

import datasets
import numpy as np
from torch.utils import data

from ultravox.data import datasets
from ultravox.model import ultravox_processing


class UltravoxDataproc(datasets.Dataproc):
    def __init__(
        self,
        dataset: data.IterableDataset,
        processor: ultravox_processing.UltravoxProcessor,
        train_on_inputs: bool = False,
        inference_mode: bool = False,
        include_alt_fields: bool = False,
    ) -> None:
        """
        Pre-processing for the Ultravox model: applies tokenization and audio processing using the UltravoxProcessor
        and prepares the shape of the data for being fed into the model.

        Args:
            dataset: The dataset to wrap/preprocess.
            processor: The processor.
            train_on_inputs: If True, the token_ids for prompt (user input) are also included in the labels,
                so the model learns to predict the input message.
            inference_mode: If True, only the input message is included in input_ids and labels, and the assistant
                message is removed from the sample. This is used for inference (e.g. testing) since the model should
                generate the assistant message. For training and validation, this should be False.
            include_alt_fields: If True, the alt_input_ids, alt_attention_mask, and alt_labels are included in the output,
                computed with <|audio|> replaced by the audio transcript.
        """
        super().__init__(dataset)
        self.processor = processor
        self.train_on_inputs = train_on_inputs
        self.inference_mode = inference_mode
        if self.inference_mode:
            self.train_on_inputs = True
        self.include_alt_fields = include_alt_fields

    def _process(self, sample: datasets.VoiceSample) -> Dict[str, Any]:
        if self.inference_mode:
            # remove the assistant message from the sample so that the model can generate it
            sample.messages = sample.messages[:-1]

        text = self.processor.tokenizer.apply_chat_template(
            sample.messages, tokenize=False
        )

        # Process audio and text using UltravoxProcessor.
        # Audio is expanded to be a [C x M] array, although C=1 for mono audio.
        audio = (
            np.expand_dims(sample.audio, axis=0) if sample.audio is not None else None
        )
        inputs = self.processor(
            text=text,
            audio=audio,
            transcript=sample.audio_transcript,
            return_tensors="pt",
            sampling_rate=sample.sample_rate,
        )

        # Extract input_ids, attention_mask, and audio_values from the processed inputs
        input_ids = inputs["input_ids"].squeeze_(0)
        inputs["attention_mask"].squeeze_(0)
        if "audio_values" in inputs:
            inputs["audio_values"].squeeze_(0)
            inputs["audio_token_start_idx"].squeeze_(0)
            inputs["audio_token_len"].squeeze_(0)

        # No need to shift the labels as the model does it internally
        labels = input_ids.clone()

        if not self.train_on_inputs and sample.messages[-1]["role"] == "assistant":
            # Mask the prompt tokens and only compute loss on the assistant message, not the prompt.
            # The idea is that the model should only be able to predict the assistant message given the user message.
            # One reason is that there's very little randomness in the prompt, so the model would be forced to memorize it.
            #
            # Example (-100 is the ignore index):
            #   Tokens: <user> Transcribe\n<|audio|> </s> <assistant> Brown fox jumps over the lazy dog </s>
            #   Labels: -100   -100        -100    -100   <assistant> Brown fox jumps over the lazy dog </s>
            #
            # Note: The above might look weird because I'm mixing token IDs and text, but that's just for illustration.

            output_text = self.processor.tokenizer.apply_chat_template(
                sample.messages[-1:], tokenize=False
            )
            output_token_len = self.processor(text=output_text)["input_ids"].shape[-1]
            input_token_len = len(input_ids) - output_token_len
            labels[:input_token_len] = -100

        # If include_alt_fields is True, also include alt_input_ids, alt_attention_mask, and alt_labels
        if self.include_alt_fields:
            # sample.audio_transcript should never be None but currently not gauranteed, need to be investigated.
            alt_text = text.replace("<|audio|>", sample.audio_transcript or "")

            alt_inputs = self.processor(
                text=alt_text,
                return_tensors="pt",
            )
            alt_input_ids = alt_inputs["input_ids"].squeeze_(0)
            alt_inputs["attention_mask"].squeeze_(0)

            alt_labels = alt_input_ids.clone()
            if not self.train_on_inputs and sample.messages[-1]["role"] == "assistant":
                alt_input_token_len = (
                    input_token_len + len(alt_input_ids) - len(input_ids)
                )
                alt_labels[:alt_input_token_len] = -100

            inputs["alt_input_ids"] = alt_input_ids
            inputs["alt_attention_mask"] = alt_inputs["attention_mask"]
            inputs["alt_labels"] = alt_labels

        return {
            # input_ids, attention_mask, audio_values, audio_token_start_idx, audio_token_len
            # if include_alt_fields is True, also include alt_input_ids, alt_attention_mask, alt_labels
            **inputs,
            "labels": labels,
        }
