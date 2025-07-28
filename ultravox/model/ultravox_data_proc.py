from typing import Any, Dict, Optional

import numpy as np

from ultravox import data as datasets
from ultravox.model import ultravox_config
from ultravox.model import ultravox_processing


class UltravoxDataproc(datasets.Dataproc):
    def __init__(
        self,
        dataset: datasets.SizedIterableDataset,
        processor: ultravox_processing.UltravoxProcessor,
        loss_mask_type: ultravox_config.LossMaskType,
        augmentation: Optional[datasets.Augmentation] = None,
        inference_mode: bool = False,
        include_alt_fields: bool = False,
        max_response_tokens: Optional[int] = None,
        chat_template: Optional[str] = None,
    ) -> None:
        """
        Pre-processing for the Ultravox model: applies tokenization and audio processing using the UltravoxProcessor
        and prepares the shape of the data for being fed into the model.

        Args:
            dataset: The dataset to wrap/preprocess.
            processor: The processor.
            augmentation: The augmentation to apply to the audio.
            inference_mode: If True, only the input message is included in input_ids and labels, and the assistant
                message is removed from the sample. This is used for inference (e.g. testing) since the model should
                generate the assistant message. For training and validation, this should be False.
            include_alt_fields: If True, the alt_input_ids, alt_attention_mask, and alt_labels are included in the output,
                computed with <|audio|> replaced by the audio transcript.
        """
        super().__init__(dataset)
        self.processor = processor
        self.augmentation = augmentation
        self.inference_mode = inference_mode
        self.include_alt_fields = include_alt_fields
        self.max_response_tokens = max_response_tokens
        self.chat_template = chat_template
        self.loss_mask_type = loss_mask_type

    def _compute_loss_mask_len(
        self, sample: datasets.VoiceSample, audio: Optional[np.ndarray]
    ) -> int:
        # TODO: this might be slow due to calling audio_processor twice. We can compute modified input_text_len directly too.
        # Revisit when using WhisperProcessor.
        # Computing the length of the mask.
        if self.loss_mask_type == ultravox_config.LossMaskType.AFTER_AUDIO:
            user_text = self.processor.tokenizer.apply_chat_template(
                sample.messages, tokenize=False, chat_template=self.chat_template
            )
            user_text = user_text.split("<|audio|>")[0] + "<|audio|>"
            loss_mask_len = self.processor(
                text=user_text,
                audios=audio,
                sampling_rate=sample.sample_rate,
            )["input_ids"].shape[-1]

        elif self.loss_mask_type == ultravox_config.LossMaskType.LAST_ASSISTANT:
            user_text = self.processor.tokenizer.apply_chat_template(
                sample.messages[:-1], tokenize=False, chat_template=self.chat_template
            )
            loss_mask_len = self.processor(
                text=user_text,
                audios=audio,
                sampling_rate=sample.sample_rate,
            )["input_ids"].shape[-1]

        elif self.loss_mask_type == ultravox_config.LossMaskType.ALL:
            # This does not work with KL loss.
            loss_mask_len = 0
        return loss_mask_len

    def _process(self, sample: datasets.VoiceSample) -> Dict[str, Any]:
        if self.augmentation:
            sample = self.augmentation.apply_sample(sample)

        if self.inference_mode:
            # remove the assistant message from the sample so that the model can generate it
            sample.messages = sample.messages[:-1]

        text = self.processor.tokenizer.apply_chat_template(
            sample.messages, tokenize=False, chat_template=self.chat_template
        )

        # Process audio and text using UltravoxProcessor.
        # Audio is expanded to be a [C x M] array, although C=1 for mono audio.
        audio = (
            np.expand_dims(sample.audio, axis=0) if sample.audio is not None else None
        )
        inputs = self.processor(
            text=text,
            audios=audio,
            return_tensors="pt",
            sampling_rate=sample.sample_rate,
        )

        # Extract input_ids, attention_mask, and audio_values from the processed inputs
        input_ids = inputs["input_ids"].squeeze_(0)
        inputs["attention_mask"].squeeze_(0)

        # No need to shift the labels as the model does it internally
        labels = input_ids.clone()

        loss_mask_len = self._compute_loss_mask_len(sample, audio)

        labels[:loss_mask_len] = -100

        # If include_alt_fields is True, also include alt_input_ids, alt_attention_mask, and alt_labels
        if self.include_alt_fields:
            # sample.audio_transcript should never be None but currently not gauranteed, need to be investigated.
            alt_text = text.replace("<|audio|>", sample.audio_transcript or "")

            alt_inputs = self.processor(
                text=alt_text,
                audio=None,
                return_tensors="pt",
            )
            alt_input_ids = alt_inputs["input_ids"].squeeze_(0)
            alt_inputs["attention_mask"].squeeze_(0)

            alt_loss_mask_len = loss_mask_len + len(alt_input_ids) - len(input_ids)
            alt_labels = alt_input_ids.clone()
            alt_labels[:alt_loss_mask_len] = -100

            inputs["alt_input_ids"] = alt_input_ids
            inputs["alt_attention_mask"] = alt_inputs["attention_mask"]
            inputs["alt_labels"] = alt_labels.tolist()

        # Truncate the input_ids and labels if the response is longer than max_response_tokens
        if self.max_response_tokens and loss_mask_len + self.max_response_tokens < len(
            input_ids
        ):
            max_tokens = loss_mask_len + self.max_response_tokens
            inputs["input_ids"] = inputs["input_ids"][:max_tokens]
            inputs["attention_mask"] = inputs["attention_mask"][:max_tokens]
            labels = labels[:max_tokens]
            if self.include_alt_fields:
                max_alt_tokens = alt_loss_mask_len + self.max_response_tokens
                inputs["alt_input_ids"] = inputs["alt_input_ids"][:max_alt_tokens]
                inputs["alt_attention_mask"] = inputs["alt_attention_mask"][
                    :max_alt_tokens
                ]
                inputs["alt_labels"] = inputs["alt_labels"][:max_alt_tokens]

        return {
            # input_ids, attention_mask, audio_values, audio_token_start_idx, audio_token_len
            # if include_alt_fields is True, also include alt_input_ids, alt_attention_mask, alt_labels
            **inputs,
            "labels": labels.tolist(),  # Handle excessive warnings from HF
        }
