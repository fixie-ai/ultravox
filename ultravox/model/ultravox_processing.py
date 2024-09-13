import re
from typing import Optional, Union

import numpy as np
import torch
import transformers

from .ultravox_config import UltravoxConfig


# TODO: update the comments to reflect the actual implementation
class UltravoxProcessor(transformers.ProcessorMixin):
    """
    Constructs an Ultravox processor which wraps an audio processor and a tokenizer into a single processor.

    Args:
        audio_processor: The audio processor for the audio encoder.
        tokenizer: The tokenizer for the language model.
    """

    attributes = ["audio_processor", "tokenizer"]
    audio_processor_class = (
        "Wav2Vec2Processor",
        "SeamlessM4TFeatureExtractor",
        "WhisperProcessor",
    )
    tokenizer_class = (
        "PreTrainedTokenizer",
        "PreTrainedTokenizerFast",
    )

    tokenizer: transformers.PreTrainedTokenizerBase
    audio_processor: transformers.ProcessorMixin

    def __init__(
        self,
        audio_processor=None,
        tokenizer=None,
        audio_padding: str = "longest",
        audio_placeholder: str = "<|audio|>",
    ):
        """
        Args:
            audio_processor: The audio processor for the audio encoder.
            tokenizer: The tokenizer for the language model.
            audio_padding: The padding strategy for the audio encoder.
            audio_placeholder: The placeholder for the audio in the text.
        """
        self.audio_padding = audio_padding
        self.audio_placeholder = audio_placeholder
        # The tokenizer treats spaces around special tokens differently depending on the context.
        # When using the audio placeholder <|audio|>, we should always ignore any extra surrounding spaces
        # to maintain consistency in tokenization, as extra spaces are meaningless in this context.
        # When part of the text is marked by <|audio|> indicating audio input, we should tokenize
        # the three segments separated by the audio placeholder independently. This ensures consistency
        # between text input and audio input.
        self.audio_placeholder_regex = re.compile(
            rf" *{re.escape(audio_placeholder)} *"
        )
        self.audio_token_replacement = tokenizer.eos_token
        assert (
            self.audio_token_replacement is not None
        ), "The tokenizer has no EOS token. Cannot recover."
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(audio_processor=audio_processor, tokenizer=tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config: UltravoxConfig = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        audio_processor = transformers.AutoProcessor.from_pretrained(
            config.audio_model_id
            or config.audio_config._name_or_path
            or "facebook/wav2vec2-base-960h"
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        return cls(
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            stack_factor=config.stack_factor,
        )

    def __call__(
        self,
        text: Optional[str] = None,
        audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        transcript: Optional[str] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[
            Union[str, transformers.TensorType]
        ] = transformers.TensorType.PYTORCH,
        **kwargs,
    ) -> transformers.BatchFeature:
        """
        Main method to prepare for the model one text sequence and audio. This method forwards the `text`
        and `kwargs` arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio`, `sampling_rate` and `kwargs` arguments to
        audio processor's [`~Wav2Vec2Processor.__call__`] if `audio` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`):
                The sequence to be encoded. Sequence can be a string or (pretokenized string).
            audio (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The audio to be prepared. Audio can be NumPy array or PyTorch tensor. In case of a
                NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T the
                sample length of the audio.
            sampling_rate (`int`, *optional*, defaults to 16000):
                Sampling rate of the input audio. We expect 16kHz audio. Don't change this value unless you know what
                you are doing.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **audio_values** -- Processed audio values to be fed to a model. Returned when `audio` is not `None`.
            - **audio_token_len** -- Predicted number of audio frames: this value is guaranteed to be a close upper bound.
              Returned when `audio` is not `None`.
            - **audio_token_start_idx** -- The index in the tokenized text where the audio starts. Returned when `audio` is not `None`.
        """
        # TODO: Add support for multiple audio and text inputs.

        if not text or not isinstance(text, str):
            raise ValueError(
                f"Non-empty text of type str (found {type(text)}) must be provided, regardless of whether it contains audio_placeholder."
            )

        data = {}
        if audio is not None and len(audio) > 0:
            # We don't enforce non-empty transcript here, as the audio may not contain any speech or the transcript may not be provided as during inference.
            if self.audio_padding == "max_length":
                # 30 seconds is the expected length for Whisper
                assert sampling_rate is not None, "Sampling rate must be provided."
                num_audio_samples = 30 * sampling_rate
            else:
                num_audio_samples = audio.shape[-1]

            # Process audio into features before passing to the audio encoder.
            x = self.audio_processor(
                audio,
                sampling_rate=sampling_rate,
                padding="longest",
                max_length=num_audio_samples,
                **kwargs,
            )
            if "input_features" in x:
                data["audio_values"] = x.input_features
            else:
                data["audio_values"] = x.input_values
            # Note: ideally we should compute audio_len from x.attention_mask, but the HF WhisperFeatureExtractor implementation of attention_mask is off by 1 
            data["audio_len"] = [data["audio_values"].shape[-1]]

            if transcript:
                tokenized_transcript = self.tokenizer(
                    transcript, add_special_tokens=False, **kwargs
                )
                data["transcript_ids"] = [tokenized_transcript.input_ids]
                data["transcript_len"] = [len(tokenized_transcript.input_ids)]
            else:
                data["transcript_ids"] = [[]]
                data["transcript_len"] = [0]

        # Find and process audio placeholders
        matches = list(self.audio_placeholder_regex.finditer(text))
        if len(matches) == 0:
            # No audio placeholder found, tokenize the entire text
            tokenized = self.tokenizer([text], add_special_tokens=False, **kwargs)
            data.update(tokenized)
        elif len(matches) == 1:
            match = matches[0]
            if len(data["audio_values"]) == 0:
                raise ValueError(
                    f"audio must be non-empty when using audio placeholder ({self.audio_placeholder}) in text."
                )

            # The text before and after the audio placeholder is tokenized independently and concatenated to ensure consistency when tokens from the transcript or audio are inserted.
            text_pre_audio = text[: match.start()]
            text_post_audio = text[match.end() :]

            tokenized_pre_audio = self.tokenizer(
                text_pre_audio, add_special_tokens=False, **kwargs
            )
            tokenized_post_audio = self.tokenizer(
                text_post_audio, add_special_tokens=False, **kwargs
            )

            data["input_ids"] = [
                tokenized_pre_audio.input_ids + tokenized_post_audio.input_ids
            ]
            data["attention_mask"] = [
                tokenized_pre_audio.attention_mask + tokenized_post_audio.attention_mask
            ]
            data["audio_start_idx"] = [len(tokenized_pre_audio.input_ids)]

        else:
            raise ValueError(
                f"Multiple audio placeholders (found {len(matches)}) are not supported yet."
            )
        return transformers.BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names
        return list(set(tokenizer_input_names + audio_processor_input_names))


UltravoxProcessor.register_for_auto_class()

transformers.AutoProcessor.register(UltravoxConfig, UltravoxProcessor)
