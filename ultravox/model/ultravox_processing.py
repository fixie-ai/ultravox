from typing import List, Optional, Union

import numpy as np
import torch
import transformers


def collate_tokens(values: List[List[any]], pad_token_id=0, padding_side="right"):
    # Convert lists to tensors
    tensors = [torch.tensor(v) for v in values]

    # Get max length
    max_length = max(len(v) for v in values)

    # Pad tensors
    padded_tensors = []
    pad_lengths = []

    for t in tensors:
        pad_length = max_length - t.size(0)
        pad_lengths.append(pad_length)

        if padding_side == "right":
            padded_tensor = torch.nn.functional.pad(
                t, (0, pad_length), value=pad_token_id
            )
        elif padding_side == "left":
            padded_tensor = torch.nn.functional.pad(
                t, (pad_length, 0), value=pad_token_id
            )
        else:
            raise ValueError("padding_side must be either 'left' or 'right'")

        padded_tensors.append(padded_tensor)

    # Stack tensors
    stacked_tensor = torch.stack(padded_tensors)

    return stacked_tensor, pad_lengths


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
        encoder_ds_factor: int = 320,
        stack_factor: int = 8,
        audio_placeholder: str = "<|audio|>",
    ):
        """
        Args:
            audio_processor: The audio processor for the audio encoder.
            tokenizer: The tokenizer for the language model.
            audio_padding: The padding strategy for the audio encoder.
            encoder_ds_factor: The downsample factor of the audio encoder.
            stack_factor: The factor by which the audio encoder output is stacked in the multimodal projector.
            audio_placeholder: The placeholder for the audio in the text.
        """
        self.audio_padding = audio_padding
        self.encoder_ds_factor = encoder_ds_factor
        self.stack_factor = stack_factor
        self.audio_placeholder = audio_placeholder
        self.audio_token_replacement = tokenizer.eos_token
        assert (
            self.audio_token_replacement is not None
        ), "The tokenizer has no EOS token. Cannot recover."
        super().__init__(audio_processor=audio_processor, tokenizer=tokenizer)

    def __call__(
        self,
        texts: Optional[List[str]] = None,
        audios: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
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

        data = {}
        if audios is not None:
            # collate audios
            audios, _ = collate_tokens(audios, 0.0, "right")
            audio_embed_frames = []
            audio_values = []
            for aud in audios:
                if self.audio_padding == "max_length":
                    # 30 seconds is the expected length for Whisper
                    assert sampling_rate is not None, "Sampling rate must be provided."
                    audio_len = 30 * sampling_rate
                else:
                    audio_len = aud.shape[-1]

                nb_encoder_frames = int(
                    round(audio_len / self.encoder_ds_factor + 1e-4)
                )
                audio_embed_frames.append(
                    int(np.ceil(nb_encoder_frames / self.stack_factor))
                )
                x = self.audio_processor(
                    aud,
                    sampling_rate=sampling_rate,
                    padding="longest",
                    max_length=audio_len,
                    **kwargs,
                )
                val = x.input_features if "input_features" in x else x.input_values
                audio_values.append(val.squeeze())

            data["audio_values"] = audio_values
            data["audio_token_len"] = audio_embed_frames

        if texts is not None:
            processed_texts = []

            for i, t in enumerate(texts):
                assert isinstance(
                    t, str
                ), f"Text must be a string. Got {type(t)} for item {i}."

                if self.audio_placeholder in t:
                    if "audio_token_len" not in data:
                        raise ValueError(
                            f"Audio must be provided when using audio placeholder ({self.audio_placeholder}) in text."
                        )
                    t = t.replace(
                        self.audio_placeholder,
                        self.audio_token_replacement * data["audio_token_len"][i],
                    )
                else:
                    raise ValueError(f"Audio must include audio_placeholder token")

                processed_texts.append(t)

            tokenized_texts = self.tokenizer(
                processed_texts, add_special_tokens=False, **kwargs
            )

            tokenized_texts["input_ids"], pad_lengths = collate_tokens(
                tokenized_texts["input_ids"], 0, "left"
            )
            tokenized_texts["attention_mask"], _ = collate_tokens(
                tokenized_texts["attention_mask"], 0, "left"
            )
            data["audio_token_start_idx"] = [
                len(
                    self.tokenizer.encode(
                        t[: t.index(self.audio_placeholder)],
                        add_special_tokens=False,
                    )
                )
                + pad_lengths[i]
                for i, t in enumerate(texts)
            ]

            data.update(tokenized_texts)

            # make sure all keys are tensors
            for key, val in data.items():
                data[key] = torch.tensor(val)

        return transformers.BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names
        return list(set(tokenizer_input_names + audio_processor_input_names))
