import dataclasses
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from .ultravox_config import UltravoxConfig


@dataclasses.dataclass
class DataCollatorForSeq2SeqWithAudio(transformers.DataCollatorForSeq2Seq):
    # when enabled, the alt_input_ids, alt_attention_mask, and alt_labels fields are used for computing the KL loss in UltravoxModel
    include_alt_fields: bool = False

    def __call__(self, features, *args, **kwargs):
        audio_values = [x for f in features for x in f.pop("audio_values", [])]
        audio_lens = [x for f in features for x in f.pop("audio_lens", [])]
        audio_token_len = [x for f in features for x in f.pop("audio_token_len", [])]
        audio_token_start_idx = [
            x for f in features for x in f.pop("audio_token_start_idx", [])
        ]

        if self.include_alt_fields:
            # these fields are hard-coded in the transformer data collator, so they need special handling before calling the super method
            alt_features = [
                {
                    "input_ids": f.pop("alt_input_ids"),
                    "attention_mask": f.pop("alt_attention_mask"),
                    "labels": f.pop("alt_labels"),
                }
                for f in features
            ]

        batch = super().__call__(features, *args, **kwargs)
        if self.include_alt_fields:
            alt_batch = super().__call__(alt_features, *args, **kwargs)
            batch["alt_input_ids"] = alt_batch["input_ids"]
            batch["alt_attention_mask"] = alt_batch["attention_mask"]
            batch["alt_labels"] = alt_batch["labels"]

        batch["audio_token_start_idx"] = torch.stack(audio_token_start_idx)
        batch["audio_lens"] = torch.stack(audio_lens)
        batch["audio_token_len"] = torch.stack(audio_token_len)

        # Pad the last dimension of all audio_values to the same length, with 0s on the right.
        if audio_values:
            max_len = max([x.shape[-1] for x in audio_values])
            batch["audio_values"] = torch.stack(
                [F.pad(x, (0, max_len - x.shape[-1])) for x in audio_values]
            )
            if self.tokenizer.padding_side == "left":
                input_ids_lens = torch.LongTensor(
                    [f["input_ids"].shape[-1] for f in features]
                )
                displacement = batch["input_ids"].shape[-1] - input_ids_lens
                displacement = displacement.repeat_interleave(
                    batch["audio_batch_size"].squeeze(-1)
                )
                batch["audio_token_start_idx"] += displacement.to(
                    batch["audio_token_start_idx"].device
                )
        return batch


class UltravoxProcessor(transformers.ProcessorMixin):
    """
    Constructs an Ultravox processor which wraps an audio processor and a tokenizer into a single processor.

    Args:
        audio_processor: The audio processor for the audio encoder.
        tokenizer: The tokenizer for the language model.
    """

    attributes = ["audio_processor", "tokenizer"]
    audio_processor_class = ("WhisperProcessor",)
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
        encoder_ds_factor: int = 2,
        stack_factor: int = 8,
        audio_placeholder: str = "<|audio|>",
        # Defaults to whisper encoder context size
        audio_context_size: Optional[int] = 3000,
    ):
        """
        Args:
            audio_processor: The audio processor for the audio encoder.
            tokenizer: The tokenizer for the language model.
            audio_padding: The padding strategy for the audio encoder.
            stack_factor: The factor by which the audio encoder output is stacked in the multimodal projector.
            encoder_ds_factor: The downsampling factor of the audio encoder.
            audio_placeholder: The placeholder for the audio in the text.
            audio_context_size: The maximum number of frames that the audio encoder can handle.
        """
        self.audio_padding = audio_padding
        self.encoder_ds_factor = encoder_ds_factor
        self.stack_factor = stack_factor
        self.audio_placeholder = audio_placeholder
        self.audio_token_replacement = tokenizer.eos_token
        self.audio_context_size = audio_context_size
        assert (
            self.audio_token_replacement is not None
        ), "The tokenizer has no EOS token. Cannot recover."
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(audio_processor=audio_processor, tokenizer=tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        config: UltravoxConfig = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        audio_processor = transformers.AutoProcessor.from_pretrained(
            config.audio_model_id
            or config.audio_config._name_or_path
            or "openai/whisper-tiny"
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

    def _chunk_and_pad_audio(
        self, audio_values: torch.Tensor, audio_lens: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Processes the audio batch by chunking any items in the batch according to the audio_context_size,
        padding the last chunk if needed, and returns a dictionary with updated audio data.

        Args:
            audio_values (torch.Tensor): A tensor of audio values (e.g., in B, D, T format).
            audio_lens (torch.Tensor): A tensor of audio lengths.

        Returns:
            Dict[str, Any]: Dictionary with the following keys:
                - "audio_values": The concatenated audio tensor after chunking and padding.
                - "audio_lens": Tensor of lengths for each chunk.
                - "audio_is_continuation": Tensor of booleans indicating if the chunk is a continuation of the previous chunk.
                - "audio_batch_size": A Tensor with one integer representing the number of chunks.

        """
        chunked_audio_values: List[torch.Tensor] = []
        chunked_audio_lens: List[int] = []
        is_continuation_list: List[bool] = []
        context_size = self.audio_context_size or audio_values.shape[-1]

        for audio, audio_len in zip(audio_values, audio_lens):
            for offset in range(0, audio_len, context_size):
                is_continuation = offset > 0
                chunk = audio[..., offset : offset + context_size]
                if is_continuation and chunk.shape[-1] < context_size:
                    # N.B. We only need to pad continuation chunks. If none of the samples require chunking, the
                    # batch might not (need to) be padded all the way to the audio_context_size, in which case
                    # we've already included the padding above. On the other hand, if we have any continuation
                    # chunks we know that the batch needs to be padded to audio_context_size because that's what
                    # we're slicing to.
                    chunk = F.pad(chunk, (0, context_size - chunk.shape[-1]))
                chunked_audio_values.append(torch.as_tensor(chunk))
                chunked_audio_lens.append(min(audio_len - offset, context_size))
                is_continuation_list.append(is_continuation)

        return {
            "audio_values": torch.stack(chunked_audio_values),
            "audio_lens": torch.tensor(chunked_audio_lens),
            "audio_is_continuation": torch.tensor(is_continuation_list),
            "audio_batch_size": torch.tensor([len(chunked_audio_values)]),
        }

    def __call__(
        self,
        text: Optional[str] = None,
        audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        audios: Optional[
            Union[
                List[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]
            ]
        ] = None,
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
        audio processor's [`~WhisperProcessor.__call__`] if `audio` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`):
                The sequence to be encoded. Sequence can be a string or (pretokenized string).
            audio (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The audio to be prepared. Audio can be a single-channel (1-dimensional) NumPy array or PyTorch tensor.
            audios (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                A list or two dimensional array of audio to be prepared.
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
        # TODO: Add support for multiple text inputs.
        if audio is not None and audios is not None:
            raise ValueError("Only one of `audio` or `audios` should be provided.")
        elif audio is not None:
            audios = audio if isinstance(audio, list) or audio.ndim == 2 else [audio]
        elif audios is None:
            audios = []

        data = {}
        audio_is_continuation = []
        if len(audios) > 0:
            audios = [x.numpy() if isinstance(x, torch.Tensor) else x for x in audios]

            # Pad out each audio to at least 2 hops (the minimum required by the processor).
            hop_length = self.audio_processor.feature_extractor.hop_length
            audios = [
                (
                    np.pad(x, (0, 2 * hop_length - len(x)), mode="constant")
                    if len(x) < 2 * hop_length
                    else x
                )
                for x in audios
            ]

            # Main audio processing. The processor is model-specific.
            x: transformers.BatchFeature = self.audio_processor(
                audios,
                sampling_rate=sampling_rate,
                padding="longest",
                pad_to_multiple_of=hop_length,  # The attention mask effectively gets padded to the hop length, so pad the audio to be consistent.
                truncation=False,
                return_attention_mask=True,
                **kwargs,
            )

            data.update(
                self._chunk_and_pad_audio(
                    audio_values=torch.as_tensor(
                        x.input_features if "input_features" in x else x.input_values
                    ),
                    audio_lens=torch.as_tensor(x.attention_mask).sum(-1),
                )
            )

            audio_is_continuation = data.pop("audio_is_continuation")
            data["audio_token_len"] = torch.ceil(
                data["audio_lens"] / (self.encoder_ds_factor * self.stack_factor)
            ).to(dtype=torch.int)

        if text is not None:
            if not isinstance(text, str):
                raise ValueError("Text must be a string. Batch mode not supported yet.")

            # Special tokens like BOS should already have been added by the caller.
            tokenized_parts = self.tokenizer(
                text.split(
                    "<|audio|>"  # The placeholder isn't part of the vocabulary, so split the text around it.
                ),
                add_special_tokens=False,
                **kwargs,
            )

            audio_token_start_idx = []
            replacement_token_id = self.tokenizer.get_vocab()[
                self.audio_token_replacement
            ]
            placeholder_index = -1
            split_input_ids = tokenized_parts["input_ids"]
            input_ids: List[int] = []

            for i, token_len in enumerate(data.get("audio_token_len", [])):
                if not audio_is_continuation[i]:
                    placeholder_index += 1
                    if placeholder_index >= len(split_input_ids):
                        raise ValueError(
                            f"Text contains too few audio placeholders. (Expected {len(audios)} placeholders)"
                        )

                    input_ids.extend(split_input_ids[placeholder_index])

                audio_token_start_idx.append(len(input_ids))

                input_ids.extend([replacement_token_id] * token_len)

            # Include any tokens after the last audio.
            placeholder_index += 1
            if placeholder_index != len(split_input_ids) - 1:
                raise ValueError(
                    f"Text contains too many audio placeholders. (Expected {len(audios)} placeholders)"
                )
            input_ids.extend(split_input_ids[placeholder_index])

            if "audio_token_len" in data:
                data["audio_token_start_idx"] = torch.as_tensor(audio_token_start_idx)

            data["input_ids"] = [input_ids]
            data["attention_mask"] = [[1] * len(input_ids)]

            # Ensure that there are no audio placeholders after the last audio.

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
