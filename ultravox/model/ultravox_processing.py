import dataclasses
from typing import Any, Dict, Optional, Union

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
        audio_values = [f.pop("audio_values", None) for f in features]
        audio_lens = [f.pop("audio_lens", None) for f in features]
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

        # Pad the last dimension of all audio_values to the same length, with 0s on the right.
        if audio_values and audio_values[0] is not None:
            max_len = max([x.shape[-1] for x in audio_values])
            batch["audio_values"] = torch.cat(
                [F.pad(x, (0, max_len - x.shape[-1])) for x in audio_values]
            )
            if self.tokenizer.padding_side == "left":
                input_ids_lens = torch.LongTensor(
                    [f["input_ids"].shape[-1] for f in features]
                )
                displacement = batch["input_ids"].shape[-1] - input_ids_lens
                batch["audio_token_start_idx"] += displacement.to(
                    batch["audio_token_start_idx"].device
                )
        # batch["audio_lens"].shape = (B,)
        batch["audio_lens"] = torch.cat(audio_lens)
        return batch


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
        # Defaults to whisper encoder context size
        audio_context_size: Optional[int] = 3000,
    ):
        """
        Args:
            audio_processor: The audio processor for the audio encoder.
            tokenizer: The tokenizer for the language model.
            audio_padding: The padding strategy for the audio encoder.
            encoder_ds_factor: The downsample factor of the audio encoder.
            stack_factor: The factor by which the audio encoder output is stacked in the multimodal projector.
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

    def _chunk_and_pad_audio(self, audio_values: torch.Tensor) -> Dict[str, Any]:
        """
        Processes the audio tensor by chunking it according to the audio_context_size,
        padding the last chunk if needed, and returns a dictionary with updated audio data.

        Args:
            audio_values (torch.Tensor): A tensor of audio values (e.g., in B, D, T format).

        Returns:
            Dict[str, Any]: Dictionary with the following keys:
                - "audio_values": The concatenated audio tensor after chunking and padding.
                - "audio_lens": List of lengths (as torch.Tensor) for each chunk.
                - "audio_batch_size": A list with one integer representing the number of chunks.
        """
        result: Dict[str, Any] = {}
        if self.audio_context_size and audio_values.shape[-1] > self.audio_context_size:
            audio_chunks = list(
                torch.split(audio_values, self.audio_context_size, dim=-1)
            )
            valid_lengths = [chunk.shape[-1] for chunk in audio_chunks]
            result = {
                "audio_lens": [torch.as_tensor(length) for length in valid_lengths]
            }
            # Pad the last chunk to the full context length if needed.
            last_chunk = audio_chunks[-1]
            pad_size = self.audio_context_size - last_chunk.shape[-1]
            if pad_size > 0:
                audio_chunks[-1] = F.pad(last_chunk, (0, pad_size))
        else:
            audio_chunks = [audio_values]
            result = {"audio_lens": [torch.as_tensor(audio_values.shape[-1])]}
        result["audio_values"] = torch.cat(audio_chunks)
        result["audio_batch_size"] = [result["audio_values"].shape[0]]
        return result

    def __call__(
        self,
        text: Optional[str] = None,
        audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
        data: Dict[str, Any] = {}
        audio_embed_frames = 0
        if audio is not None and len(audio) > 0:
            audio_len = audio.shape[-1]
            # It's guaranteed that the number of frames is less than or equal to this amount.
            # For Whisper this is exact AFAICT, but for Wav2Vec2 it's an upper bound.
            # Currently, StackAudioFrames makes sure an over-estimation won't cause issues by padding the audio embeddings.
            nb_encoder_frames = int(round(audio_len / self.encoder_ds_factor + 1e-4))
            audio_embed_frames = int(np.ceil(nb_encoder_frames / self.stack_factor))
            data["audio_token_len"] = [audio_embed_frames]

            # Main audio processing. The processor is model-specific.
            x = self.audio_processor(
                audio,
                sampling_rate=sampling_rate,
                padding="longest",
<<<<<<< HEAD
                max_length=audio_len,
=======
                max_length=audio_len,  # The whisper audio_processor can handle audio lengths longer than 30 seconds
>>>>>>> upstream/main
                return_attention_mask=True,
                **kwargs,
            )

            if "input_features" in x:
                audio_values = x.input_features
            else:
                audio_values = x.input_values

            audio_values = torch.tensor(audio_values)
            chunk_and_pad_results = self._chunk_and_pad_audio(audio_values)
            data["audio_values"] = chunk_and_pad_results["audio_values"]
            data["audio_lens"] = chunk_and_pad_results["audio_lens"]
            data["audio_batch_size"] = chunk_and_pad_results["audio_batch_size"]

            # data["audio_len"] is the number of frames in the audio, used for creating attention masks in whisper encoder
            if (
                self.audio_padding == "max_length"
            ):  # audio is padded to max length, so we rely on the attention mask to determine audio_len
                data["audio_len"] = (
                    x.attention_mask.sum(-1) - 1
                )  # Whisper attention mask includes an extra 1 at the end that needs to be subtracted
            else:  # audio is not padded, so we can directly use the audio length
                data["audio_len"] = [torch.as_tensor(data["audio_values"]).shape[-1]]

        if text is not None:
            assert isinstance(
                text, str
            ), "Text must be a string. Batch mode not supported yet."
            if self.audio_placeholder in text:
                if "audio_token_len" not in data:
                    raise ValueError(
                        f"audio must be provided when using audio placeholder ({self.audio_placeholder}) in text."
                    )

                start_idx = len(
                    self.tokenizer.encode(
                        text[: text.index(self.audio_placeholder)],
                        add_special_tokens=False,
                    )
                )
                data["audio_token_start_idx"] = [start_idx]

                # Replace the audio placeholder with the audio token.
                #   e.g. "Transcribe\n<|audio|>" -> "Transcribe\n</s></s></s></s></s></s></s></s>"
                #        where the number of </s> is the number of audio frames.
                text = text.replace(
                    self.audio_placeholder,
                    self.audio_token_replacement * audio_embed_frames,
                )

            # Special tokens like BOS should already have been added by the caller.
            data.update(self.tokenizer([text], add_special_tokens=False, **kwargs))

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
