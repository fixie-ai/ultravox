from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import transformers
from torch.utils import data

from ultravox.data import datasets


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
        encoder_ds_factor: int = 320,
        stack_factor: int = 8,
        audio_placeholder: str = "<|audio|>",
    ):
        """
        Args:
            audio_processor: The audio processor for the audio encoder.
            tokenizer: The tokenizer for the language model.
            encoder_ds_factor: The downsample factor of the audio encoder.
            stack_factor: The factor by which the audio encoder output is stacked in the multimodal projector.
            audio_placeholder: The placeholder for the audio in the text.
        """
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
        data = {}
        audio_embed_frames = 0
        if audio is not None and len(audio) > 0:
            audio_len = audio.shape[-1]
            # It's guaranteed that the number of frames is less than or equal to this amount.
            # For Whisper this is exact AFAICT, but for Wav2Vec2 it's an upper bound.
            # Currently, StackAudioFrames makes sure an over-estimation won't cause issues by padding the audio embeddings.
            nb_encoder_frames = int(round(audio_len / self.encoder_ds_factor + 1e-4))
            audio_embed_frames = int(np.ceil(nb_encoder_frames / self.stack_factor))
            data["audio_token_len"] = [audio_embed_frames]

            x = self.audio_processor(
                audio, sampling_rate=sampling_rate, padding="longest", **kwargs
            )
            if "input_features" in x:
                data["audio_values"] = x.input_features
            else:
                data["audio_values"] = x.input_values

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


class UltravoxDataproc(datasets.Dataproc):
    def __init__(
        self,
        dataset: data.IterableDataset,
        processor: UltravoxProcessor,
        train_on_inputs: bool = False,
        inference_mode: bool = False,
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
        """
        super().__init__(dataset)
        self.processor = processor
        self.train_on_inputs = train_on_inputs
        self.inference_mode = inference_mode
        if self.inference_mode:
            self.train_on_inputs = True

    def _process(self, sample: datasets.VoiceSample) -> Dict[str, Any]:
        if self.inference_mode:
            # remove the assistant message from the sample so that the model can generate it
            sample.messages = sample.messages[:-1]

        text = self.processor.tokenizer.apply_chat_template(
            sample.messages, tokenize=False
        )

        # Process audio and text using GazelleProcessor.
        # Audio is expanded to be a [C x M] array, although C=1 for mono audio.
        audio = (
            np.expand_dims(sample.audio, axis=0) if sample.audio is not None else None
        )
        inputs = self.processor(
            text=text,
            audio=audio,
            return_tensors="pt",
            sampling_rate=sample.sample_rate,
        )

        # Extract input_ids, attention_mask, and audio_values from the processed inputs
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        audio_values = inputs["audio_values"].squeeze(0)
        audio_token_start_idx = inputs["audio_token_start_idx"].squeeze(0)
        audio_token_len = inputs["audio_token_len"].squeeze(0)

        # No need to shift the labels as the model does it internally
        labels = input_ids.clone()

        if not self.train_on_inputs:
            # Mask the prompt tokens and only compute loss on the assistant message, not the prompt.
            # The idea is that the model should only be able to predict the assistant message given the user message.
            # One reason is that there's very little randomness in the prompt, so the model would be forced to memorize it.
            #
            # Example (-100 is the ignore index):
            #   Tokens: <user> Transcribe <|audio|> </s> <assistant> Brown fox jumps over the lazy dog </s>
            #   Labels:  -100    -100       -100    -100 <assistant> Brown fox jumps over the lazy dog </s>
            #
            # Note: The above might look weird because I'm mixing token IDs and text, but that's just for illustration.
            input_text = self.processor.tokenizer.apply_chat_template(
                sample.messages[:-1], tokenize=False
            )

            # TODO: this might be slow due to calling audio_processor twice. We can compute modified input_text_len directly too.
            # Revisit when using WhisperProcessor.
            input_text_len = self.processor(
                text=input_text,
                audio=audio,
                sampling_rate=sample.sample_rate,
            )["input_ids"].shape[-1]
            labels[:input_text_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_values": audio_values,
            "labels": labels,
            "audio_token_start_idx": audio_token_start_idx,
            "audio_token_len": audio_token_len,
        }
