from typing import Any, Dict

import datasets
import numpy as np
from torch.utils import data

from ultravox.data import datasets
from ultravox.ultravoxls import ultravoxls_processing


class UltravoxLSDataproc(datasets.Dataproc):
    def __init__(
        self,
        dataset: data.IterableDataset,
        processor: ultravoxls_processing.UltravoxLSProcessor,
    ) -> None:
        """
        Pre-processing for the UltravoxLS model: applies tokenization the UltravoxLSProcessor
        and prepares the shape of the data for being fed into the model.

        Args:
            dataset: The dataset to wrap/preprocess.
            processor: The processor.
        """
        super().__init__(dataset)
        self.processor = processor

    def _process(self, sample: datasets.VoiceSample) -> Dict[str, Any]:
        # Process audio using UltravoxLSProcessor.
        # Audio is expanded to be a [C x M] array, although C=1 for mono audio.
        audio = (
            np.expand_dims(sample.audio, axis=0) if sample.audio is not None else None
        )
        inputs = self.processor(
            audio=audio,
            return_tensors="pt",
            sampling_rate=sample.sample_rate,
        )

        # Extract input_ids, attention_mask, and audio_values from the processed inputs
        input_ids = inputs["input_ids"].squeeze_(0)
        inputs["attention_mask"].squeeze_(0)

        # No need to shift the labels as the model does it internally
        labels = input_ids.clone()

        return {
            # input_ids, attention_mask, audio_values, audio_token_start_idx, audio_token_len
            # if include_alt_fields is True, also include alt_input_ids, alt_attention_mask, alt_labels
            **inputs,
            "labels": labels,
        }
