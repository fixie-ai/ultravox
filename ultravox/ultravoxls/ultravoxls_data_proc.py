from typing import Any, Dict

import datasets
import librosa
from torch.utils import data

from ultravox.data import datasets
from ultravox.ultravoxls import ultravoxls_processing


class UltravoxLSDataproc(datasets.Dataproc):
    def __init__(
        self,
        dataset: data.IterableDataset,
        processor: ultravoxls_processing.UltravoxLSProcessor,
        expected_audio_length_seconds: int,
    ) -> None:
        """
        Pre-processing for the UltravoxLS model: applies tokenization the UltravoxLSProcessor
        and prepares the shape of the data for being fed into the model.

        Args:
            dataset: The dataset to wrap/preprocess.
            processor: The processor.
        """
        super().__init__(dataset)
        self._dataset = dataset
        self.processor = processor
        self.expected_audio_length_seconds = expected_audio_length_seconds

    def __iter__(self):
        for sample in self._dataset:
            seconds_in_sample = librosa.get_duration(
                y=sample.audio, sr=sample.sample_rate
            )

            if seconds_in_sample < self.expected_audio_length_seconds:
                continue  # Skip samples that are too short

            else:
                # Calculate the number of samples for each chunk
                chunk_size_samples = int(
                    self.expected_audio_length_seconds * sample.sample_rate
                )

                # Split the entire audio into chunks of the specified length
                for chunk_start in range(0, len(sample.audio), chunk_size_samples):
                    chunk_end = chunk_start + chunk_size_samples
                    chunk = sample.audio[chunk_start:chunk_end]

                    # If the chunk is shorter than the specified length (last chunk), throw it away
                    if len(chunk) < chunk_size_samples:
                        continue  # Skip the last chunk

                    yield self._process(
                        datasets.VoiceSample(
                            messages=sample.messages,
                            audio=chunk,
                            sample_rate=sample.sample_rate,
                        )
                    )

    def _process(self, sample: datasets.VoiceSample) -> Dict[str, Any]:
        # Process audio using UltravoxLSProcessor.
        # Audio is expanded to be a [C x M] array, although C=1 for mono audio.

        inputs = self.processor.dataproc(sample)

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
