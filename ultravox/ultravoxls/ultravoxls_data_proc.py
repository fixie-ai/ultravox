import math
from typing import Any, Dict

import librosa
import numpy as np

from ultravox import data as datasets
from ultravox.ultravoxls import ultravoxls_processing


class UltravoxLSDataproc(datasets.Dataproc):
    def __init__(
        self,
        dataset: datasets.SizedIterableDataset,
        processor: ultravoxls_processing.UltravoxLSProcessor,
        expected_audio_length_seconds: float,
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
        self.expected_audio_length_seconds = expected_audio_length_seconds
        self.min_audio_length_seconds = expected_audio_length_seconds / 2

    def __iter__(self):
        for sample in self._dataset:
            seconds_in_sample = librosa.get_duration(
                y=sample.audio, sr=sample.sample_rate
            )

            if seconds_in_sample < self.min_audio_length_seconds:
                continue  # Skip samples that are too short

            else:
                # Calculate the number of samples for each chunk
                chunk_size_samples = int(
                    self.expected_audio_length_seconds * sample.sample_rate
                )

                audio_len = len(sample.audio)
                num_chunks = int(math.ceil(audio_len / chunk_size_samples))

                # spread out the chunks evenly across the audio
                starts = np.linspace(0, audio_len - chunk_size_samples, num_chunks)

                # Split the entire audio into chunks of the specified length
                for chunk_start in starts.round().astype(int):
                    chunk = sample.audio[chunk_start : chunk_start + chunk_size_samples]

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

        return inputs
