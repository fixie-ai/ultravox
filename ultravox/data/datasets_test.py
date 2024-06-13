import itertools
from typing import Optional, Union

import datasets as hf_datasets
import numpy as np
import pytest
import torch
from torch.utils import data
from transformers.feature_extraction_utils import BatchFeature

from ultravox.data import datasets


class FakeIterableDataset(data.IterableDataset):
    """Fake version of a PyTorch IterableDataset."""

    def __init__(self, n, start=0):
        self.data = range(start, start + n)

    def __iter__(self):
        return (i for i in self.data)


class FakeHuggingFaceIterableDataset(hf_datasets.IterableDataset):
    """Fake version of an ASR Hugging Face IterableDataset."""

    def __init__(self, n):
        self.data = [
            {
                "text": str(i),
                "audio": {"array": np.full(256, float(i)), "sampling_rate": 16000},
            }
            for i in range(n)
        ]

    def __iter__(self):
        return (i for i in self.data)


class FakeTranscribeDataset(datasets.VoiceDataset):
    """Fake version of our VoiceDataset using a transcribe prompt."""

    def __init__(self, n: int, args: Optional[datasets.VoiceDatasetArgs] = None):
        super().__init__(args or datasets.VoiceDatasetArgs())
        self._init_dataset(FakeHuggingFaceIterableDataset(n))

    def _get_sample(self, idx: int, row: BatchFeature) -> datasets.VoiceSample:
        return self._get_transcribe_sample(idx, row)


class FakeDataproc(datasets.Dataproc):
    def __init__(self, dataset):
        super().__init__(dataset)

    def _process(self, sample):
        return -sample


def test_dataproc():
    ds = FakeIterableDataset(5)
    s = FakeDataproc(ds)
    assert list(s) == [0, -1, -2, -3, -4]


def test_interleaved():
    # We put the smallest iterator last to test for that edge case.
    ds1 = FakeIterableDataset(5)
    s = datasets.InterleaveDataset([ds1])
    assert list(s) == [0, 1, 2, 3, 4]
    ds2 = FakeIterableDataset(9)
    ds3 = FakeIterableDataset(3)
    s = datasets.InterleaveDataset([ds1, ds2, ds3])
    assert list(s) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8]
    s = datasets.InterleaveDataset([])
    assert list(s) == []


def test_interleaved_repeat():
    ds1 = FakeIterableDataset(4)
    ds2 = FakeIterableDataset(2, start=10)
    s = datasets.InterleaveDataset([ds1, ds2], repeat=True)
    # repeat=True makes the dataset infinite, so we cannot safely use list()
    assert list(itertools.islice(s, 9)) == [0, 10, 1, 11, 2, 10, 3, 11, 0]


def test_interleaved_with_multiprocessing():
    ds = FakeIterableDataset(5)
    s = datasets.InterleaveDataset([ds])

    dl = data.DataLoader(s, num_workers=1, batch_size=5)

    batch = next(iter(dl))
    assert torch.allclose(batch, torch.tensor([0, 1, 2, 3, 4]))


def test_range():
    ds = FakeIterableDataset(10)
    s = datasets.Range(ds, 5)
    assert list(s) == [0, 1, 2, 3, 4]
    s = datasets.Range(ds, 100)
    assert list(s) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    s = datasets.Range(ds)
    assert list(s) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_transcribe_dataset():
    ds = FakeTranscribeDataset(5)
    sample = next(iter(ds))
    assert isinstance(sample, datasets.VoiceSample)
    assert sample.messages == [
        {"role": "user", "content": "Transcribe <|audio|>"},
        {"role": "assistant", "content": "0"},
    ]
    assert np.array_equal(sample.audio, np.zeros(256))
    assert sample.sample_rate == 16000
    assert sample.audio_transcript == "0"


def test_num_prompts():
    ds = FakeTranscribeDataset(5, datasets.VoiceDatasetArgs(num_prompts=3))
    samples = list(ds)
    assert samples[0].messages[0]["content"] == "Transcribe <|audio|>"
    assert (
        samples[1].messages[0]["content"]
        == "Transcribe exactly what is said here <|audio|>"
    )
    assert (
        samples[2].messages[0]["content"]
        == "Repeat exactly what is written here: <|audio|>"
    )
    assert samples[3].messages[0]["content"] == "Transcribe <|audio|>"


def _create_sine_wave(
    freq: int = 440,
    duration: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.1,
    target_dtype: str = "float32",
) -> Union[
    np.typing.NDArray[np.float32],
    np.typing.NDArray[np.float64],
    np.typing.NDArray[np.int16],
    np.typing.NDArray[np.int32],
]:
    t = np.arange(sample_rate * duration, dtype=np.float32) / sample_rate
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    match target_dtype:
        case "int16":
            wave = np.int16(wave * 32767)
        case "int32":
            wave = np.int32(wave * 2147483647)
        case "float32":
            # Already float32, nothing needed.
            pass
        case "float64":
            wave = wave.astype(np.float64)
        case _:
            raise ValueError(f"Unsupported dtype: {target_dtype}")
    return wave


def _create_and_validate_sample(target_dtype: str = "float32"):
    # Create a sine wave at 440 Hz with a duration of 1.0 second, sampled at 16
    # kHz, with an amplitude of 0.1, and the specified dtype.
    array = _create_sine_wave(target_dtype=target_dtype)
    sample = datasets.VoiceSample.from_prompt_and_raw(
        "Transcribe <|audio|>", array, 16000
    )
    assert sample.sample_rate == 16000
    assert sample.audio is not None, "sample.audio should not be None"
    assert len(sample.audio) == 16000
    assert sample.audio.dtype == np.float32
    assert sample.messages == [
        {"role": "user", "content": "Transcribe <|audio|>"},
    ]
    # Serialize and deserialize the sample.
    json = sample.to_json()
    sample2 = datasets.VoiceSample.from_json(json)
    assert sample2.sample_rate == sample.sample_rate
    assert sample2.audio is not None, "sample2.audio should not be None"
    assert len(sample2.audio) == len(sample.audio)
    assert sample2.audio.dtype == sample.audio.dtype
    assert sample2.messages == sample.messages
    assert np.allclose(sample2.audio, sample.audio, rtol=0.0001, atol=0.0001)


def test_create_sample__int16():
    _create_and_validate_sample("int16")


def test_create_sample__int32():
    _create_and_validate_sample("int32")


def test_create_sample__float32():
    _create_and_validate_sample("float32")


def test_create_sample__float64():
    _create_and_validate_sample("float64")


def test_create_sample__raises_on_unsupported_dtype():
    with pytest.raises(AssertionError):
        array = np.ndarray(shape=(16000,), dtype=np.uint8)
        sample = datasets.VoiceSample.from_prompt_and_raw(
            "Transcribe <|audio|>", array, 16000
        )
