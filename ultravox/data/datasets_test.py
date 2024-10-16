import itertools
from typing import Optional, Union

import datasets as hf_datasets
import numpy as np
import pytest
import torch
from torch.utils import data
from transformers.feature_extraction_utils import BatchFeature

from ultravox.data import datasets


class FakeSizedIterableDataset(datasets.SizedIterableDataset):
    """Fake version of datasets.SizedIterableDataset"""

    def __init__(self, n, start=0, length=0):
        self.data = range(start, start + n)
        self._length = length

    def __iter__(self):
        for sample in self.data:
            yield sample

    def __len__(self):
        return self._length


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
        self._split = "fake"

    def __iter__(self):
        return (i for i in self.data)


class FakeTranscribeDataset(datasets.VoiceDataset):
    """Fake version of our VoiceDataset."""

    def __init__(self, n: int, args: Optional[datasets.VoiceDatasetArgs] = None):
        super().__init__(
            args or datasets.VoiceDatasetArgs(),
        )
        self._init_dataset(FakeHuggingFaceIterableDataset(n), n)

    def _get_sample(self, row: BatchFeature) -> Optional[datasets.VoiceSample]:
        messages = self._make_messages("<|audio|>", row["text"])
        return self._make_sample(messages, np.zeros(256), row["text"])


class FakeGenericDataset(datasets.GenericDataset):
    """Fake version of GenericDataset, hooked to return a FakeHuggingFaceIterableDataset."""

    def __init__(
        self,
        n: int,
        config: datasets.DatasetConfig,
        args: Optional[datasets.VoiceDatasetArgs] = None,
    ):
        self._n = n
        super().__init__(args or datasets.VoiceDatasetArgs(), config)

    def _load_hf_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        streaming: bool = True,
        audio_field: Optional[str] = None,
    ) -> data.Dataset:
        return FakeHuggingFaceIterableDataset(self._n)


class FakeDataproc(datasets.Dataproc):
    def __init__(self, dataset):
        super().__init__(dataset)

    def _process(self, sample):
        return -sample


def test_dataproc():
    ds = FakeSizedIterableDataset(5)
    s = FakeDataproc(ds)
    assert list(s) == [0, -1, -2, -3, -4]


def test_interleaved_first_exhausted():
    ds1 = FakeSizedIterableDataset(5)
    s = datasets.InterleaveDataset([ds1])
    assert list(s) == [0, 1, 2, 3, 4]
    ds2 = FakeSizedIterableDataset(9)
    ds3 = FakeSizedIterableDataset(3)
    s = datasets.InterleaveDataset(
        [ds1, ds2, ds3],
        stop_strategy=datasets.StopStrategy.FIRST_EXHAUSTED,
        static=True,
    )
    # static=True disables random sampling of datasets, so the order is deterministic
    # stop_strategy=first_exhausted will stop interleave when the first dataset is exhausted
    assert list(s) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    s = datasets.InterleaveDataset([])
    assert list(s) == []


def test_interleaved_last_exhausted():
    ds1 = FakeSizedIterableDataset(4)
    ds2 = FakeSizedIterableDataset(2, start=10)
    s = datasets.InterleaveDataset(
        [ds1, ds2],
        stop_strategy=datasets.StopStrategy.LAST_EXHAUSTED,
        static=True,
    )
    # static=True disables random sampling of datasets, so the order is deterministic
    # stop_strategy=last_exhausted will stop interleave when the last dataset is exhausted
    assert list(s) == [0, 10, 1, 11, 2, 10, 3, 11]


def test_interleaved_never_stop():
    ds1 = FakeSizedIterableDataset(4)
    ds2 = FakeSizedIterableDataset(2, start=10)
    s = datasets.InterleaveDataset(
        [ds1, ds2],
        stop_strategy=datasets.StopStrategy.NEVER_STOP,
        static=True,
    )
    # static=True disables random sampling of datasets, so the order is deterministic
    # stop_strategy=never_stop will continue interleaving forever
    assert list(itertools.islice(s, 12)) == [0, 10, 1, 11, 2, 10, 3, 11, 0, 10, 1, 11]


def test_interleaved_random():
    ds1 = FakeSizedIterableDataset(4)
    ds2 = FakeSizedIterableDataset(2, start=10)
    s = datasets.InterleaveDataset(
        [ds1, ds2],
        [10.0, 1.0],
    )
    # stop_strategy=last_exhausted will stop interleaving when the last dataset is exhausted (attempted after exhaustion)
    assert list(s) == [
        0,
        1,
        2,
        3,
        0,
        10,
        1,
        2,
        3,
        0,
        1,
        11,
        2,
        3,
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]


def test_interleaved_with_multiprocessing():
    ds = FakeSizedIterableDataset(5)
    s = datasets.InterleaveDataset([ds])

    dl = data.DataLoader(s, num_workers=1, batch_size=5)

    batch = next(iter(dl))
    assert torch.allclose(batch, torch.tensor([0, 1, 2, 3, 4]))


def test_range():
    ds = FakeSizedIterableDataset(10, length=10)
    s = datasets.Range(ds, 5)
    assert len(s) == 5
    assert list(s) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="exceeds dataset length"):
        s = datasets.Range(ds, 100)
    s = datasets.Range(ds, 10)
    assert list(s) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    s = datasets.Range(ds)
    assert len(s) == 10
    assert list(s) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_transcribe_dataset():
    ds = FakeTranscribeDataset(5)
    assert len(ds) == 5
    sample = next(iter(ds))
    assert isinstance(sample, datasets.VoiceSample)
    assert sample.messages == [
        {"role": "user", "content": "<|audio|>"},
        {"role": "assistant", "content": "0"},
    ]
    assert np.array_equal(sample.audio, np.zeros(256))
    assert sample.sample_rate == 16000
    assert sample.audio_transcript == "0"


def test_generic_dataset():
    mock_config = datasets.DatasetConfig(
        path="mock_path",
        splits=[datasets.DatasetSplitConfig(name="fake", num_samples=5)],
    )
    ds = FakeGenericDataset(5, mock_config)
    assert len(ds) == 5
    sample = next(iter(ds))
    assert isinstance(sample, datasets.VoiceSample)
    assert sample.messages == [
        {"role": "user", "content": "<|audio|>"},
        {"role": "assistant", "content": "0"},
    ]
    assert np.array_equal(sample.audio, np.zeros(256))
    assert sample.sample_rate == 16000
    assert sample.audio_transcript == "0"


def test_generic_dataset_custom_templates():
    mock_config = datasets.DatasetConfig(
        path="mock_path",
        splits=[datasets.DatasetSplitConfig(name="fake", num_samples=5)],
        user_template="Listen to the following and respond with 'xyzzy':\n<|audio|>",
        assistant_template="xyzzy",
        transcript_template="{{text}}",
    )
    ds = FakeGenericDataset(5, mock_config)
    assert len(ds) == 5
    sample = next(iter(ds))
    assert isinstance(sample, datasets.VoiceSample)
    assert sample.messages == [
        {
            "role": "user",
            "content": "Listen to the following and respond with 'xyzzy':\n<|audio|>",
        },
        {"role": "assistant", "content": "xyzzy"},
    ]
    assert np.array_equal(sample.audio, np.zeros(256))
    assert sample.sample_rate == 16000
    assert sample.audio_transcript == "0"


def test_generic_dataset_length_mismatch():
    mock_config = datasets.DatasetConfig(
        path="mock_path",
        splits=[datasets.DatasetSplitConfig(name="fake", num_samples=5)],
    )
    ds = FakeGenericDataset(10, mock_config)
    assert len(ds) == 5

    pattern = r"(has been exceeded|Mismatch between presumed length)"
    with pytest.warns(UserWarning, match=pattern):
        list(ds)

    mock_config = datasets.DatasetConfig(
        path="mock_path",
        splits=[datasets.DatasetSplitConfig(name="fake", num_samples=10)],
    )
    ds = FakeGenericDataset(5, mock_config)
    assert len(ds) == 10

    with pytest.warns(UserWarning, match="Mismatch between presumed length"):
        list(ds)


def test_generic_dataset_multiple_splits():
    mock_config = datasets.DatasetConfig(
        path="mock_path",
        splits=[
            datasets.DatasetSplitConfig(name="train", num_samples=90),
            datasets.DatasetSplitConfig(name="validation", num_samples=10),
        ],
    )
    ds = FakeGenericDataset(100, mock_config)
    assert len(ds) == 90
    ds = FakeGenericDataset(
        100,
        mock_config,
        datasets.VoiceDatasetArgs(split=datasets.DatasetSplit.VALIDATION),
    )
    assert len(ds) == 10


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
        "Transcribe\n<|audio|>", array, 16000
    )
    assert sample.sample_rate == 16000
    assert sample.audio is not None, "sample.audio should not be None"
    assert len(sample.audio) == 16000
    assert sample.audio.dtype == np.float32
    assert sample.messages == [
        {"role": "user", "content": "Transcribe\n<|audio|>"},
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
        _ = datasets.VoiceSample.from_prompt_and_raw(
            "Transcribe\n<|audio|>", array, 16000
        )


def test_get_messages():
    messages = datasets._get_messages("Yo!", "Hi!")
    assert messages == [
        {"role": "user", "content": "Yo!"},
        {"role": "assistant", "content": "Hi!"},
    ]

    messages = datasets._get_messages(
        "Yo!", "Hi!", assistant_last=False, sys_prompt="Be nice!"
    )
    assert messages == [
        {"role": "system", "content": "Be nice!"},
        {"role": "assistant", "content": "Yo!"},
        {"role": "user", "content": "Hi!"},
    ]

    messages = datasets._get_messages("A", "B", "C")
    assert messages == [
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "C"},
    ]
