from typing import Optional

import datasets as hf_datasets
import numpy as np
import pytest
import torch
from torch.utils import data
from transformers.feature_extraction_utils import BatchFeature

from ultravox.data import data_sample
from ultravox.data import datasets
from ultravox.data import registry
from ultravox.data import types


class FakeSizedIterableDataset(datasets.SizedIterableDataset):
    """Fake version of datasets.SizedIterableDataset"""

    def __init__(self, n, start=0, length=0):
        self.data = range(start, start + n)
        self._length = length or n

    def __iter__(self):
        for sample in self.data:
            yield sample

    def __len__(self):
        return self._length

    def __str__(self):
        return "FakeSizedIterableDataset"

    @property
    def name(self):
        return "fake"


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

    def __init__(self, n: int, args: Optional[types.VoiceDatasetArgs] = None):
        super().__init__(args or types.VoiceDatasetArgs())
        self._init_dataset(FakeHuggingFaceIterableDataset(n), "fake", n)

    def _get_sample(self, row: BatchFeature) -> Optional[data_sample.VoiceSample]:
        messages = self._make_messages("<|audio|>", row["text"])
        return self._make_sample(messages, np.zeros(256), row["text"])

    def __str__(self):
        return "FakeTranscribeDataset"

    @property
    def name(self):
        return "fake_transcribe"


class FakeGenericDataset(datasets.GenericDataset):
    """Fake version of GenericDataset, hooked to return a FakeHuggingFaceIterableDataset."""

    def __init__(
        self,
        n: int,
        config: types.DatasetConfig,
        args: Optional[types.VoiceDatasetArgs] = None,
    ):
        self._n = n
        super().__init__(args or types.VoiceDatasetArgs(), config)

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


def test_interleaved_empty():
    s = datasets.InterleaveDataset([])
    assert list(s) == []


def test_interleaved_single_set():
    ds1 = FakeSizedIterableDataset(4)
    s = datasets.InterleaveDataset([ds1])
    assert list(s) == [0, 1, 2, 3]


def test_interleaved_normal_weights():
    ds1 = FakeSizedIterableDataset(4)
    ds2 = FakeSizedIterableDataset(8, start=10)
    ds3 = FakeSizedIterableDataset(2, start=100)
    s = datasets.InterleaveDataset([ds1, ds2, ds3])
    assert list(s) == [0, 10, 100, 11, 1, 12, 13, 2, 14, 101, 15, 3, 16, 17]


def test_interleaved_specific_weights():
    ds1 = FakeSizedIterableDataset(4)
    ds2 = FakeSizedIterableDataset(2, start=10)
    s = datasets.InterleaveDataset([ds1, ds2], [0.5, 2.0])
    assert list(s) == [0, 10, 11, 1, 10, 11]


def test_interleaved_zero_weights():
    ds1 = FakeSizedIterableDataset(4)
    ds2 = FakeSizedIterableDataset(2, start=10)
    s = datasets.InterleaveDataset([ds1, ds2], [0.0, 0.0])
    assert list(s) == []


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
    with pytest.warns(UserWarning, match="exceeds dataset length"):
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
    assert isinstance(sample, data_sample.VoiceSample)
    assert sample.messages == [
        {"role": "user", "content": "<|audio|>"},
        {"role": "assistant", "content": "0"},
    ]
    assert np.array_equal(sample.audio, np.zeros(256))
    assert sample.sample_rate == 16000
    assert sample.audio_transcript == "0"


def test_dataset_config():
    config = types.DatasetConfig(
        name="fake_dataset",
        path="mock_path",
        splits=[
            types.DatasetSplitConfig(
                name="clean", num_samples=5000, split=types.DatasetSplit.TRAIN
            ),
            types.DatasetSplitConfig(
                name="other", num_samples=10000, split=types.DatasetSplit.TRAIN
            ),
            types.DatasetSplitConfig(name="validation", num_samples=1000),
            types.DatasetSplitConfig(
                name="another_validation",
                num_samples=1000,
                split=types.DatasetSplit.VALIDATION,
            ),
        ],
    )
    assert config.name == "fake_dataset"
    assert config.path == "mock_path"
    assert len(config.splits) == 4
    assert config.splits[0].name == "clean"
    assert config.splits[0].num_samples == 5000
    assert config.splits[0].split == types.DatasetSplit.TRAIN
    assert config.splits[1].name == "other"
    assert config.splits[1].num_samples == 10000
    assert config.splits[1].split == types.DatasetSplit.TRAIN
    assert config.splits[2].name == "validation"
    assert config.splits[2].num_samples == 1000
    assert config.splits[2].split == types.DatasetSplit.VALIDATION
    assert config.splits[3].name == "another_validation"
    assert config.splits[3].num_samples == 1000
    assert config.splits[3].split == types.DatasetSplit.VALIDATION


def test_dataset_config_serialization():
    config = types.DatasetConfig(
        name="fake_dataset",
        path="fake_path",
        splits=[
            types.DatasetSplitConfig(
                name="clean", num_samples=5000, split=types.DatasetSplit.TRAIN
            ),
            types.DatasetSplitConfig(
                name="other", num_samples=10000, split=types.DatasetSplit.TRAIN
            ),
        ],
    )
    serialized = config.dumps_yaml()
    deserialized = types.DatasetConfig.loads_yaml(serialized)
    assert isinstance(deserialized, types.DatasetConfig)
    assert deserialized.name == "fake_dataset"
    assert deserialized.path == "fake_path"
    assert len(deserialized.splits) == 2
    assert deserialized.splits[0].name == "clean"
    assert deserialized.splits[0].num_samples == 5000
    assert deserialized.splits[1].name == "other"
    assert deserialized.splits[1].num_samples == 10000


def test_generic_dataset():
    config = types.DatasetConfig(
        name="fake_dataset",
        path="fake_path",
        splits=[
            types.DatasetSplitConfig(
                name="fake", num_samples=5, split=types.DatasetSplit.TRAIN
            )
        ],
    )
    ds = FakeGenericDataset(5, config)
    assert len(ds) == 5
    sample = next(iter(ds))
    assert isinstance(sample, data_sample.VoiceSample)
    assert sample.messages == [
        {"role": "user", "content": "<|audio|>"},
        {"role": "assistant", "content": "0"},
    ]
    assert np.array_equal(sample.audio, np.zeros(256))
    assert sample.sample_rate == 16000
    assert sample.audio_transcript == "0"


def test_generic_dataset_custom_templates():
    config = types.DatasetConfig(
        name="fake_dataset",
        path="fake_path",
        splits=[
            types.DatasetSplitConfig(
                name="fake", num_samples=5, split=types.DatasetSplit.TRAIN
            )
        ],
        user_template="Listen to the following and respond with 'xyzzy':\n<|audio|>",
        assistant_template="xyzzy",
        transcript_template="{{text}}",
    )
    ds = FakeGenericDataset(5, config)
    assert len(ds) == 5
    sample = next(iter(ds))
    assert isinstance(sample, data_sample.VoiceSample)
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


def test_generic_dataset_text_only():
    config = types.DatasetConfig(
        name="fake_dataset",
        path="fake_path",
        splits=[
            types.DatasetSplitConfig(
                name="fake", num_samples=5, split=types.DatasetSplit.TRAIN
            )
        ],
        user_template="Transcribe\n<|audio|>",
    )
    ds = FakeGenericDataset(5, config, types.VoiceDatasetArgs(include_audio=False))
    assert len(ds) == 5
    sample = next(iter(ds))
    assert isinstance(sample, data_sample.VoiceSample)
    assert sample.messages == [
        {"role": "user", "content": 'Transcribe\n"0"'},
        {"role": "assistant", "content": "0"},
    ]
    assert sample.audio is None


def test_generic_dataset_merge_configs():
    base_config = types.DatasetConfig(
        name="fake_base",
        path="fake_path",
        splits=[
            types.DatasetSplitConfig(
                name="fake", num_samples=5, split=types.DatasetSplit.TRAIN
            )
        ],
    )
    mid_config = types.DatasetConfig(
        name="fake_mid",
        base="fake_base",
        user_template="fake_user_template",
        user_template_args={"a": 1},
        transcript_template="fake_transcript_template",
    )
    leaf_config = types.DatasetConfig(
        name="fake_leaf",
        base="fake_mid",
        audio_field="fake_audio_field",
    )
    config = registry._merge_configs([base_config, mid_config, leaf_config])
    assert config.name == "fake_leaf"
    assert config.base is None
    assert config.path == "fake_path"
    assert config.splits[0].name == "fake"
    assert config.splits[0].num_samples == 5
    assert config.splits[0].split == types.DatasetSplit.TRAIN
    assert config.user_template == "fake_user_template"
    assert config.user_template_args == {"a": 1}
    assert config.assistant_template == "{{text}}"  # the default
    assert config.transcript_template == "fake_transcript_template"
    assert config.audio_field == "fake_audio_field"


# This test is disabled as we don't have a good way to measure the actual length of the dataset when num_workers > 1
# def test_generic_dataset_length_mismatch():
#     config = types.DatasetConfig(
#         name="fake_dataset",
#         path="fake_path",
#         splits=[
#             types.DatasetSplitConfig(
#                 name="fake", num_samples=5, split=types.DatasetSplit.TRAIN
#             )
#         ],
#     )
#     ds = FakeGenericDataset(10, config)
#     assert len(ds) == 5

#     pattern = r"(has been exceeded|Mismatch between presumed length)"
#     with pytest.warns(UserWarning, match=pattern):
#         list(ds)

#     config = types.DatasetConfig(
#         name="fake_dataset",
#         path="fake_path",
#         splits=[
#             types.DatasetSplitConfig(
#                 name="fake", num_samples=10, split=types.DatasetSplit.TRAIN
#             )
#         ],
#     )
#     ds = FakeGenericDataset(5, config)
#     assert len(ds) == 10

#     with pytest.warns(UserWarning, match="Mismatch between presumed length"):
#         list(ds)


def test_generic_dataset_multiple_splits():
    config = types.DatasetConfig(
        name="fake_dataset",
        path="fake_path",
        splits=[
            types.DatasetSplitConfig(name="train", num_samples=90),
            types.DatasetSplitConfig(name="validation", num_samples=10),
        ],
    )
    ds = FakeGenericDataset(100, config)
    assert len(ds) == 90
    ds = FakeGenericDataset(
        100, config, types.VoiceDatasetArgs(split=types.DatasetSplit.VALIDATION)
    )
    assert len(ds) == 10


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
