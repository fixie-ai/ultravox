import dataclasses
import enum
import json
from typing import Any, Dict, List, Optional

from simple_parsing import helpers

AUDIO_PLACEHOLDER = "<|audio|>"

TRANSLATION_USER_TEMPLATE = f"Please translate the text to {{{{target}}}}. Your response should only include the {{{{target}}}} translation, without any additional words:\n\n{AUDIO_PLACEHOLDER}"
CONTINUATION_USER_TEMPLATE = (
    f"Continue the following text using less than 50 words:\n\n{AUDIO_PLACEHOLDER}"
)
CONTINUATION_ASSISTANT_TEMPLATE = "{{continuation}}"
QA_USER_TEMPLATE = f"Answer the following question:\n\n{AUDIO_PLACEHOLDER}"
TRANSCRIPTION_USER_TEMPLATE = (
    f"Repeat the following text, without any explanation: {AUDIO_PLACEHOLDER}"
)


class DatasetSplit(str, enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclasses.dataclass
class DatasetOptions:
    name: str
    weight: float = 1.0


@dataclasses.dataclass
class VoiceDatasetArgs:
    """Global arguments for train/val/test dataset creation."""

    split: DatasetSplit = DatasetSplit.TRAIN
    """Which split of the dataset to use."""
    include_audio: bool = True
    """Whether to include audio in the samples."""
    shuffle: bool = False
    """Whether to shuffle the dataset."""
    shuffle_seed: int = 42
    """Seed for shuffling the dataset."""
    shuffle_buffer_size: int = 1000
    """Buffer size for shuffling the dataset. Only used for streaming datasets."""
    max_audio_duration_secs: float = 16
    """Whether to skip samples with audio longer than this duration."""
    max_samples: Optional[int] = None
    """max number of samples to use per dataset"""

    def __post_init__(self):
        if isinstance(self.split, str):
            self.split = DatasetSplit(self.split.lower())


@dataclasses.dataclass
class TrainDatasetArgs(VoiceDatasetArgs):
    split: DatasetSplit = DatasetSplit.TRAIN
    shuffle: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.split == DatasetSplit.TRAIN


@dataclasses.dataclass
class ValDatasetArgs(VoiceDatasetArgs):
    split: DatasetSplit = DatasetSplit.VALIDATION
    max_samples: Optional[int] = 64

    def __post_init__(self):
        super().__post_init__()
        assert self.split == DatasetSplit.VALIDATION
        assert self.shuffle is False


@dataclasses.dataclass
class EvalDatasetArgs(VoiceDatasetArgs):
    split: DatasetSplit = DatasetSplit.TEST
    max_audio_duration_secs: float = -1

    def __post_init__(self):
        super().__post_init__()
        assert self.split == DatasetSplit.TEST
        assert self.shuffle is False


@dataclasses.dataclass
class DatasetSplitConfig(helpers.Serializable):
    name: str
    """Name of the split."""
    num_samples: int
    """Number of samples in the split"""
    split: Optional[DatasetSplit] = None
    """Type of split, i.e., train, test, or validation."""

    def __post_init__(self):
        """Automatically set split type based on split name"""
        if self.split is None:
            try:
                self.split = DatasetSplit(self.name.lower())
            except ValueError:
                raise ValueError(
                    f"Could not automatically determine split type from split name '{self.name}'. Please explicitly specify split_type for splits that are not named 'train', 'validation', or 'test'."
                )


# Eval config for a single metric, added to the dataset config
@dataclasses.dataclass
class EvalConfig(helpers.Serializable):
    metric: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DatasetConfig(helpers.Serializable):
    # Note that subclasses can override any of these fields, but they currently can't
    # extend structured fields like splits or user_template_args.
    # See _merge_configs below for the current implementation.
    name: str
    """Name of the dataset."""
    base: Optional[str] = None
    """Base dataset config to inherit from."""
    path: Optional[str] = None
    """Directory of the dataset, or huggingface dataset name; must be set for "generic" datasets. If not set, it is automatically inferred for predefined dataset types."""
    subset: Optional[str] = None
    """Name of the dataset, or huggingface dataset config/subset name."""
    splits: Optional[List[DatasetSplitConfig]] = None
    """List of splits to use, e.g. [{"name": "train", "num_samples": 1000}, {"name": "validation", "num_samples": 100}]."""
    user_template: Optional[str] = None
    """Template for the user message."""
    user_template_args: Optional[Dict[str, str]] = None
    """Optional arguments (e.g., target language) for the user template."""
    assistant_template: Optional[str] = None
    """Template for the assistant message."""
    transcript_template: Optional[str] = None
    """Template for the transcript."""
    audio_field: Optional[str] = None
    """Field in the dataset that contains the audio, use None if the dataset does not contain audio."""
    use_mds: Optional[bool] = None
    """Set to True to load the dataset from GCP (using MDS) instead of Hugging Face."""
    mds_batch_size: Optional[int] = None
    """Batch size for the dataset when using MDS."""
    eval_config: Optional[EvalConfig] = None
    """Eval config for the dataset."""

    def __post_init__(self):
        """Set defaults only if this is a root config, so that said defaults in a subclass don't act as overrides."""
        DEFAULTS = {
            "splits": [],
            "user_template": AUDIO_PLACEHOLDER,
            "user_template_args": {},
            "assistant_template": "{{text}}",
            "transcript_template": "{{text}}",
            "audio_field": "audio",
            "use_mds": False,
            "mds_batch_size": 32,
            "eval_config": {},
        }
        if self.base is None:
            for attr, default_value in DEFAULTS.items():
                if getattr(self, attr) is None:
                    setattr(self, attr, default_value)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
