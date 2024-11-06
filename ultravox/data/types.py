import dataclasses
import enum
from typing import Dict, List, Optional

from simple_parsing import helpers

AUDIO_PLACEHOLDER = "<|audio|>"

TRANSLATION_USER_TEMPLATE = f"Please translate the text to {{{{target}}}}. Your response should only include the {{{{target}}}} translation, without any additional words:\n\n{AUDIO_PLACEHOLDER}"
CONTINUATION_USER_TEMPLATE = (
    f"Continue the following text using less than 50 words:\n\n{AUDIO_PLACEHOLDER}"
)
CONTINUATION_ASSISTANT_TEMPLATE = "{{continuation}}"
TRANSCRIPTION_USER_TEMPLATE = f"Transcribe\n{AUDIO_PLACEHOLDER}"


class DatasetSplit(str, enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclasses.dataclass
class VoiceDatasetArgs:
    """Global arguments for voice datasets."""

    batch_size: int = 4
    """Batch size for train, eval, or validation."""
    include_audio: bool = True
    """Whether to include audio in the samples."""
    shuffle: bool = False
    """Whether to shuffle the dataset."""
    shuffle_seed: int = 42
    """Seed for shuffling the dataset."""
    max_audio_duration_secs: Optional[float] = None
    """Whether to skip samples with audio longer than this duration."""
    split: DatasetSplit = DatasetSplit.TRAIN
    """Which split of the dataset to use."""

    def __post_init__(self):
        if isinstance(self.split, str):
            self.split = DatasetSplit(self.split.lower())


@dataclasses.dataclass
class DatasetSplitConfig(helpers.Serializable):
    name: str
    """Name of the split."""
    num_samples: int
    """Number of samples in the split"""
    split_type: Optional[DatasetSplit] = None
    """Type of split, i.e., train, test, or validation."""

    def __post_init__(self):
        """Automatically set split type based on split name"""
        if self.split_type is None:
            try:
                self.split_type = DatasetSplit(self.name.lower())
            except ValueError:
                raise ValueError(
                    f"Could not automatically determine split type from split name '{self.name}'. Please explicitly specify split_type for splits that are not named 'train', 'validation', or 'test'."
                )


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
        }
        if self.base is None:
            for attr, default_value in DEFAULTS.items():
                if getattr(self, attr) is None:
                    setattr(self, attr, default_value)
