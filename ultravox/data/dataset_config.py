import dataclasses
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class StopStrategy(str, Enum):
    FIRST_EXHAUSTED = "first_exhausted"
    LAST_EXHAUSTED = "last_exhausted"
    NEVER_STOP = "never_stop"


class DataDictConfig(BaseModel):
    path: str  # Name of the dataset, or huggingface dataset config/subset
    name: Optional[str] = None
    splits: List[str] = dataclasses.field(default_factory=list)
    num_samples: Optional[int] = None
    total_samples: int = 1
    streaming: bool = True
    user_template: str = "<|audio|>"
    assistant_template: str = "{{text}}"
    transcript_template: str = "{{text}}"

    def post_init(self):
        if not self.splits:
            raise ValueError("At least one split must be provided")


class DatasetMultiplier(BaseModel):
    dataset: DataDictConfig
    multiplier: float


class InterleaveDataConfig(BaseModel):
    # In InterleaveDataset, when to stop interleave: choose from last_exhausted (default), first_exhausted, or never_stop
    stop_strategy: StopStrategy = StopStrategy.LAST_EXHAUSTED
    datasets_with_multiplier: List[DatasetMultiplier]
