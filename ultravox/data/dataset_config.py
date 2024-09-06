import dataclasses
from typing import List, Optional

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    # Path to the dataset, or huggingface dataset id
    path: str
    # Name of the dataset, or huggingface dataset config/subset
    name: Optional[str] = None
    splits: List[str] = dataclasses.field(default_factory=list)
    num_samples: Optional[int] = None
    total_samples: int = 1
    weight: float = 1.0
    streaming: bool = True
    user_template: str = "<|audio|>"
    assistant_template: str = "{{text}}"
    transcript_template: str = "{{text}}"

    def __post_init__(self):
        if not self.splits:
            raise ValueError("At least one split must be provided")
