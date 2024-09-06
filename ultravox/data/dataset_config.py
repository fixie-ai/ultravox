import dataclasses
from typing import List, Optional, Any

from ultravox.evaluation.eval_types import EvalConfig
from ultravox.utils import string_helpers
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
    audio_field: str = "audio"
    eval_config: Optional[EvalConfig] = None
    alias: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if not self.splits:
            raise ValueError("At least one split must be provided")
        
        if self.alias is None:
            alias = [self.path]
            if self.name:
                alias.append(self.name)
            if self.splits:
                alias.append(":".join(self.splits))
            self.alias = "/".join(alias)
        normalized_alias = string_helpers.normalize_filename(self.alias)
        if normalized_alias != self.alias:
            print(f"Alias '{self.alias}' normalized to '{normalized_alias}' for use as a valid filename")
            self.alias = normalized_alias