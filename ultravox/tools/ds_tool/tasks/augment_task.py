import dataclasses
import random
from typing import Any, List

import datasets
import simple_parsing

from ultravox.data import AugRegistry
from ultravox.tools.ds_tool import ds_commons


@dataclasses.dataclass
class AugmentationTask(ds_commons.DSToolTask):

    # Source audio field to process
    audio_column_name: str = simple_parsing.field(default="audio", alias="-a")
    # Random seed for reproducibility
    random_seed: int = simple_parsing.field(default=42, alias="-r")
    # List of augmentations to apply, or yaml file with augmentations
    augmentations: List[str] = simple_parsing.list_field("null", alias="-A")

    def __post_init__(self):
        random.seed(self.random_seed)
        configs = [AugRegistry.get_config(a) for a in self.augmentations]
        augmentations = [AugRegistry.create_augmentation(conf) for conf in configs]
        if len(augmentations) > 1:
            self.augmentation = AugRegistry.create_parent_augmentation(augmentations)
        else:
            self.augmentation = augmentations[0]

    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: set[str],
    ) -> Any:

        print(f"Augmenting {len(ds_split)} samples with {self.augmentation}...")

        ds_split = ds_split.cast_column(
            self.audio_column_name,
            datasets.Audio(sampling_rate=self.augmentation.sample_rate),
        )

        ds_split = ds_split.map(
            self._augment_audio, num_proc=num_proc, writer_batch_size=writer_batch_size
        )

        return ds_split

    def _augment_audio(self, sample: dict[str, Any]) -> dict[str, Any]:
        # augment the samples in place
        audio = sample[self.audio_column_name]
        assert (
            audio["sampling_rate"] == self.augmentation.sample_rate
        ), f"Sample rate mismatch: {audio['sampling_rate']} != {self.augmentation.sample_rate}"
        sample[self.audio_column_name]["array"] = self.augmentation(audio["array"])
        return sample

    @classmethod
    def chunking_allowed(cls) -> bool:
        return True
