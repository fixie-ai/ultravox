import random
from typing import List, Optional, overload

import numpy as np

from ultravox.data.data_sample import VoiceSample

from .config import AugmentationArgs


class Augmentation:
    def __init__(
        self,
        args: AugmentationArgs,
        children: Optional[List["Augmentation"]] = None,
    ):
        super().__init__()
        assert 0 <= args.p <= 1, "Probability must be between 0 and 1"
        self.p = args.p
        self.children = children or []
        self.sample_rate = args.sample_rate

    def __call__(self, audio: np.ndarray | None) -> np.ndarray | None:
        """Apply augmentation with probability p unless dataset is excluded."""
        if random.random() > self.p:
            return audio

        # Otherwise apply this augmentation
        return self.apply(audio)

    def apply_sample(self, sample: VoiceSample) -> VoiceSample:
        """Apply this augmentation to a sample."""
        if sample.audio is None:
            return sample
        sample.audio = self(sample.audio)
        return sample

    @overload
    def apply(self, audio: None) -> None: ...

    @overload
    def apply(self, audio: np.ndarray) -> np.ndarray: ...

    def apply(self, audio: np.ndarray | None) -> np.ndarray | None:
        """Apply this augmentation."""
        if audio is None:
            return None

        if self.children:
            for aug in self.children:
                audio = aug(audio)
                if audio is None:
                    return None
        return self._apply(audio)

    def _apply(self, audio: np.ndarray) -> np.ndarray:
        """Implementation of the actual augmentation."""
        return audio
