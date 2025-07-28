import librosa
import numpy as np

from .base import Augmentation
from .config import AugmentationArgs
from .config import AugmentationConfig
from .registry import AugRegistry


@AugRegistry.register_type("resampling")
class Resampling(Augmentation):
    def __init__(
        self,
        args: AugmentationArgs,
        target_sr: int = 8000,
    ):
        super().__init__(args)
        self.target_sr = target_sr

    def _apply(self, audio: np.ndarray) -> np.ndarray:
        return apply_resampling(audio, self.sample_rate, self.target_sr)


def apply_resampling(audio: np.ndarray, sample_rate: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target_sr, then back to sample_rate.
    """
    audio = librosa.resample(
        audio, orig_sr=sample_rate, target_sr=target_sr, res_type="soxr_hq"
    )
    audio = librosa.resample(
        audio, orig_sr=target_sr, target_sr=sample_rate, res_type="soxr_hq"
    )
    return audio


AugRegistry.register_config(
    "8kHz_resample",
    AugmentationConfig(
        name="8kHz_resample",
        type="resampling",
        params={"target_sr": 8000},
    ),
)
