import numpy as np
import scipy.signal as signal

from .base import Augmentation
from .config import AugmentationArgs
from .config import AugmentationConfig
from .registry import AugRegistry


@AugRegistry.register_type("bandpass_filter")
class BandpassFilter(Augmentation):
    def __init__(
        self,
        args: AugmentationArgs,
        lowcut: float = 300,
        highcut: float = 3400,
        order: int = 5,  # sharp rolloff
    ):
        super().__init__(args)
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def _apply(self, audio: np.ndarray) -> np.ndarray:
        return apply_bandpass_filter(
            audio, self.sample_rate, self.lowcut, self.highcut, self.order
        )


@AugRegistry.register_type("lowpass_filter")
class LowpassFilter(Augmentation):
    def __init__(
        self,
        args: AugmentationArgs,
        cutoff: float = 3400,
        order: int = 2,
    ):
        super().__init__(args)
        self.cutoff = cutoff
        self.order = order

    def _apply(self, audio: np.ndarray) -> np.ndarray:
        return apply_lowpass_filter(audio, self.sample_rate, self.cutoff, self.order)


def apply_bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    lowcut: float = 300,
    highcut: float = 3400,
    order: int = 5,
) -> np.ndarray:
    nyquist = sample_rate / 2
    lowcut = lowcut / nyquist
    highcut = highcut / nyquist
    b, a = signal.butter(order, [lowcut, highcut], btype="band")
    audio = signal.lfilter(b, a, audio)
    return audio


def apply_lowpass_filter(
    audio: np.ndarray, sample_rate: int, cutoff: float = 3400, order: int = 2
) -> np.ndarray:
    nyquist = sample_rate / 2
    cutoff = cutoff / nyquist
    b, a = signal.butter(order, cutoff, btype="low")
    audio = signal.lfilter(b, a, audio)
    return audio


AugRegistry.register_config(
    "telephone_bandpass",
    AugmentationConfig(
        name="telephone_bandpass",
        type="bandpass_filter",
        params={"lowcut": 300, "highcut": 3400, "order": 5},
    ),
)

AugRegistry.register_config(
    "telephone_lowpass",
    AugmentationConfig(
        name="telephone_lowpass",
        type="lowpass_filter",
        params={"cutoff": 3400, "order": 2},
    ),
)
