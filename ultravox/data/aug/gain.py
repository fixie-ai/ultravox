import numpy as np

from .base import Augmentation
from .config import AugmentationArgs
from .config import AugmentationConfig
from .registry import AugRegistry


@AugRegistry.register_type("gain")
class Gain(Augmentation):
    def __init__(
        self,
        args: AugmentationArgs,
        gain_db: float = 0.0,
    ):
        super().__init__(args)
        self.gain_db = gain_db

    def _apply(self, waveform: np.ndarray) -> np.ndarray:
        return apply_gain(waveform, self.gain_db)


@AugRegistry.register_type("random_gain")
class RandomGain(Augmentation):
    def __init__(
        self,
        args: AugmentationArgs,
        min_gain_db: float = -24.0,
        max_gain_db: float = 15.0,
        # defaults from Moshi
    ):
        super().__init__(args)
        assert min_gain_db <= max_gain_db
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def _apply(self, waveform: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(self.min_gain_db, self.max_gain_db)
        return apply_gain(waveform, gain_db)


def apply_gain(waveform: np.ndarray, gain_db: float = 0.0) -> np.ndarray:
    gain_linear = 10 ** (gain_db / 20)
    return np.clip(waveform * gain_linear, -1.0, 1.0)


# configs
AugRegistry.register_config(
    "gain",
    AugmentationConfig(
        name="gain", type="gain", params={"gain_db": 0.0}, description="Apply no gain."
    ),
)

AugRegistry.register_config(
    "gain_10db",
    AugmentationConfig(
        name="gain_10db",
        type="gain",
        params={"gain_db": 10.0},
        description="Apply 10dB gain.",
    ),
)

AugRegistry.register_config(
    "gain_minus_10db",
    AugmentationConfig(
        name="gain_minus_10db",
        type="gain",
        params={"gain_db": -10.0},
        description="Apply -10dB gain.",
    ),
)

AugRegistry.register_config(
    "random_gain",
    AugmentationConfig(
        name="random_gain",
        type="random_gain",
        params={"min_gain_db": -24.0, "max_gain_db": 15.0},
        description="Apply random gain.",
    ),
)
