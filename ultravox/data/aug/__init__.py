from ultravox.data.aug.base import Augmentation
from ultravox.data.aug.config import AugmentationArgs
from ultravox.data.aug.config import AugmentationConfig
from ultravox.data.aug.registry import AugRegistry

# Need to import all augmentation modules to register them in the registry
from ultravox.data.aug.gain import *  # noqa: F401, F403  # isort: skip
from ultravox.data.aug.filter import *  # noqa: F401, F403  # isort: skip
from ultravox.data.aug.noise import *  # noqa: F401, F403  # isort: skip
from ultravox.data.aug.resampling import *  # noqa: F401, F403  # isort: skip
from ultravox.data.aug.compression import *  # noqa: F401, F403  # isort: skip

__all__ = [
    "AugRegistry",
    "AugmentationConfig",
    "AugmentationArgs",
    "Augmentation",
]
