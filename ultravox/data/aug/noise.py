import numpy as np

from .base import Augmentation
from .config import AugmentationArgs
from .config import AugmentationConfig
from .registry import AugRegistry


@AugRegistry.register_type("colored_noise")
class ColoredNoise(Augmentation):
    def __init__(
        self, args: AugmentationArgs, noise_type: str = "white", snr_db: float = 20
    ):
        super().__init__(args)
        assert noise_type in ["white", "pink", "brown", "blue", "violet"]
        self.noise_type = noise_type
        self.snr_db = snr_db

    def _apply(self, waveform: np.ndarray) -> np.ndarray:
        return add_colored_noise(waveform, self.noise_type, self.snr_db)


@AugRegistry.register_type("random_colored_noise")
class RandomColoredNoise(Augmentation):
    def __init__(
        self,
        args: AugmentationArgs,
        noise_type: str = "white",
        min_snr_db: float = 0.0,
        max_snr_db: float = 20.0,
    ):
        super().__init__(args)
        assert noise_type in ["white", "pink", "brown", "blue", "violet"]
        self.noise_type = noise_type
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def _apply(self, waveform: np.ndarray) -> np.ndarray:
        snr_db = np.random.uniform(self.min_snr_db, self.max_snr_db)
        return add_colored_noise(waveform, self.noise_type, snr_db)


def generate_colored_noise(length, noise_type="white"):
    """
    Generate different colors of noise.

    Args:
        length: Length of noise array to generate
        noise_type: Type of noise ('white', 'pink', 'brown', 'blue', 'violet')
    """
    # Generate white noise
    white = np.random.normal(0, 1, length)

    if noise_type == "white":
        return white

    # Generate frequency array
    f = np.fft.fftfreq(length)
    f = np.abs(f)
    f[0] = f[1]  # Avoid divide by zero

    # Different noise colors have different spectral densities
    spectral_density = {
        "pink": f ** (-1.0),  # 1/f
        "brown": f ** (-2.0),  # 1/f^2
        "blue": f ** (1.0),  # f
        "violet": f ** (2.0),  # f^2
    }

    if noise_type not in spectral_density:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    # Generate colored noise
    white_fft = np.fft.fft(white)
    colored_fft = white_fft * np.sqrt(spectral_density[noise_type])
    colored = np.fft.ifft(colored_fft)

    return colored.real


def add_colored_noise(waveform, noise_type="white", snr_db=20):
    """
    Add colored noise to the audio waveform with specified SNR in dB.

    Args:
        waveform: Input audio signal
        noise_type: Type of noise ('white', 'pink', 'brown', 'blue', 'violet')
        target_snr_db: Target Signal-to-Noise Ratio in dB
    """
    # Calculate signal power
    signal_power = np.mean(waveform**2)

    # Convert SNR from dB to linear scale
    target_snr_linear = 10 ** (snr_db / 10)

    # Calculate noise power based on SNR
    noise_power = signal_power / target_snr_linear

    # Generate colored noise
    noise = generate_colored_noise(len(waveform), noise_type)

    # Normalize noise to desired power
    current_noise_power = np.mean(noise**2)
    noise = noise * np.sqrt(noise_power / current_noise_power)

    return waveform + noise


AugRegistry.register_config(
    "white_noise",
    AugmentationConfig(
        name="white_noise",
        type="colored_noise",
        params={"noise_type": "white", "snr_db": 20},
    ),
)

AugRegistry.register_config(
    "pink_noise",
    AugmentationConfig(
        name="pink_noise",
        type="colored_noise",
        params={"noise_type": "pink", "snr_db": 20},
    ),
)

AugRegistry.register_config(
    "random_pink_noise",
    AugmentationConfig(
        name="random_pink_noise",
        type="random_colored_noise",
        params={"noise_type": "pink", "min_snr_db": 0.0, "max_snr_db": 20.0},
    ),
)
