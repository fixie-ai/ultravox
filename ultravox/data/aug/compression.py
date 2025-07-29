import audioop
import platform
import subprocess
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf

from .base import Augmentation
from .config import AugmentationArgs
from .config import AugmentationConfig
from .registry import AugRegistry


@AugRegistry.register_type("ffmpeg_compression")
class FfmpegCompression(Augmentation):
    def __init__(
        self, args: AugmentationArgs, codec: str = "amr", bitrate: Optional[int] = None
    ):
        super().__init__(args)
        self.codec = codec
        self.bitrate = bitrate

        # Check if ffmpeg is available
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg is not available. Run `just import-augs-system` to install FFmpeg and necessary encoders."
            )

    def _apply(self, audio: np.ndarray) -> np.ndarray:
        return apply_ffmpeg_compression(
            audio, self.sample_rate, self.codec, self.bitrate
        )


@AugRegistry.register_type("audioop_compression")
class AudioopCompression(Augmentation):
    def __init__(
        self,
        args: AugmentationArgs,
        codec: str = "mulaw",
        bit_depth: int = 16,
    ):
        super().__init__(args)
        self.codec = codec
        self.bit_depth = bit_depth

    def _apply(self, audio: np.ndarray) -> np.ndarray:
        return apply_audioop_compression(audio, self.codec, self.bit_depth)


def apply_ffmpeg_compression(
    audio: np.ndarray,
    sr: int,
    codec: str,
    bitrate: Optional[int] = None,
) -> np.ndarray:
    """Generic FFmpeg codec application.

    Args:
        audio: Input audio signal (-1 to 1)
        sr: Sample rate
        codec: FFmpeg codec name (e.g., 'libopus', 'amrwbenc', 'g722')
        ext: File extension for the compressed format
        bitrate: Optional bitrate in bits per second
    """

    EXT_MAP = {
        "amr": "amr",
        "amrwb": "amr",
    }

    assert codec in EXT_MAP, f"Unsupported codec: {codec}"

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as wav_in, tempfile.NamedTemporaryFile(
        suffix=f".{EXT_MAP[codec]}"
    ) as compressed, tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as wav_out:

        # Write input audio to WAV
        sf.write(wav_in.name, audio, sr)

        # Encode to compressed format and decode back to WAV
        if codec == "amr":
            assert bitrate is not None, "Bitrate must be specified for AMR compression"
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    wav_in.name,
                    "-ac",
                    "1",
                    "-ar",
                    "8000",
                    "-ab",
                    str(int(bitrate)),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    compressed.name,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    compressed.name,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    wav_out.name,
                ],
                check=True,
            )
        elif codec == "amrwb":
            assert (
                bitrate is not None
            ), "Bitrate must be specified for AMR-WB compression"
            if platform.system() == "Darwin":
                raise SystemError("AMR-WB compression is not supported on macOS")
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    wav_in.name,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-ab",
                    str(int(bitrate)),
                    "-acodec",
                    "amr_wb",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    compressed.name,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    compressed.name,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    wav_out.name,
                ],
                check=True,
            )
        else:
            raise ValueError(f"Unsupported codec: {codec}")

        # Read processed audio
        processed_audio, _ = sf.read(wav_out.name)

        return processed_audio


def apply_audioop_compression(
    audio: np.ndarray, codec: str = "mulaw", bit_depth: int = 16
) -> np.ndarray:
    audio_int16 = (audio * 32767).astype(np.int16)
    bytes_per_sample = bit_depth // 8
    if codec == "mulaw":
        compressed = audioop.lin2ulaw(audio_int16.tobytes(), bytes_per_sample)
        decompressed = audioop.ulaw2lin(compressed, bytes_per_sample)
    elif codec == "alaw":
        compressed = audioop.lin2alaw(audio_int16.tobytes(), bytes_per_sample)
        decompressed = audioop.alaw2lin(compressed, bytes_per_sample)
    else:
        raise ValueError(f"Unsupported codec: {codec}")
    return np.frombuffer(decompressed, dtype=np.int16).astype(np.float32) / 32767.0


@AugRegistry.register_type("amr_compression")
class AmrCompression(Augmentation):
    def __init__(self, args: AugmentationArgs):
        super().__init__(args)

    def _apply(self, audio: np.ndarray) -> np.ndarray:
        amr_nb_bitrates = [4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200]
        amr_wb_bitrates = [6600, 8850, 12650, 14250, 15850, 18250, 19850, 23050, 23850]
        bitrates = amr_nb_bitrates + amr_wb_bitrates
        compression_codes = ["amr"] * len(amr_nb_bitrates) + ["amrwb"] * len(
            amr_wb_bitrates
        )

        idx = np.random.randint(0, len(bitrates))
        code = compression_codes[idx]
        bitrate = bitrates[idx]
        return apply_ffmpeg_compression(audio, self.sample_rate, code, bitrate)


# configs
AugRegistry.register_config(
    "mulaw_compression",
    AugmentationConfig(
        name="mulaw_compression",
        type="audioop_compression",
        params={"codec": "mulaw", "bit_depth": 16},
    ),
)

AugRegistry.register_config(
    "alaw_compression",
    AugmentationConfig(
        name="alaw_compression",
        type="audioop_compression",
        params={"codec": "alaw", "bit_depth": 16},
    ),
)

AugRegistry.register_config(
    "amr_4_75kbps",
    AugmentationConfig(
        name="amr_4_75kbps",
        type="ffmpeg_compression",
        params={"codec": "amr", "bitrate": 4750},
    ),
)


AugRegistry.register_config(
    "amr_12kbps",
    AugmentationConfig(
        name="amr_12kbps",
        type="ffmpeg_compression",
        params={"codec": "amr", "bitrate": 12200},
    ),
)

AugRegistry.register_config(
    "amr_wb",
    AugmentationConfig(
        name="amr_wb",
        type="ffmpeg_compression",
        params={"codec": "amrwb", "bitrate": 23850},
    ),
)

AugRegistry.register_config(
    "random_amr_compression",
    AugmentationConfig(
        name="random_amr_compression",
        type="amr_compression",
    ),
)
