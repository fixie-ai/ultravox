import base64
import dataclasses
import io
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
from numpy import typing as npt

SAMPLE_RATE = 16000


def audio_from_file(path: str) -> np.ndarray:
    """Load audio from a file, converting to float32 PCM @ 16 kHz."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    assert audio.dtype == np.float32
    return audio


def audio_from_buf(buf: bytes) -> np.ndarray:
    """Load audio from a buffer, converting to float32 PCM @ 16 kHz."""
    audio, _ = librosa.load(io.BytesIO(buf), sr=SAMPLE_RATE)
    assert audio.dtype == np.float32
    return audio


def audio_to_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert audio to WAV format, 16-bit PCM @ 16 kHz."""
    assert audio.dtype == np.float32
    with io.BytesIO() as buf:
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()


def audio_to_wav_base64(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Convert audio to a base64-encoded WAV file."""
    return base64.b64encode(audio_to_wav(audio, sample_rate)).decode("utf-8")


def audio_to_data_uri(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Convert audio to a data URI."""
    return f"data:audio/wav;base64,{audio_to_wav_base64(audio, sample_rate)}"


def messages_from_prompt(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt}]


@dataclasses.dataclass
class VoiceSample:
    @staticmethod
    def from_json(data: Dict[str, Any]) -> "VoiceSample":
        """Convert from JSON format; audio is expected as base64ed WAV."""
        bytes = base64.b64decode(data["audio"])
        return VoiceSample(data["messages"], audio_from_buf(bytes))

    @staticmethod
    def from_prompt(prompt: str) -> "VoiceSample":
        """Create a VoiceSample from a prompt only."""
        return VoiceSample(messages_from_prompt(prompt), None)

    @staticmethod
    def from_prompt_and_file(prompt: str, path: str) -> "VoiceSample":
        """Create a VoiceSample from a prompt and an audio file."""
        return VoiceSample(messages_from_prompt(prompt), audio_from_file(path))

    @staticmethod
    def from_prompt_and_buf(prompt: str, buf: bytes) -> "VoiceSample":
        """Create a VoiceSample from a prompt and an encoded audio buffer."""
        return VoiceSample(messages_from_prompt(prompt), audio_from_buf(buf))

    @staticmethod
    def from_prompt_and_raw(
        prompt: str, buf: np.ndarray, sample_rate: int
    ) -> "VoiceSample":
        """Create a VoiceSample from a prompt and raw audio data with sample rate."""
        # Keep in native sample rate; we'll resample later if needed.
        return VoiceSample(messages_from_prompt(prompt), buf, sample_rate)

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON format; audio is written as base64ed WAV."""
        obj: Dict[str, Any] = {"messages": self.messages}
        if self.audio is not None:
            obj["audio"] = audio_to_wav_base64(self.audio, self.sample_rate)
        return obj

    def __post_init__(self):
        """Ensure audio is float32 PCM."""
        if self.audio is not None:
            if self.audio.dtype == np.float64:
                self.audio = self.audio.astype(np.float32)
            elif self.audio.dtype == np.int16:
                self.audio = self.audio.astype(np.float32) / np.float32(32768.0)
            elif self.audio.dtype == np.int32:
                self.audio = self.audio.astype(np.float32) / np.float32(2147483648.0)
            assert (
                self.audio.dtype == np.float32
            ), f"Unexpected audio dtype: {self.audio.dtype}"
            assert self.audio.ndim == 1, f"Unexpected audio shape: {self.audio.shape}"

    def add_past_messages(self, past_messages: List[Dict[str, str]]):
        self.messages = past_messages + self.messages

    messages: List[Dict[str, str]]
    """List of messages, each with a "role" and "content" field."""
    audio: Optional[npt.NDArray[np.float32]] = None
    """Audio data as float32 PCM @ `sample_rate`."""
    sample_rate: int = SAMPLE_RATE
    """Audio sample rate in Hz."""
    audio_transcript: Optional[str] = None
    """For evaluations, the known transcript of the audio."""
