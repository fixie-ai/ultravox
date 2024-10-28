from typing import Union

import numpy as np
import pytest

from ultravox.data import data_sample


def _create_sine_wave(
    freq: int = 440,
    duration: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.1,
    target_dtype: str = "float32",
) -> Union[
    np.typing.NDArray[np.float32],
    np.typing.NDArray[np.float64],
    np.typing.NDArray[np.int16],
    np.typing.NDArray[np.int32],
]:
    t = np.arange(sample_rate * duration, dtype=np.float32) / sample_rate
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    match target_dtype:
        case "int16":
            wave = np.int16(wave * 32767)
        case "int32":
            wave = np.int32(wave * 2147483647)
        case "float32":
            # Already float32, nothing needed.
            pass
        case "float64":
            wave = wave.astype(np.float64)
        case _:
            raise ValueError(f"Unsupported dtype: {target_dtype}")
    return wave


def _create_and_validate_sample(target_dtype: str = "float32"):
    # Create a sine wave at 440 Hz with a duration of 1.0 second, sampled at 16
    # kHz, with an amplitude of 0.1, and the specified dtype.
    array = _create_sine_wave(target_dtype=target_dtype)
    sample = data_sample.VoiceSample.from_prompt_and_raw(
        "Transcribe\n<|audio|>", array, 16000
    )
    assert sample.sample_rate == 16000
    assert sample.audio is not None, "sample.audio should not be None"
    assert len(sample.audio) == 16000
    assert sample.audio.dtype == np.float32
    assert sample.messages == [
        {"role": "user", "content": "Transcribe\n<|audio|>"},
    ]
    # Serialize and deserialize the sample.
    json = sample.to_json()
    sample2 = data_sample.VoiceSample.from_json(json)
    assert sample2.sample_rate == sample.sample_rate
    assert sample2.audio is not None, "sample2.audio should not be None"
    assert len(sample2.audio) == len(sample.audio)
    assert sample2.audio.dtype == sample.audio.dtype
    assert sample2.messages == sample.messages
    assert np.allclose(sample2.audio, sample.audio, rtol=0.0001, atol=0.0001)


def test_create_sample__int16():
    _create_and_validate_sample("int16")


def test_create_sample__int32():
    _create_and_validate_sample("int32")


def test_create_sample__float32():
    _create_and_validate_sample("float32")


def test_create_sample__float64():
    _create_and_validate_sample("float64")


def test_create_sample__raises_on_unsupported_dtype():
    with pytest.raises(AssertionError):
        array = np.ndarray(shape=(16000,), dtype=np.uint8)
        _ = data_sample.VoiceSample.from_prompt_and_raw(
            "Transcribe\n<|audio|>", array, 16000
        )
