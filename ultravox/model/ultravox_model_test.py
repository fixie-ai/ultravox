import pytest
import torch
from transformers import WhisperConfig

from ultravox.model import ultravox_model


@pytest.fixture
def encoder():
    config = WhisperConfig(
        max_source_positions=1500,
        d_model=256,
        encoder_attention_heads=4,
        encoder_layers=4,
    )
    return ultravox_model.ModifiedWhisperEncoder(config)


def test_init_latency_mask_none(encoder):
    encoder.init_latency_mask(None, torch.float32)
    assert encoder.audio_streaming_mask is None


def test_init_latency_mask_valid(encoder):
    block_size = 100
    encoder.init_latency_mask(block_size, torch.float32)
    assert encoder.audio_streaming_mask is not None

    assert len(encoder.audio_streaming_mask.shape) == 4
    assert encoder.audio_streaming_mask.shape[0] == 1
    assert encoder.audio_streaming_mask.shape[1] == 1

    mask = encoder.audio_streaming_mask[0, 0]
    # 100*30=3000
    source_mask = (
        torch.tril(torch.ones(30, 30), diagonal=0)
        .repeat_interleave(block_size, dim=0)
        .repeat_interleave(block_size, dim=1)
    )
    source_mask = (1.0 - source_mask) * torch.finfo(torch.float32).min
    print(mask.shape)
    assert torch.allclose(mask, source_mask)


def test_init_latency_mask_invalid_block_size(encoder):
    invalid_block_size = 13

    with pytest.raises(AssertionError, match="must divide .* evenly"):
        encoder.init_latency_mask(invalid_block_size, torch.float32)


def test_init_latency_mask_different_dtypes(encoder):
    block_size = 50
    for dtype in (torch.float32, torch.float16):
        encoder.init_latency_mask(block_size, dtype)
        assert encoder.audio_streaming_mask.min() == torch.finfo(dtype).min


def test_init_latency_mask_persistence(encoder):
    block_size = 50
    encoder.init_latency_mask(block_size, torch.float32)
    assert "audio_streaming_mask" in encoder._buffers
