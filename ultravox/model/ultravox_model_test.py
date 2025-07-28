import os

import pytest
import safetensors.torch
import torch
import transformers

from ultravox.model import ultravox_config
from ultravox.model import ultravox_model

TINY_MODEL_PATH = "./assets/tiny_ultravox"


@pytest.fixture
def encoder():
    config = transformers.WhisperConfig(
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


def assert_equal_state_dict(dict_a, dict_b):
    assert set(dict_a.keys()) == set(dict_b.keys())
    for k, v in dict_a.items():
        torch.testing.assert_close(
            v,
            dict_b[k],
            msg=f"Key {k} does not match",
            atol=0,
            rtol=2**-8,
            check_dtype=False,
        )


def create_tiny_model():
    """
    This is a helper that was used to create the tiny model for test assets.
    """
    config = ultravox_config.UltravoxConfig(
        text_model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        audio_model_id="optimum-internal-testing/tiny-random-whisper",
        torch_dtype=torch.bfloat16,
        pad_token_id=0,
        projector_ln_mid=False,
        audio_model_lora_config=ultravox_config.LoraConfigSimplified(r=4),
    )
    model = ultravox_model.UltravoxModel(config)
    model.save_pretrained(TINY_MODEL_PATH)
    safetensors.torch.save_file(
        model.state_dict(), os.path.join(TINY_MODEL_PATH, "full_model.safetensors")
    )


def test_load_pretrained_model():
    # We run create_tiny_model() once on a working model to create the test assets.
    orig_state_dict = safetensors.torch.load_file(
        os.path.join(TINY_MODEL_PATH, "full_model.safetensors")
    )
    model = ultravox_model.UltravoxModel.from_pretrained(
        TINY_MODEL_PATH, low_cpu_mem_usage=True
    )
    assert_equal_state_dict(orig_state_dict, model.state_dict())
