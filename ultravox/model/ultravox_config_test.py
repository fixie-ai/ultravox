from typing import Dict, Tuple

import pytest
import transformers

from ultravox.model import ultravox_config


def exclude_key(d: Dict, key_to_exclude: Tuple) -> Dict:
    """Exclude a specific key from a dictionary."""
    return {k: v for k, v in d.items() if k not in key_to_exclude}


@pytest.mark.parametrize(
    "model_id",
    ["fixie-ai/ultravox-v0_2", "fixie-ai/ultravox-v0_3", "fixie-ai/ultravox-v0_4"],
)
def test_can_load_release(model_id: str):
    orig_config: transformers.PretrainedConfig = (
        transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    )
    config_from_dict = ultravox_config.UltravoxConfig(**orig_config.to_dict())
    config_from_diff_dict = ultravox_config.UltravoxConfig(**orig_config.to_diff_dict())
    keys_to_ignore = ("audio_latency_block_size",)

    assert (
        exclude_key(config_from_dict.to_dict(), keys_to_ignore) == orig_config.to_dict()
    )
    assert (
        exclude_key(config_from_diff_dict.to_dict(), keys_to_ignore)
        == orig_config.to_dict()
    )

    assert config_from_dict.text_config.to_dict() == orig_config.text_config.to_dict()
    assert config_from_dict.audio_config.to_dict() == orig_config.audio_config.to_dict()

    config_reloaded = ultravox_config.UltravoxConfig(**config_from_dict.to_dict())
    config_reloaded_diff = ultravox_config.UltravoxConfig(
        **config_from_dict.to_diff_dict()
    )
    assert (
        exclude_key(config_reloaded.to_dict(), keys_to_ignore) == orig_config.to_dict()
    )
    assert (
        exclude_key(config_reloaded_diff.to_dict(), keys_to_ignore)
        == orig_config.to_dict()
    )


def test_no_config_when_id_present():
    config = ultravox_config.UltravoxConfig(audio_model_id="openai/whisper-small")
    assert "audio_config" not in config.to_diff_dict()

    config = ultravox_config.UltravoxConfig(text_model_id="microsoft/phi-2")
    assert "text_config" not in config.to_diff_dict()
