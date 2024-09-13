from ultravox.training.helpers import prefetch_weights

TEXT_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
AUDIO_MODEL = "hf-internal-testing/tiny-random-WhisperForCausalLM"


def test_prefetch_weights():
    # It would be nice to test this, but there isn't an easy way to clear the cache
    # with pytest.raises(huggingface_hub.utils.LocalEntryNotFoundError):
    #     prefetch_weights.raise_on_weights_not_downloaded([TEXT_MODEL, AUDIO_MODEL])

    prefetch_weights.main(["--text-model", TEXT_MODEL, "--audio-model", AUDIO_MODEL])

    prefetch_weights.raise_on_weights_not_downloaded([TEXT_MODEL, AUDIO_MODEL])
