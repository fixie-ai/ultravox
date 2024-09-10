import transformers
from ultravox.training.helpers import prefetch_weights

TEXT_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
AUDIO_MODEL = "hf-internal-testing/tiny-random-WhisperForCausalLM"


def test_prefetch_weights():
    prefetch_weights.main(["--text-model", TEXT_MODEL, "--audio-model", AUDIO_MODEL])

    # With local_files_only=True, from_pretrained will throw an error if the weights are not downloaded
    transformers.AutoModel.from_pretrained(TEXT_MODEL, local_files_only=True)
    transformers.AutoModel.from_pretrained(AUDIO_MODEL, local_files_only=True)
