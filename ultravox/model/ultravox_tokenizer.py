import logging

import transformers

AUDIO_TOKEN = "<|audio|>"


def from_pretrained_text_tokenizer(
    *args, **kwargs
) -> transformers.PreTrainedTokenizerBase:
    """
    Create a tokenizer with the additional special token for audio.
    This is mainly used for VLLM to work properly. This repo does not currently require it.
    """

    tokenizer = transformers.AutoTokenizer.from_pretrained(*args, **kwargs)
    tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_TOKEN]})
    logging.info(f"Audio token id: {get_audio_token_id(tokenizer)}")
    return tokenizer


def get_audio_token_id(tokenizer: transformers.PreTrainedTokenizerBase) -> int:
    audio_token_id = tokenizer.encode(AUDIO_TOKEN, add_special_tokens=False)
    assert len(audio_token_id) == 1, "Audio token should be a single token"
    return audio_token_id[0]
