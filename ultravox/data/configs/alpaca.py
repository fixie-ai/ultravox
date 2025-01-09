from ultravox.data import types

# convert original Alpaca dataset to use F5-TTS generated audio
ALPACA_TTS_CONFIG = types.DatasetConfig(
    name="alpaca-tts",
    path="fixie-ai/alpaca-tts",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=33327),
        types.DatasetSplitConfig(
            name="validation",
            num_samples=1024,
            split=types.DatasetSplit.VALIDATION,
        ),
        types.DatasetSplitConfig(
            name="validation", num_samples=1024, split=types.DatasetSplit.TEST
        ),  # reuse the validation set for testing
    ],
    user_template=f"{types.AUDIO_PLACEHOLDER}",
    assistant_template="{{output}}",
    transcript_template="{{user}}",
    eval_config=types.EvalConfig(metric="bleu"),
)

# adapt fixie-ai/alpaca-tts to use llama-3.1-8b-instruct generated response (limit response to 50 words)
ALPACA_TTS_LLAMA_CONFIG = types.DatasetConfig(
    name="alpaca-tts-llama",
    path="fixie-ai/alpaca-tts-llama-v1",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=33327),
        types.DatasetSplitConfig(
            name="validation",
            num_samples=1024,
            split=types.DatasetSplit.VALIDATION,
        ),
        types.DatasetSplitConfig(
            name="validation", num_samples=1024, split=types.DatasetSplit.TEST
        ),  # reuse the validation set for testing
    ],
    user_template=f"Please respond to the following statement using less than 50 words: \n\n{types.AUDIO_PLACEHOLDER}",
    assistant_template="{{response}}",
    transcript_template="{{user}}",
    eval_config=types.EvalConfig(metric="bleu"),
)

configs = [
    ALPACA_TTS_CONFIG,
    ALPACA_TTS_LLAMA_CONFIG,
]
