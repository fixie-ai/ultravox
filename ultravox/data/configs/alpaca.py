from ultravox.data import types

ALPACA_CONFIG = types.DatasetConfig(
    name="alpaca",
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
)

configs = [
    ALPACA_CONFIG,
]
