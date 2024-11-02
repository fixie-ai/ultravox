from ultravox.data import types

BOOLQ_CONFIG = types.DatasetConfig(
    name="boolq",
    path="fixie-ai/boolq-audio",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=10000),
        types.DatasetSplitConfig(name="validation", num_samples=1000),
    ],
    user_template=f"{{{{passage}}}}\n\n{types.AUDIO_PLACEHOLDER}",
    assistant_template="{{'True' if answer else 'False'}}",
    transcript_template="{{question}}",
)

configs = [
    BOOLQ_CONFIG,
]
