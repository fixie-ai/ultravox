from ultravox.data import types

VP_EN_CONFIG = types.DatasetConfig(
    name="voxpopuli-en",
    path="facebook/voxpopuli",
    subset="en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_000_000),
        types.DatasetSplitConfig(name="validation", num_samples=10_000),
    ],
    assistant_template="{{raw_text}}",
    transcript_template="{{raw_text}}",
)

configs = [
    VP_EN_CONFIG,
]
