"""
Datasets from the unlabeled LibriLight dataset.

subset,               hours, books,  files, per-spk hours, total spks
------------------------------------------------------------------------
small (unlab-600),    577.2,   202,   2588,          1.18,  489
medium (unlab-6k),   5770.7,  1106,  21327,          3.31, 1742
large (unlab-60k),  57706.4,  9860, 219041,          7.84, 7439

Note: The smaller cuts are included in the larger ones.

Title: LibriLight: A Benchmark for Low-Resource Speech Recognition
Link: https://arxiv.org/pdf/1912.07875
"""

from ultravox.data import types

LIBRI_LIGHT_SMALL_CONFIG = types.DatasetConfig(
    name="libri_light_small",
    path="fixie-ai/libri_light_partial",
    subset="small",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=2588),
    ],
    user_template="<|audio|>",
    assistant_template="empty",
    transcript_template="empty",
)

LIBRI_LIGHT_MEDIUM_CONFIG = types.DatasetConfig(
    name="libri_light_medium",
    path="fixie-ai/libri_light_partial",
    subset="medium",
    splits=[types.DatasetSplitConfig(name="train", num_samples=18739)],
    user_template="<|audio|>",
    assistant_template="empty",
    transcript_template="empty",
)

LIBRI_LIGHT_LARGE_CONFIG = types.DatasetConfig(
    name="libri_light_large",
    # The fixie-ai/libri_light dataset is huge and it contains all three subsets,
    # however, we've had issues loading it which is why libri_light_partial was created.
    path="fixie-ai/libri_light",
    subset="large",
    splits=[types.DatasetSplitConfig(name="train", num_samples=197714)],
    user_template="<|audio|>",
    assistant_template="empty",
    transcript_template="empty",
)

configs = [
    LIBRI_LIGHT_SMALL_CONFIG,
    LIBRI_LIGHT_MEDIUM_CONFIG,
    LIBRI_LIGHT_LARGE_CONFIG,
]
