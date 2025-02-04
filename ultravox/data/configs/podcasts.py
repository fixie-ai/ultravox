"""
Close to 2T of podcast dialogue data. It has 23 chunks, each containing one podcast (channel).
All chunks (except for the last one) have 20480 samples, totaling 7430660 samples in total.
Most samples are 2 minutes long, but a few (final segments) are shorter.

Here, the last 2 chunks are used for validation and test, respectively.

Link: https://huggingface.co/datasets/YuKuanFu/Podcast_Dialogue
"""

from ultravox.data import types

PODCAST_DIALOGUE_CONFIG = types.DatasetConfig(
    name="podcast_dialogue",
    path="YuKuanFu/Podcast_Dialogue",
    splits=[
        # fmt: off
        types.DatasetSplitConfig(name="chunk_0", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_1", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_2", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_3", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_4", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_5", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_6", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_7", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_8", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_9", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_10", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_11", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_12", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_13", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_14", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_15", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_16", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_17", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_18", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_19", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_20", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_21", num_samples=20480, split=types.DatasetSplit.TRAIN),
        types.DatasetSplitConfig(name="chunk_22", num_samples=20480, split=types.DatasetSplit.VALIDATION),
        types.DatasetSplitConfig(name="chunk_23", num_samples=2893, split=types.DatasetSplit.TEST),
        # fmt: on
    ],
    assistant_template="empty",  # the dataset has no text annotation (only audio)
    transcript_template="empty",  # the dataset has no text annotation (only audio)
)

configs = [PODCAST_DIALOGUE_CONFIG]
