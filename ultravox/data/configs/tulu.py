from ultravox.data import types

# Convert original Tulu v3 dataset, used to train OLMo, to use F5-TTS generated audio (with voices cloned from fixie-ai/combined-voices)
# Original dataset: allenai/tulu-3-sft-olmo-2-mixture
# Only a subset of the samples are kept:
# - only kept samples that are English or Chinese (only languages supported by F5-TTS)
# - only kept samples where the initial user message is less than 200 characters
TULU_3_SFT_TTS_CONFIG = types.DatasetConfig(
    name="tulu-tts",
    path="fixie-ai/tulu-3-sft-olmo-2-mixture-en-zh-short-tts",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=221565),
        types.DatasetSplitConfig(name="validation", num_samples=2000),
        types.DatasetSplitConfig(name="test", num_samples=2000),
    ],
    transcript_template="{{messages[0]['content']}}",
    assistant_template="{{messages[1]['content']}}",
    eval_config=types.EvalConfig(metric="bleu"),
)


configs = [
    TULU_3_SFT_TTS_CONFIG,
]
