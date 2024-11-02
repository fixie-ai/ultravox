from ultravox.data import types

WS_BASE_CONFIG = types.DatasetConfig(
    name="wenetspeech",
    path="fixie-ai/wenetspeech",
    subset="L_fixed",
    splits=[types.DatasetSplitConfig(name="train", num_samples=14_621_415)],
    transcript_template="{{text}}",
)

WS_TRANS_CONFIG = types.DatasetConfig(
    name="wenetspeech-transcription",
    base="wenetspeech",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

WS_CONT_CONFIG = types.DatasetConfig(
    name="wenetspeech-continuation",
    base="wenetspeech",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

configs = [
    WS_BASE_CONFIG,
    WS_TRANS_CONFIG,
    WS_CONT_CONFIG,
]
