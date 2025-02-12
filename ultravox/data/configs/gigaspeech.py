from ultravox.data import types

GS_XL_CONFIG = types.DatasetConfig(
    name="gigaspeech-xl",
    path="fixie-ai/gigaspeech",
    subset="xl-empty-audio-removed",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=8_266_422),
    ],
    transcript_template="{{text_proc.format_asr_text(text)}}",
    assistant_template="{{text_proc.format_asr_text(text)}}",
)

GS_XL_TRANS_CONFIG = types.DatasetConfig(
    name="gigaspeech-xl-transcription",
    base="gigaspeech-xl",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "en"}),
)

GS_XL_CONT_CONFIG = types.DatasetConfig(
    name="gigaspeech-xl-continuation",
    base="gigaspeech-xl",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

configs = [GS_XL_CONFIG, GS_XL_TRANS_CONFIG, GS_XL_CONT_CONFIG]
