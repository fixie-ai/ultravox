from ultravox.data import types

PS_BASE_CONFIG = types.DatasetConfig(
    name="peoplespeech",
    path="fixie-ai/peoples_speech",
    subset="clean",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_501_271),
        types.DatasetSplitConfig(
            name="test", num_samples=34_898, split=types.DatasetSplit.VALIDATION
        ),
    ],
    assistant_template="{{text_proc.format_asr_text(text)}}",
    transcript_template="{{text_proc.format_asr_text(text)}}",
)

PS_TRANS_CONFIG = types.DatasetConfig(
    name="peoplespeech-clean-transcription",
    base="peoplespeech",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "en"}),
)

PS_CONT_CONFIG = types.DatasetConfig(
    name="peoplespeech-clean-continuation",
    base="peoplespeech",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

configs = [
    PS_BASE_CONFIG,
    PS_TRANS_CONFIG,
    PS_CONT_CONFIG,
]
