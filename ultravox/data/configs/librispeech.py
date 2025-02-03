from ultravox.data import types

LS_BASE_CONFIG = types.DatasetConfig(
    name="librispeech",
    path="fixie-ai/librispeech_asr",
    transcript_template="{{text_proc.format_asr_text(text)}}",
    assistant_template="{{text_proc.format_asr_text(text)}}",
)

LS_CLEAN_CONFIG = types.DatasetConfig(
    name="librispeech-clean",
    base="librispeech",
    subset="clean",
    splits=[
        types.DatasetSplitConfig(
            name="train.100", num_samples=28_539, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="train.360", num_samples=104_014, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=2620),
    ],
)

LS_OTHER_CONFIG = types.DatasetConfig(
    name="librispeech-other",
    base="librispeech",
    subset="other",
    splits=[
        types.DatasetSplitConfig(
            name="train.500", num_samples=148_688, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=2939),
    ],
)

LS_CLEAN_TRANS_CONFIG = types.DatasetConfig(
    name="librispeech-clean-transcription",
    base="librispeech-clean",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "en"}),
)

LS_OTHER_TRANS_CONFIG = types.DatasetConfig(
    name="librispeech-other-transcription",
    base="librispeech-other",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "en"}),
)

LS_CLEAN_CONT_CONFIG = types.DatasetConfig(
    name="librispeech-clean-continuation",
    base="librispeech-clean",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)
LS_OTHER_CONT_CONFIG = types.DatasetConfig(
    name="librispeech-other-continuation",
    base="librispeech-other",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

configs = [
    LS_BASE_CONFIG,
    LS_CLEAN_CONFIG,
    LS_OTHER_CONFIG,
    LS_CLEAN_TRANS_CONFIG,
    LS_OTHER_TRANS_CONFIG,
    LS_CLEAN_CONT_CONFIG,
    LS_OTHER_CONT_CONFIG,
]
