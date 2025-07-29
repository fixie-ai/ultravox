from ultravox.data import types

AMI_BASE_CONFIG = types.DatasetConfig(
    name="ami",
    path="edinburghcstr/ami",
    transcript_template="{{text}}",
    assistant_template="{{text}}",
    user_template_args={"transcript_language": "English"},
)

AMI_IHM_CONFIG = types.DatasetConfig(
    name="ami-ihm",
    base="ami",
    subset="ihm",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=29_100),
        types.DatasetSplitConfig(name="validation", num_samples=13_098),
        types.DatasetSplitConfig(name="test", num_samples=12_643),
    ],
    transcript_template="{{text_proc.format_asr_text(text)}}",
    assistant_template="{{text_proc.format_asr_text(text)}}",
)

AMI_IHM_TRANS_CONFIG = types.DatasetConfig(
    name="ami-ihm-transcription",
    base="ami-ihm",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "en"}),
)

configs = [
    AMI_BASE_CONFIG,
    AMI_IHM_CONFIG,
    AMI_IHM_TRANS_CONFIG,
]
