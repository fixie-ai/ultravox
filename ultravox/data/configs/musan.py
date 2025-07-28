from ultravox.data import types

# Configuration for the Musan dataset - a collection of audio samples for noise, music, and speech
MUSAN_CONFIG = types.DatasetConfig(
    name="musan",
    path="fixie-ai/musan-segments-v2",  # Replace with actual HF path if different
    subset="no_speech",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=8487),
        types.DatasetSplitConfig(name="validation", num_samples=472),
        types.DatasetSplitConfig(name="test", num_samples=472),
    ],
    assistant_template="((noise))",
    transcript_template="((noise))",
)

MUSAN_NOISE_CONFIG = types.DatasetConfig(
    name="musan-noise",
    base="musan",
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template=types.UNINTELLIGIBLE_TRAIN_INSTRUCTION,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "en"}),
)

MUSAN_TRANS_CONFIG = types.DatasetConfig(
    name="musan-transcription",
    base="musan",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    user_template_args={"transcript_language": "English"},
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "en"}),
)

MUSAN_NOISE_EXACT_MATCH_CONFIG = types.DatasetConfig(
    name="musan-noise-exact-match",
    base="musan",
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template=types.UNINTELLIGIBLE_TRAIN_INSTRUCTION,
    eval_config=types.EvalConfig(metric="exact_match"),
)

MUSAN_TRANS_EXACT_MATCH_CONFIG = types.DatasetConfig(
    name="musan-transcription-exact-match",
    base="musan",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    user_template_args={"transcript_language": "English"},
    eval_config=types.EvalConfig(metric="exact_match"),
)

MUSAN_NOISE_PARTIAL_MATCH_CONFIG = types.DatasetConfig(
    name="musan-noise-partial-match",
    base="musan",
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template=types.UNINTELLIGIBLE_TRAIN_INSTRUCTION,
    eval_config=types.EvalConfig(metric="partial_match"),
)

MUSAN_TRANS_PARTIAL_MATCH_CONFIG = types.DatasetConfig(
    name="musan-transcription-partial-match",
    base="musan",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    user_template_args={"transcript_language": "English"},
    eval_config=types.EvalConfig(metric="partial_match"),
)

# CommonVoice Musan Base Config
CV_MUSAN_BASE_CONFIG = types.DatasetConfig(
    name="commonvoice-musan",
    path="fixie-ai/common_voice_17_0-musan",
    transcript_template="{{sentence}}",
    assistant_template="{{sentence}}",
)

# English
CV_MUSAN_EN_CONFIG = types.DatasetConfig(
    name="commonvoice-musan-en",
    base="commonvoice-musan",
    subset="en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=19_999),
    ],
    transcript_template="{{text_proc.format_asr_text(sentence)}}",
    assistant_template="{{text_proc.format_asr_text(sentence)}}",
    user_template_args={"transcript_language": "English"},
)

CV_MUSAN_EN_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-musan-en-transcription",
    base="commonvoice-musan-en",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "en"}),
)

CV_MUSAN_EN_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-musan-en-continuation",
    base="commonvoice-musan-en",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

configs = [
    MUSAN_CONFIG,
    MUSAN_NOISE_CONFIG,
    MUSAN_TRANS_CONFIG,
    MUSAN_NOISE_EXACT_MATCH_CONFIG,
    MUSAN_TRANS_EXACT_MATCH_CONFIG,
    MUSAN_NOISE_PARTIAL_MATCH_CONFIG,
    MUSAN_TRANS_PARTIAL_MATCH_CONFIG,
    CV_MUSAN_BASE_CONFIG,
    CV_MUSAN_EN_CONFIG,
    CV_MUSAN_EN_TRANS_CONFIG,
    CV_MUSAN_EN_CONT_CONFIG,
]
