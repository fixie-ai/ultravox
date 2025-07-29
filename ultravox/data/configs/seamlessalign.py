from ultravox.data import types

# Base config for SeamlessAlign dataset
SEAMLESS_BASE_CONFIG = types.DatasetConfig(
    name="seamless",
    path="fixie-ai/SeamlessAlign",
    subset="indic2en",
    # audio_field="audio_filepath",
    transcript_template="{{transcription}}",
)

# Hindi
SEAMLESS_HI_CONFIG = types.DatasetConfig(
    name="seamless-hi",
    base="seamless",
    splits=[
        types.DatasetSplitConfig(
            name="hindi", num_samples=1_020_000, split=types.DatasetSplit.TRAIN
        ),
    ],
    user_template_args={"transcript_language": "Hindi"},
)

# Kannada
SEAMLESS_KN_CONFIG = types.DatasetConfig(
    name="seamless-kn",
    base="seamless",
    splits=[
        types.DatasetSplitConfig(
            name="kannada", num_samples=68_600, split=types.DatasetSplit.TRAIN
        ),
    ],
    user_template_args={"transcript_language": "Kannada"},
)

# Tamil
SEAMLESS_TA_CONFIG = types.DatasetConfig(
    name="seamless-ta",
    base="seamless",
    splits=[
        types.DatasetSplitConfig(
            name="tamil", num_samples=479_000, split=types.DatasetSplit.TRAIN
        ),
    ],
    user_template_args={"transcript_language": "Tamil"},
)

# Telugu
SEAMLESS_TE_CONFIG = types.DatasetConfig(
    name="seamless-te",
    base="seamless",
    splits=[
        types.DatasetSplitConfig(
            name="telugu", num_samples=329_000, split=types.DatasetSplit.TRAIN
        ),
    ],
    user_template_args={"transcript_language": "Telugu"},
)

# Urdu
SEAMLESS_UR_CONFIG = types.DatasetConfig(
    name="seamless-ur",
    base="seamless",
    splits=[
        types.DatasetSplitConfig(
            name="urdu", num_samples=1_110_000, split=types.DatasetSplit.TRAIN
        ),
    ],
    user_template_args={"transcript_language": "Urdu"},
)

# Transcription configs
SEAMLESS_HI_TRANS_CONFIG = types.DatasetConfig(
    name="seamless-hi-transcription",
    base="seamless-hi",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    user_template_args={"transcript_language": "Hindi"},
)

SEAMLESS_KN_TRANS_CONFIG = types.DatasetConfig(
    name="seamless-kn-transcription",
    base="seamless-kn",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    user_template_args={"transcript_language": "Kannada"},
)

SEAMLESS_TA_TRANS_CONFIG = types.DatasetConfig(
    name="seamless-ta-transcription",
    base="seamless-ta",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    user_template_args={"transcript_language": "Tamil"},
)

SEAMLESS_TE_TRANS_CONFIG = types.DatasetConfig(
    name="seamless-te-transcription",
    base="seamless-te",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    user_template_args={"transcript_language": "Telugu"},
)

SEAMLESS_UR_TRANS_CONFIG = types.DatasetConfig(
    name="seamless-ur-transcription",
    base="seamless-ur",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    user_template_args={"transcript_language": "Urdu"},
)

# Add these continuation configs after the transcription configs

SEAMLESS_HI_CONT_CONFIG = types.DatasetConfig(
    name="seamless-hi-continuation",
    base="seamless-hi",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
    user_template_args={"transcript_language": "Hindi"},
)

SEAMLESS_KN_CONT_CONFIG = types.DatasetConfig(
    name="seamless-kn-continuation",
    base="seamless-kn",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
    user_template_args={"transcript_language": "Kannada"},
)

SEAMLESS_TA_CONT_CONFIG = types.DatasetConfig(
    name="seamless-ta-continuation",
    base="seamless-ta",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
    user_template_args={"transcript_language": "Tamil"},
)

SEAMLESS_TE_CONT_CONFIG = types.DatasetConfig(
    name="seamless-te-continuation",
    base="seamless-te",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
    user_template_args={"transcript_language": "Telugu"},
)

SEAMLESS_UR_CONT_CONFIG = types.DatasetConfig(
    name="seamless-ur-continuation",
    base="seamless-ur",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
    user_template_args={"transcript_language": "Urdu"},
)

configs = [
    SEAMLESS_BASE_CONFIG,
    SEAMLESS_HI_CONFIG,
    SEAMLESS_KN_CONFIG,
    SEAMLESS_TA_CONFIG,
    SEAMLESS_TE_CONFIG,
    SEAMLESS_UR_CONFIG,
    SEAMLESS_HI_TRANS_CONFIG,
    SEAMLESS_KN_TRANS_CONFIG,
    SEAMLESS_TA_TRANS_CONFIG,
    SEAMLESS_TE_TRANS_CONFIG,
    SEAMLESS_UR_TRANS_CONFIG,
    SEAMLESS_HI_CONT_CONFIG,
    SEAMLESS_KN_CONT_CONFIG,
    SEAMLESS_TA_CONT_CONFIG,
    SEAMLESS_TE_CONT_CONFIG,
    SEAMLESS_UR_CONT_CONFIG,
]
