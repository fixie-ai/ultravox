from ultravox.data import types

# Base config for Shrutilipi dataset
SHRUTILIPI_BASE_CONFIG = types.DatasetConfig(
    name="shrutilipi",
    path="fixie-ai/Shrutilipi_a",
    # audio_field="audio_filepath",
    transcript_template="{{text}}",
)

# Assamese
SHRUTILIPI_AS_CONFIG = types.DatasetConfig(
    name="shrutilipi-as",
    base="shrutilipi",
    subset="assamese",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=41_800, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Assamese"},
)

# Bengali
SHRUTILIPI_BN_CONFIG = types.DatasetConfig(
    name="shrutilipi-bn",
    base="shrutilipi",
    subset="bengali",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=17_900, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Bengali"},
)

# Dogri
SHRUTILIPI_DO_CONFIG = types.DatasetConfig(
    name="shrutilipi-do",
    base="shrutilipi",
    subset="dogri",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=23_700, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Dogri"},
)

# Gujarati
SHRUTILIPI_GU_CONFIG = types.DatasetConfig(
    name="shrutilipi-gu",
    base="shrutilipi",
    subset="gujarati",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=138_000, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Gujarati"},
)

# Hindi
SHRUTILIPI_HI_CONFIG = types.DatasetConfig(
    name="shrutilipi-hi",
    base="shrutilipi",
    subset="hindi",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=735_000, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Hindi"},
)

# Kannada
SHRUTILIPI_KN_CONFIG = types.DatasetConfig(
    name="shrutilipi-kn",
    base="shrutilipi",
    subset="kannada",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=192_000, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Kannada"},
)

# Konkani
SHRUTILIPI_KO_CONFIG = types.DatasetConfig(
    name="shrutilipi-ko",
    base="shrutilipi",
    subset="konkani",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=12_200, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Konkani"},
)

# Maithili
SHRUTILIPI_MAI_CONFIG = types.DatasetConfig(
    name="shrutilipi-mai",
    base="shrutilipi",
    subset="maithili",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=22_200, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Maithili"},
)

# Malayalam
SHRUTILIPI_ML_CONFIG = types.DatasetConfig(
    name="shrutilipi-ml",
    base="shrutilipi",
    subset="malayalam",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=224_000, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Malayalam"},
)

# Marathi
SHRUTILIPI_MR_CONFIG = types.DatasetConfig(
    name="shrutilipi-mr",
    base="shrutilipi",
    subset="marathi",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=336_000, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Marathi"},
)

# Nepali
SHRUTILIPI_NE_CONFIG = types.DatasetConfig(
    name="shrutilipi-ne",
    base="shrutilipi",
    subset="nepali",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=32_000, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Nepali"},
)

# Odia
SHRUTILIPI_OR_CONFIG = types.DatasetConfig(
    name="shrutilipi-or",
    base="shrutilipi",
    subset="odia",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=127_000, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Odia"},
)

# Punjabi
SHRUTILIPI_PA_CONFIG = types.DatasetConfig(
    name="shrutilipi-pa",
    base="shrutilipi",
    subset="punjabi",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=21_100, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Punjabi"},
)

# Sanskrit
SHRUTILIPI_SA_CONFIG = types.DatasetConfig(
    name="shrutilipi-sa",
    base="shrutilipi",
    subset="sanskrit",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=14_200, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Sanskrit"},
)

# Tamil
SHRUTILIPI_TA_CONFIG = types.DatasetConfig(
    name="shrutilipi-ta",
    base="shrutilipi",
    subset="tamil",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=282_000, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Tamil"},
)

# Telugu
SHRUTILIPI_TE_CONFIG = types.DatasetConfig(
    name="shrutilipi-te",
    base="shrutilipi",
    subset="telugu",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=56_100, split=types.DatasetSplit.TRAIN
        )
    ],
    user_template_args={"transcript_language": "Telugu"},
)

# Transcription configs
SHRUTILIPI_AS_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-as-transcription",
    base="shrutilipi-as",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_BN_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-bn-transcription",
    base="shrutilipi-bn",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_DO_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-do-transcription",
    base="shrutilipi-do",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_GU_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-gu-transcription",
    base="shrutilipi-gu",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_HI_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-hi-transcription",
    base="shrutilipi-hi",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_KN_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-kn-transcription",
    base="shrutilipi-kn",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_KO_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-ko-transcription",
    base="shrutilipi-ko",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_MAI_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-mai-transcription",
    base="shrutilipi-mai",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_ML_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-ml-transcription",
    base="shrutilipi-ml",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_MR_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-mr-transcription",
    base="shrutilipi-mr",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_NE_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-ne-transcription",
    base="shrutilipi-ne",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_OR_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-or-transcription",
    base="shrutilipi-or",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_PA_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-pa-transcription",
    base="shrutilipi-pa",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_SA_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-sa-transcription",
    base="shrutilipi-sa",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_TA_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-ta-transcription",
    base="shrutilipi-ta",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

SHRUTILIPI_TE_TRANS_CONFIG = types.DatasetConfig(
    name="shrutilipi-te-transcription",
    base="shrutilipi-te",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-continuation",
    base="shrutilipi",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_AS_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-as-continuation",
    base="shrutilipi-as",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_BN_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-bn-continuation",
    base="shrutilipi-bn",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_DO_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-do-continuation",
    base="shrutilipi-do",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_GU_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-gu-continuation",
    base="shrutilipi-gu",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_HI_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-hi-continuation",
    base="shrutilipi-hi",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_KN_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-kn-continuation",
    base="shrutilipi-kn",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_KO_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-ko-continuation",
    base="shrutilipi-ko",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_MAI_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-mai-continuation",
    base="shrutilipi-mai",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_ML_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-ml-continuation",
    base="shrutilipi-ml",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_MR_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-mr-continuation",
    base="shrutilipi-mr",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_NE_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-ne-continuation",
    base="shrutilipi-ne",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_OR_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-or-continuation",
    base="shrutilipi-or",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_PA_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-pa-continuation",
    base="shrutilipi-pa",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_SA_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-sa-continuation",
    base="shrutilipi-sa",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_TA_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-ta-continuation",
    base="shrutilipi-ta",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

SHRUTILIPI_TE_CONT_CONFIG = types.DatasetConfig(
    name="shrutilipi-te-continuation",
    base="shrutilipi-te",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

configs = [
    SHRUTILIPI_BASE_CONFIG,
    SHRUTILIPI_AS_CONFIG,
    SHRUTILIPI_BN_CONFIG,
    SHRUTILIPI_DO_CONFIG,
    SHRUTILIPI_GU_CONFIG,
    SHRUTILIPI_HI_CONFIG,
    SHRUTILIPI_KN_CONFIG,
    SHRUTILIPI_KO_CONFIG,
    SHRUTILIPI_MAI_CONFIG,
    SHRUTILIPI_ML_CONFIG,
    SHRUTILIPI_MR_CONFIG,
    SHRUTILIPI_NE_CONFIG,
    SHRUTILIPI_OR_CONFIG,
    SHRUTILIPI_PA_CONFIG,
    SHRUTILIPI_SA_CONFIG,
    SHRUTILIPI_TA_CONFIG,
    SHRUTILIPI_TE_CONFIG,
    SHRUTILIPI_AS_TRANS_CONFIG,
    SHRUTILIPI_BN_TRANS_CONFIG,
    SHRUTILIPI_DO_TRANS_CONFIG,
    SHRUTILIPI_GU_TRANS_CONFIG,
    SHRUTILIPI_HI_TRANS_CONFIG,
    SHRUTILIPI_KN_TRANS_CONFIG,
    SHRUTILIPI_KO_TRANS_CONFIG,
    SHRUTILIPI_MAI_TRANS_CONFIG,
    SHRUTILIPI_ML_TRANS_CONFIG,
    SHRUTILIPI_MR_TRANS_CONFIG,
    SHRUTILIPI_NE_TRANS_CONFIG,
    SHRUTILIPI_OR_TRANS_CONFIG,
    SHRUTILIPI_PA_TRANS_CONFIG,
    SHRUTILIPI_SA_TRANS_CONFIG,
    SHRUTILIPI_TA_TRANS_CONFIG,
    SHRUTILIPI_TE_TRANS_CONFIG,
    SHRUTILIPI_AS_CONT_CONFIG,
    SHRUTILIPI_BN_CONT_CONFIG,
    SHRUTILIPI_DO_CONT_CONFIG,
    SHRUTILIPI_GU_CONT_CONFIG,
    SHRUTILIPI_HI_CONT_CONFIG,
    SHRUTILIPI_KN_CONT_CONFIG,
    SHRUTILIPI_KO_CONT_CONFIG,
    SHRUTILIPI_MAI_CONT_CONFIG,
    SHRUTILIPI_ML_CONT_CONFIG,
    SHRUTILIPI_MR_CONT_CONFIG,
    SHRUTILIPI_NE_CONT_CONFIG,
    SHRUTILIPI_OR_CONT_CONFIG,
    SHRUTILIPI_PA_CONT_CONFIG,
    SHRUTILIPI_SA_CONT_CONFIG,
    SHRUTILIPI_TA_CONT_CONFIG,
    SHRUTILIPI_TE_CONT_CONFIG,
]
