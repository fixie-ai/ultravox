from ultravox.data import types

# Base config for Kathbath dataset
KATHBATH_BASE_CONFIG = types.DatasetConfig(
    name="kathbath",
    path="fixie-ai/Kathbath",
    # audio_field="audio_filepath",
    transcript_template="{{text}}",
)

# Bengali
KATHBATH_BN_CONFIG = types.DatasetConfig(
    name="kathbath-bn",
    base="kathbath",
    subset="bengali",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=47_400, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_240, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Bengali"},
)

# Gujarati
KATHBATH_GU_CONFIG = types.DatasetConfig(
    name="kathbath-gu",
    base="kathbath",
    subset="gujarati",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=66_900, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_910, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Gujarati"},
)

# Hindi
KATHBATH_HI_CONFIG = types.DatasetConfig(
    name="kathbath-hi",
    base="kathbath",
    subset="hindi",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=91_800, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=3_150, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Hindi"},
)

# Kannada
KATHBATH_KN_CONFIG = types.DatasetConfig(
    name="kathbath-kn",
    base="kathbath",
    subset="kannada",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=67_400, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_060, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Kannada"},
)

# Malayalam
KATHBATH_ML_CONFIG = types.DatasetConfig(
    name="kathbath-ml",
    base="kathbath",
    subset="malayalam",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=45_100, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=1_770, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Malayalam"},
)

# Marathi
KATHBATH_MR_CONFIG = types.DatasetConfig(
    name="kathbath-mr",
    base="kathbath",
    subset="marathi",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=84_100, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_380, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Marathi"},
)

# Odia
KATHBATH_OR_CONFIG = types.DatasetConfig(
    name="kathbath-or",
    base="kathbath",
    subset="odia",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=47_700, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_400, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Odia"},
)

# Punjabi
KATHBATH_PA_CONFIG = types.DatasetConfig(
    name="kathbath-pa",
    base="kathbath",
    subset="punjabi",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=83_300, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=3_260, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Punjabi"},
)

# Sanskrit
KATHBATH_SA_CONFIG = types.DatasetConfig(
    name="kathbath-sa",
    base="kathbath",
    subset="sanskrit",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=26_800, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=1_180, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Sanskrit"},
)

# Tamil
KATHBATH_TA_CONFIG = types.DatasetConfig(
    name="kathbath-ta",
    base="kathbath",
    subset="tamil",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=95_800, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_770, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Tamil"},
)

# Telugu
KATHBATH_TE_CONFIG = types.DatasetConfig(
    name="kathbath-te",
    base="kathbath",
    subset="telugu",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=70_700, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_380, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Telugu"},
)

# Urdu
KATHBATH_UR_CONFIG = types.DatasetConfig(
    name="kathbath-ur",
    base="kathbath",
    subset="urdu",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=49_200, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=3_230, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Urdu"},
)

# Transcription configs
KATHBATH_BN_TRANS_CONFIG = types.DatasetConfig(
    name="kathbath-bn-transcription",
    base="kathbath-bn",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "bn"}),
)

KATHBATH_GU_TRANS_CONFIG = types.DatasetConfig(
    name="kathbath-gu-transcription",
    base="kathbath-gu",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "gu"}),
)

KATHBATH_HI_TRANS_CONFIG = types.DatasetConfig(
    name="kathbath-hi-transcription",
    base="kathbath-hi",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "hi"}),
)

KATHBATH_KN_TRANS_CONFIG = types.DatasetConfig(
    name="kathbath-kn-transcription",
    base="kathbath-kn",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "kn"}),
)

KATHBATH_ML_TRANS_CONFIG = types.DatasetConfig(
    name="kathbath-ml-transcription",
    base="kathbath-ml",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "ml"}),
)

KATHBATH_MR_TRANS_CONFIG = types.DatasetConfig(
    name="kathbath-mr-transcription",
    base="kathbath-mr",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "mr"}),
)

KATHBATH_OR_TRANS_CONFIG = types.DatasetConfig(
    name="kathbath-or-transcription",
    base="kathbath-or",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "or"}),
)

KATHBATH_PA_TRANS_CONFIG = types.DatasetConfig(
    name="kathbath-pa-transcription",
    base="kathbath-pa",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "pa"}),
)

KATHBATH_SA_TRANS_CONFIG = types.DatasetConfig(
    name="kathbath-sa-transcription",
    base="kathbath-sa",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "sa"}),
)

KATHBATH_TA_TRANS_CONFIG = types.DatasetConfig(
    name="kathbath-ta-transcription",
    base="kathbath-ta",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "ta"}),
)

KATHBATH_TE_TRANS_CONFIG = types.DatasetConfig(
    name="kathbath-te-transcription",
    base="kathbath-te",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "te"}),
)

KATHBATH_UR_TRANS_CONFIG = types.DatasetConfig(
    name="kathbath-ur-transcription",
    base="kathbath-ur",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "ur"}),
)

# Add these continuation configs after the transcription configs

KATHBATH_BN_CONT_CONFIG = types.DatasetConfig(
    name="kathbath-bn-continuation",
    base="kathbath-bn",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

KATHBATH_GU_CONT_CONFIG = types.DatasetConfig(
    name="kathbath-gu-continuation",
    base="kathbath-gu",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

KATHBATH_HI_CONT_CONFIG = types.DatasetConfig(
    name="kathbath-hi-continuation",
    base="kathbath-hi",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

KATHBATH_KN_CONT_CONFIG = types.DatasetConfig(
    name="kathbath-kn-continuation",
    base="kathbath-kn",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

KATHBATH_ML_CONT_CONFIG = types.DatasetConfig(
    name="kathbath-ml-continuation",
    base="kathbath-ml",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

KATHBATH_MR_CONT_CONFIG = types.DatasetConfig(
    name="kathbath-mr-continuation",
    base="kathbath-mr",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

KATHBATH_OR_CONT_CONFIG = types.DatasetConfig(
    name="kathbath-or-continuation",
    base="kathbath-or",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

KATHBATH_PA_CONT_CONFIG = types.DatasetConfig(
    name="kathbath-pa-continuation",
    base="kathbath-pa",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

KATHBATH_SA_CONT_CONFIG = types.DatasetConfig(
    name="kathbath-sa-continuation",
    base="kathbath-sa",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

KATHBATH_TA_CONT_CONFIG = types.DatasetConfig(
    name="kathbath-ta-continuation",
    base="kathbath-ta",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

KATHBATH_TE_CONT_CONFIG = types.DatasetConfig(
    name="kathbath-te-continuation",
    base="kathbath-te",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

KATHBATH_UR_CONT_CONFIG = types.DatasetConfig(
    name="kathbath-ur-continuation",
    base="kathbath-ur",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

configs = [
    KATHBATH_BASE_CONFIG,
    KATHBATH_BN_CONFIG,
    KATHBATH_GU_CONFIG,
    KATHBATH_HI_CONFIG,
    KATHBATH_KN_CONFIG,
    KATHBATH_ML_CONFIG,
    KATHBATH_MR_CONFIG,
    KATHBATH_OR_CONFIG,
    KATHBATH_PA_CONFIG,
    KATHBATH_SA_CONFIG,
    KATHBATH_TA_CONFIG,
    KATHBATH_TE_CONFIG,
    KATHBATH_UR_CONFIG,
    KATHBATH_BN_TRANS_CONFIG,
    KATHBATH_GU_TRANS_CONFIG,
    KATHBATH_HI_TRANS_CONFIG,
    KATHBATH_KN_TRANS_CONFIG,
    KATHBATH_ML_TRANS_CONFIG,
    KATHBATH_MR_TRANS_CONFIG,
    KATHBATH_OR_TRANS_CONFIG,
    KATHBATH_PA_TRANS_CONFIG,
    KATHBATH_SA_TRANS_CONFIG,
    KATHBATH_TA_TRANS_CONFIG,
    KATHBATH_TE_TRANS_CONFIG,
    KATHBATH_UR_TRANS_CONFIG,
    KATHBATH_BN_CONT_CONFIG,
    KATHBATH_GU_CONT_CONFIG,
    KATHBATH_HI_CONT_CONFIG,
    KATHBATH_KN_CONT_CONFIG,
    KATHBATH_ML_CONT_CONFIG,
    KATHBATH_MR_CONT_CONFIG,
    KATHBATH_OR_CONT_CONFIG,
    KATHBATH_PA_CONT_CONFIG,
    KATHBATH_SA_CONT_CONFIG,
    KATHBATH_TA_CONT_CONFIG,
    KATHBATH_TE_CONT_CONFIG,
    KATHBATH_UR_CONT_CONFIG,
]
