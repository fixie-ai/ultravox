from ultravox.data import types

# Base config for IndicVoices dataset
INDICVOICES_BASE_CONFIG = types.DatasetConfig(
    name="indicvoices",
    path="fixie-ai/IndicVoices_a",
    # audio_field="audio_filepath",
    transcript_template="{{text}}",
)

# Assamese
INDICVOICES_AS_CONFIG = types.DatasetConfig(
    name="indicvoices-as",
    base="indicvoices",
    subset="assamese",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=248_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_860, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Assamese"},
)

# Bengali
INDICVOICES_BN_CONFIG = types.DatasetConfig(
    name="indicvoices-bn",
    base="indicvoices",
    subset="bengali",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=212_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_660, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Bengali"},
)

# Bodo
INDICVOICES_BRX_CONFIG = types.DatasetConfig(
    name="indicvoices-brx",
    base="indicvoices",
    subset="bodo",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=264_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=3_030, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Bodo"},
)

# Dogri
INDICVOICES_DOI_CONFIG = types.DatasetConfig(
    name="indicvoices-doi",
    base="indicvoices",
    subset="dogri",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=103_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=1_330, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Dogri"},
)

# Gujarati
INDICVOICES_GU_CONFIG = types.DatasetConfig(
    name="indicvoices-gu",
    base="indicvoices",
    subset="gujarati",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=34_100, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=769, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Gujarati"},
)

# Hindi
INDICVOICES_HI_CONFIG = types.DatasetConfig(
    name="indicvoices-hi",
    base="indicvoices",
    subset="hindi",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=143_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_580, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Hindi"},
)

# Kannada
INDICVOICES_KN_CONFIG = types.DatasetConfig(
    name="indicvoices-kn",
    base="indicvoices",
    subset="kannada",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=132_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_160, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Kannada"},
)

# Kashmiri
INDICVOICES_KS_CONFIG = types.DatasetConfig(
    name="indicvoices-ks",
    base="indicvoices",
    subset="kashmiri",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=115_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=1_830, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Kashmiri"},
)

# Konkani
INDICVOICES_KOK_CONFIG = types.DatasetConfig(
    name="indicvoices-kok",
    base="indicvoices",
    subset="konkani",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=85_800, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=1_230, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Konkani"},
)

# Maithili
INDICVOICES_MAI_CONFIG = types.DatasetConfig(
    name="indicvoices-mai",
    base="indicvoices",
    subset="maithili",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=204_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_310, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Maithili"},
)

# Malayalam
INDICVOICES_ML_CONFIG = types.DatasetConfig(
    name="indicvoices-ml",
    base="indicvoices",
    subset="malayalam",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=232_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=3_360, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Malayalam"},
)

# Manipuri
INDICVOICES_MNI_CONFIG = types.DatasetConfig(
    name="indicvoices-mni",
    base="indicvoices",
    subset="manipuri",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=56_700, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=1_010, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Manipuri"},
)

# Marathi
INDICVOICES_MR_CONFIG = types.DatasetConfig(
    name="indicvoices-mr",
    base="indicvoices",
    subset="marathi",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=144_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=1_940, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Marathi"},
)

# Nepali
INDICVOICES_NE_CONFIG = types.DatasetConfig(
    name="indicvoices-ne",
    base="indicvoices",
    subset="nepali",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=206_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_430, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Nepali"},
)

# Odia
INDICVOICES_OR_CONFIG = types.DatasetConfig(
    name="indicvoices-or",
    base="indicvoices",
    subset="odia",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=142_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=1_960, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Odia"},
)

# Punjabi
INDICVOICES_PA_CONFIG = types.DatasetConfig(
    name="indicvoices-pa",
    base="indicvoices",
    subset="punjabi",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=129_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=1_680, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Punjabi"},
)

# Sanskrit
INDICVOICES_SA_CONFIG = types.DatasetConfig(
    name="indicvoices-sa",
    base="indicvoices",
    subset="sanskrit",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=96_400, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=1_380, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Sanskrit"},
)

# Santali
INDICVOICES_SAT_CONFIG = types.DatasetConfig(
    name="indicvoices-sat",
    base="indicvoices",
    subset="santali",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=167_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_050, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Santali"},
)

# Sindhi
INDICVOICES_SD_CONFIG = types.DatasetConfig(
    name="indicvoices-sd",
    base="indicvoices",
    subset="sindhi",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=73_800, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=1_340, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Sindhi"},
)

# Tamil
INDICVOICES_TA_CONFIG = types.DatasetConfig(
    name="indicvoices-ta",
    base="indicvoices",
    subset="tamil",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=263_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=3_570, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Tamil"},
)

# Telugu
INDICVOICES_TE_CONFIG = types.DatasetConfig(
    name="indicvoices-te",
    base="indicvoices",
    subset="telugu",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=185_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_310, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Telugu"},
)

# Urdu
INDICVOICES_UR_CONFIG = types.DatasetConfig(
    name="indicvoices-ur",
    base="indicvoices",
    subset="urdu",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=162_000, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(
            name="valid", num_samples=2_340, split=types.DatasetSplit.VALIDATION
        ),
    ],
    user_template_args={"transcript_language": "Urdu"},
)

# Transcription configs
INDICVOICES_AS_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-as-transcription",
    base="indicvoices-as",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "as"}),
)

INDICVOICES_BN_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-bn-transcription",
    base="indicvoices-bn",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "bn"}),
)

INDICVOICES_BRX_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-brx-transcription",
    base="indicvoices-brx",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "brx"}),
)

INDICVOICES_DOI_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-doi-transcription",
    base="indicvoices-doi",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "doi"}),
)

INDICVOICES_GU_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-gu-transcription",
    base="indicvoices-gu",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "gu"}),
)

INDICVOICES_HI_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-hi-transcription",
    base="indicvoices-hi",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "hi"}),
)

INDICVOICES_KN_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-kn-transcription",
    base="indicvoices-kn",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "kn"}),
)

INDICVOICES_KS_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-ks-transcription",
    base="indicvoices-ks",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "ks"}),
)

INDICVOICES_KOK_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-kok-transcription",
    base="indicvoices-kok",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "kok"}),
)

INDICVOICES_MAI_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-mai-transcription",
    base="indicvoices-mai",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "mai"}),
)

INDICVOICES_ML_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-ml-transcription",
    base="indicvoices-ml",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "ml"}),
)

INDICVOICES_MNI_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-mni-transcription",
    base="indicvoices-mni",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "mni"}),
)

INDICVOICES_MR_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-mr-transcription",
    base="indicvoices-mr",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "mr"}),
)

INDICVOICES_NE_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-ne-transcription",
    base="indicvoices-ne",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "ne"}),
)

INDICVOICES_OR_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-or-transcription",
    base="indicvoices-or",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "or"}),
)

INDICVOICES_PA_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-pa-transcription",
    base="indicvoices-pa",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "pa"}),
)

INDICVOICES_SA_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-sa-transcription",
    base="indicvoices-sa",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "sa"}),
)

INDICVOICES_SAT_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-sat-transcription",
    base="indicvoices-sat",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "sat"}),
)

INDICVOICES_SD_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-sd-transcription",
    base="indicvoices-sd",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "sd"}),
)

INDICVOICES_TA_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-ta-transcription",
    base="indicvoices-ta",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "ta"}),
)

INDICVOICES_TE_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-te-transcription",
    base="indicvoices-te",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "te"}),
)

INDICVOICES_UR_TRANS_CONFIG = types.DatasetConfig(
    name="indicvoices-ur-transcription",
    base="indicvoices-ur",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "ur"}),
)

# Add these continuation configs after the transcription configs

INDICVOICES_AS_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-as-continuation",
    base="indicvoices-as",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_BN_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-bn-continuation",
    base="indicvoices-bn",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_BRX_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-brx-continuation",
    base="indicvoices-brx",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_DOI_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-doi-continuation",
    base="indicvoices-doi",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_GU_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-gu-continuation",
    base="indicvoices-gu",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_HI_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-hi-continuation",
    base="indicvoices-hi",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_KN_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-kn-continuation",
    base="indicvoices-kn",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_KS_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-ks-continuation",
    base="indicvoices-ks",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_KOK_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-kok-continuation",
    base="indicvoices-kok",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_MAI_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-mai-continuation",
    base="indicvoices-mai",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_ML_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-ml-continuation",
    base="indicvoices-ml",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_MNI_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-mni-continuation",
    base="indicvoices-mni",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_MR_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-mr-continuation",
    base="indicvoices-mr",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_NE_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-ne-continuation",
    base="indicvoices-ne",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_OR_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-or-continuation",
    base="indicvoices-or",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_PA_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-pa-continuation",
    base="indicvoices-pa",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_SA_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-sa-continuation",
    base="indicvoices-sa",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_SAT_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-sat-continuation",
    base="indicvoices-sat",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_SD_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-sd-continuation",
    base="indicvoices-sd",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_TA_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-ta-continuation",
    base="indicvoices-ta",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_TE_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-te-continuation",
    base="indicvoices-te",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

INDICVOICES_UR_CONT_CONFIG = types.DatasetConfig(
    name="indicvoices-ur-continuation",
    base="indicvoices-ur",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

configs = [
    INDICVOICES_BASE_CONFIG,
    INDICVOICES_AS_CONFIG,
    INDICVOICES_BN_CONFIG,
    INDICVOICES_BRX_CONFIG,
    INDICVOICES_DOI_CONFIG,
    INDICVOICES_GU_CONFIG,
    INDICVOICES_HI_CONFIG,
    INDICVOICES_KN_CONFIG,
    INDICVOICES_KS_CONFIG,
    INDICVOICES_KOK_CONFIG,
    INDICVOICES_MAI_CONFIG,
    INDICVOICES_ML_CONFIG,
    INDICVOICES_MNI_CONFIG,
    INDICVOICES_MR_CONFIG,
    INDICVOICES_NE_CONFIG,
    INDICVOICES_OR_CONFIG,
    INDICVOICES_PA_CONFIG,
    INDICVOICES_SA_CONFIG,
    INDICVOICES_SAT_CONFIG,
    INDICVOICES_SD_CONFIG,
    INDICVOICES_TA_CONFIG,
    INDICVOICES_TE_CONFIG,
    INDICVOICES_UR_CONFIG,
    INDICVOICES_AS_TRANS_CONFIG,
    INDICVOICES_BN_TRANS_CONFIG,
    INDICVOICES_BRX_TRANS_CONFIG,
    INDICVOICES_DOI_TRANS_CONFIG,
    INDICVOICES_GU_TRANS_CONFIG,
    INDICVOICES_HI_TRANS_CONFIG,
    INDICVOICES_KN_TRANS_CONFIG,
    INDICVOICES_KS_TRANS_CONFIG,
    INDICVOICES_KOK_TRANS_CONFIG,
    INDICVOICES_MAI_TRANS_CONFIG,
    INDICVOICES_ML_TRANS_CONFIG,
    INDICVOICES_MNI_TRANS_CONFIG,
    INDICVOICES_MR_TRANS_CONFIG,
    INDICVOICES_NE_TRANS_CONFIG,
    INDICVOICES_OR_TRANS_CONFIG,
    INDICVOICES_PA_TRANS_CONFIG,
    INDICVOICES_SA_TRANS_CONFIG,
    INDICVOICES_SAT_TRANS_CONFIG,
    INDICVOICES_SD_TRANS_CONFIG,
    INDICVOICES_TA_TRANS_CONFIG,
    INDICVOICES_TE_TRANS_CONFIG,
    INDICVOICES_UR_TRANS_CONFIG,
    INDICVOICES_AS_CONT_CONFIG,
    INDICVOICES_BN_CONT_CONFIG,
    INDICVOICES_BRX_CONT_CONFIG,
    INDICVOICES_DOI_CONT_CONFIG,
    INDICVOICES_GU_CONT_CONFIG,
    INDICVOICES_HI_CONT_CONFIG,
    INDICVOICES_KN_CONT_CONFIG,
    INDICVOICES_KS_CONT_CONFIG,
    INDICVOICES_KOK_CONT_CONFIG,
    INDICVOICES_MAI_CONT_CONFIG,
    INDICVOICES_ML_CONT_CONFIG,
    INDICVOICES_MNI_CONT_CONFIG,
    INDICVOICES_MR_CONT_CONFIG,
    INDICVOICES_NE_CONT_CONFIG,
    INDICVOICES_OR_CONT_CONFIG,
    INDICVOICES_PA_CONT_CONFIG,
    INDICVOICES_SA_CONT_CONFIG,
    INDICVOICES_SAT_CONT_CONFIG,
    INDICVOICES_SD_CONT_CONFIG,
    INDICVOICES_TA_CONT_CONFIG,
    INDICVOICES_TE_CONT_CONFIG,
    INDICVOICES_UR_CONT_CONFIG,
]
