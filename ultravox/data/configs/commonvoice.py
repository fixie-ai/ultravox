from ultravox.data import types

# "Other" is the split that has not yet been reviewed to be validated/invalidated
# "Validated" split is a superset of "train, test, validation."
# We include the "validation" split from some low-resource languages for training.
# In the future, we will create new training splits from the "validated" split.
CV_BASE_CONFIG = types.DatasetConfig(
    name="commonvoice",
    path="fixie-ai/common_voice_17_0",
    transcript_template="{{sentence}}",
    assistant_template="{{sentence}}",
)

# English
CV_EN_CONFIG = types.DatasetConfig(
    name="commonvoice-en",
    base="commonvoice",
    subset="en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_101_170),
        types.DatasetSplitConfig(name="validation", num_samples=16_393),
        types.DatasetSplitConfig(name="test", num_samples=16_393),
    ],
    transcript_template="{{text_proc.format_asr_text(sentence)}}",
    assistant_template="{{text_proc.format_asr_text(sentence)}}",
)

# Arabic
CV_AR_CONFIG = types.DatasetConfig(
    name="commonvoice-ar",
    base="commonvoice",
    subset="ar",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=28_369),
        types.DatasetSplitConfig(
            name="validation", num_samples=10_470, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=10_480),
    ],
)

# German
CV_DE_CONFIG = types.DatasetConfig(
    name="commonvoice-de",
    base="commonvoice",
    subset="de",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=589_100),
        types.DatasetSplitConfig(name="validation", num_samples=16_183),
        types.DatasetSplitConfig(name="test", num_samples=16_183),
    ],
)

# Spanish
CV_ES_CONFIG = types.DatasetConfig(
    name="commonvoice-es",
    base="commonvoice",
    subset="es",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=336_846),
        types.DatasetSplitConfig(name="validation", num_samples=15_857),
        types.DatasetSplitConfig(name="test", num_samples=15_857),
    ],
)

# French
CV_FR_CONFIG = types.DatasetConfig(
    name="commonvoice-fr",
    base="commonvoice",
    subset="fr",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=558_054),
        types.DatasetSplitConfig(name="validation", num_samples=16_159),
        types.DatasetSplitConfig(name="test", num_samples=16_159),
    ],
)

# Italian
CV_IT_CONFIG = types.DatasetConfig(
    name="commonvoice-it",
    base="commonvoice",
    subset="it",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=169_771),
        types.DatasetSplitConfig(name="validation", num_samples=15_149),
        types.DatasetSplitConfig(name="test", num_samples=15_155),
    ],
)

# Japanese
CV_JA_CONFIG = types.DatasetConfig(
    name="commonvoice-ja",
    base="commonvoice",
    subset="ja",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=10_039),
        types.DatasetSplitConfig(
            name="validation", num_samples=6_261, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=6_261),
    ],
)

# Portuguese
CV_PT_CONFIG = types.DatasetConfig(
    name="commonvoice-pt",
    base="commonvoice",
    subset="pt",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=21_968),
        types.DatasetSplitConfig(
            name="validation", num_samples=9_464, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=9_467),
    ],
)

# Russian
CV_RU_CONFIG = types.DatasetConfig(
    name="commonvoice-ru",
    base="commonvoice",
    subset="ru",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=26_377),
        types.DatasetSplitConfig(
            name="validation", num_samples=10_203, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=10_203),
    ],
)

# Hindi
CV_HI_CONFIG = types.DatasetConfig(
    name="commonvoice-hi",
    base="commonvoice",
    subset="hi",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=4_690),
        types.DatasetSplitConfig(
            name="validation", num_samples=2_430, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=3_154),
    ],
)

# Turkish
CV_TR_CONFIG = types.DatasetConfig(
    name="commonvoice-tr",
    base="commonvoice",
    subset="tr",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=35_147),
        types.DatasetSplitConfig(
            name="validation", num_samples=11_258, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=11_290),
    ],
)

# Swedish
CV_SV_CONFIG = types.DatasetConfig(
    name="commonvoice-sv",
    base="commonvoice",
    subset="sv-SE",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=7_744),
        types.DatasetSplitConfig(
            name="validation", num_samples=5_210, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=5_259),
    ],
)

# Ukrainian
CV_UK_CONFIG = types.DatasetConfig(
    name="commonvoice-uk",
    base="commonvoice",
    subset="uk",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=25_137),
        types.DatasetSplitConfig(
            name="validation", num_samples=10_007, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=10_011),
    ],
)

# Swahili
CV_SW_CONFIG = types.DatasetConfig(
    name="commonvoice-sw",
    base="commonvoice",
    subset="sw",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=46_494),
        types.DatasetSplitConfig(
            name="validation", num_samples=12_251, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=12_253),
    ],
)

# Persian
CV_FA_CONFIG = types.DatasetConfig(
    name="commonvoice-fa",
    base="commonvoice",
    subset="fa",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=28_893),
        types.DatasetSplitConfig(
            name="validation", num_samples=10_559, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=10_559),
    ],
)

# Thai
CV_TH_CONFIG = types.DatasetConfig(
    name="commonvoice-th",
    base="commonvoice",
    subset="th",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=32_823),
        types.DatasetSplitConfig(
            name="validation", num_samples=11_042, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=11_042),
    ],
)

# Belarusian
CV_BE_CONFIG = types.DatasetConfig(
    name="commonvoice-be",
    base="commonvoice",
    subset="be",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=347_637),
        types.DatasetSplitConfig(
            name="validation", num_samples=15_880, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=15_878),
    ],
)

# Tamil
CV_TA_CONFIG = types.DatasetConfig(
    name="commonvoice-ta",
    base="commonvoice",
    subset="ta",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=45_587),
        types.DatasetSplitConfig(
            name="validation", num_samples=12_095, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=12_074),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=93_989, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=135_391),
    ],
)

# Czech
CV_CS_CONFIG = types.DatasetConfig(
    name="commonvoice-cs",
    base="commonvoice",
    subset="cs",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=20_144),
        types.DatasetSplitConfig(
            name="validation", num_samples=9_009, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=9_067),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=148_316, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=61_391),
    ],
)

# Latvian
CV_LV_CONFIG = types.DatasetConfig(
    name="commonvoice-lv",
    base="commonvoice",
    subset="lv",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=11_364),
        types.DatasetSplitConfig(
            name="validation", num_samples=6_752, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=6_752),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=32_248, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=171_652),
    ],
)

# Georgian
CV_KA_CONFIG = types.DatasetConfig(
    name="commonvoice-ka",
    base="commonvoice",
    subset="ka",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=52_321),
        types.DatasetSplitConfig(
            name="validation", num_samples=12_545, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=12_618),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=48_563, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=97_230),
    ],
)

# Urdu
CV_UR_CONFIG = types.DatasetConfig(
    name="commonvoice-ur",
    base="commonvoice",
    subset="ur",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=5_368),
        types.DatasetSplitConfig(
            name="validation", num_samples=4_057, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=4_056),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=135_861, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=53_858),
    ],
)

# Polish
CV_PL_CONFIG = types.DatasetConfig(
    name="commonvoice-pl",
    base="commonvoice",
    subset="pl",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=20_729),
        types.DatasetSplitConfig(
            name="validation", num_samples=9_230, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=9_230),
        # types.DatasetSplitConfig(name="validated", num_samples=132_661),
    ],
)

# Hungarian
CV_HU_CONFIG = types.DatasetConfig(
    name="commonvoice-hu",
    base="commonvoice",
    subset="hu",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=37_140),
        types.DatasetSplitConfig(
            name="validation", num_samples=11_350, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=11_435),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=49_019, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=60_358),
    ],
)

# Dutch
CV_NL_CONFIG = types.DatasetConfig(
    name="commonvoice-nl",
    base="commonvoice",
    subset="nl",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=34_898),
        types.DatasetSplitConfig(
            name="validation", num_samples=11_252, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=11_266),
        # types.DatasetSplitConfig(name="validated", num_samples=90_449),
    ],
)

# Galician
CV_GL_CONFIG = types.DatasetConfig(
    name="commonvoice-gl",
    base="commonvoice",
    subset="gl",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=25_159),
        types.DatasetSplitConfig(
            name="validation", num_samples=9_982, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=9_990),
        # types.DatasetSplitConfig(name="validated", num_samples=45_780),
    ],
)

# Welsh
CV_CY_CONFIG = types.DatasetConfig(
    name="commonvoice-cy",
    base="commonvoice",
    subset="cy",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=7_960),
        types.DatasetSplitConfig(
            name="validation", num_samples=5_371, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=5_379),
        # types.DatasetSplitConfig(name="validated", num_samples=90_369),
    ],
)

# Romanian
CV_RO_CONFIG = types.DatasetConfig(
    name="commonvoice-ro",
    base="commonvoice",
    subset="ro",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=5_141),
        types.DatasetSplitConfig(
            name="validation", num_samples=3_881, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=3_896),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=23_087, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=17_737),
    ],
)

# Estonian
CV_ET_CONFIG = types.DatasetConfig(
    name="commonvoice-et",
    base="commonvoice",
    subset="et",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=3_157),
        types.DatasetSplitConfig(
            name="validation", num_samples=2_653, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=2_653),
        # types.DatasetSplitConfig(name="validated", num_samples=24_381),
    ],
)

# Breton
CV_BR_CONFIG = types.DatasetConfig(
    name="commonvoice-br",
    base="commonvoice",
    subset="br",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=2_663),
        types.DatasetSplitConfig(
            name="validation", num_samples=2_253, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=2_212),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=8_037, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=21_007),
    ],
)

# Lithuanian
CV_LT_CONFIG = types.DatasetConfig(
    name="commonvoice-lt",
    base="commonvoice",
    subset="lt",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=7_253),
        types.DatasetSplitConfig(
            name="validation", num_samples=4_436, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=4_753),
        # types.DatasetSplitConfig(name="validated", num_samples=16_643),
    ],
)

# Greek
CV_EL_CONFIG = types.DatasetConfig(
    name="commonvoice-el",
    base="commonvoice",
    subset="el",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_920),
        types.DatasetSplitConfig(
            name="validation", num_samples=1_700, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=1_701),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=10_330, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=16_199),
    ],
)

# Slovak
CV_SK_CONFIG = types.DatasetConfig(
    name="commonvoice-sk",
    base="commonvoice",
    subset="sk",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=3_258),
        types.DatasetSplitConfig(
            name="validation", num_samples=2_588, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=2_647),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=3_392, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=19_513),
    ],
)

# Bulgarian
CV_BG_CONFIG = types.DatasetConfig(
    name="commonvoice-bg",
    base="commonvoice",
    subset="bg",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=4_849),
        types.DatasetSplitConfig(
            name="validation", num_samples=2_766, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=3_201),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=2_087, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=10_832),
    ],
)

# Macedonian
CV_MK_CONFIG = types.DatasetConfig(
    name="commonvoice-mk",
    base="commonvoice",
    subset="mk",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_686),
        types.DatasetSplitConfig(
            name="validation", num_samples=1_289, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=1_097),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=12_289, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=6_512),
    ],
)

# Finnish
CV_FI_CONFIG = types.DatasetConfig(
    name="commonvoice-fi",
    base="commonvoice",
    subset="fi",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=2_076),
        types.DatasetSplitConfig(
            name="validation", num_samples=1_770, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=1_763),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=6_202, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=10_447),
    ],
)

# Marathi
CV_MR_CONFIG = types.DatasetConfig(
    name="commonvoice-mr",
    base="commonvoice",
    subset="mr",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=2_215),
        types.DatasetSplitConfig(
            name="validation", num_samples=1_780, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=1_751),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=2_805, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=10_901),
    ],
)

# Mongolian
CV_MN_CONFIG = types.DatasetConfig(
    name="commonvoice-mn",
    base="commonvoice",
    subset="mn",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=2_175),
        types.DatasetSplitConfig(
            name="validation", num_samples=1_870, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=1_896),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=5_773, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=8_757),
    ],
)

# Vietnamese
CV_VI_CONFIG = types.DatasetConfig(
    name="commonvoice-vi",
    base="commonvoice",
    subset="vi",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=2_298),
        types.DatasetSplitConfig(
            name="validation", num_samples=641, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=1_274),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=11_533, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=5_135),
    ],
)

# Danish
CV_DA_CONFIG = types.DatasetConfig(
    name="commonvoice-da",
    base="commonvoice",
    subset="da",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=3_484),
        types.DatasetSplitConfig(
            name="validation", num_samples=2_105, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=2_530),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=396, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=10_225),
    ],
)

# Slovenian
CV_SL_CONFIG = types.DatasetConfig(
    name="commonvoice-sl",
    base="commonvoice",
    subset="sl",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_388),
        types.DatasetSplitConfig(
            name="validation", num_samples=1_232, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=1_242),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=3_145, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=10_819),
    ],
)

# Serbian
CV_SR_CONFIG = types.DatasetConfig(
    name="commonvoice-sr",
    base="commonvoice",
    subset="sr",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_879),
        types.DatasetSplitConfig(
            name="validation", num_samples=1_583, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=1_539),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=1_781, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=5_970),
    ],
)

# Malayalam
CV_ML_CONFIG = types.DatasetConfig(
    name="commonvoice-ml",
    base="commonvoice",
    subset="ml",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_259),
        types.DatasetSplitConfig(
            name="validation", num_samples=764, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=710),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=5_621, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=2_984),
    ],
)

# Occitan
CV_OC_CONFIG = types.DatasetConfig(
    name="commonvoice-oc",
    base="commonvoice",
    subset="oc",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=271),
        types.DatasetSplitConfig(
            name="validation", num_samples=260, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=254),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=7632, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=1668),
    ],
)

# Bengali
CV_BN_CONFIG = types.DatasetConfig(
    name="commonvoice-bn",
    base="commonvoice",
    subset="bn",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=21_228),
        types.DatasetSplitConfig(
            name="validation", num_samples=9_327, split=types.DatasetSplit.TRAIN
        ),
        types.DatasetSplitConfig(name="test", num_samples=9_327),
        # types.DatasetSplitConfig(
        #     name="other", num_samples=99_756, split=types.DatasetSplit.TRAIN
        # ),
        # types.DatasetSplitConfig(name="validated", num_samples=44_121),
    ],
)

CV_EN_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-en-transcription",
    base="commonvoice-en",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "en"}),
)
CV_AR_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-ar-transcription",
    base="commonvoice-ar",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "ar"}),
)
CV_DE_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-de-transcription",
    base="commonvoice-de",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_ES_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-es-transcription",
    base="commonvoice-es",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_FR_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-fr-transcription",
    base="commonvoice-fr",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_IT_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-it-transcription",
    base="commonvoice-it",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_JA_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-ja-transcription",
    base="commonvoice-ja",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "ja"}),
)
CV_PT_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-pt-transcription",
    base="commonvoice-pt",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_RU_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-ru-transcription",
    base="commonvoice-ru",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_HI_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-hi-transcription",
    base="commonvoice-hi",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_TR_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-tr-transcription",
    base="commonvoice-tr",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_SV_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-sv-transcription",
    base="commonvoice-sv",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_UK_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-uk-transcription",
    base="commonvoice-uk",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_SW_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-sw-transcription",
    base="commonvoice-sw",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_FA_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-fa-transcription",
    base="commonvoice-fa",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_TH_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-th-transcription",
    base="commonvoice-th",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer", args={"lang_id": "th"}),
)
CV_BE_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-be-transcription",
    base="commonvoice-be",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_TA_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-ta-transcription",
    base="commonvoice-ta",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_CS_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-cs-transcription",
    base="commonvoice-cs",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_LV_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-lv-transcription",
    base="commonvoice-lv",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_KA_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-ka-transcription",
    base="commonvoice-ka",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_UR_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-ur-transcription",
    base="commonvoice-ur",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_PL_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-pl-transcription",
    base="commonvoice-pl",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_HU_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-hu-transcription",
    base="commonvoice-hu",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_NL_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-nl-transcription",
    base="commonvoice-nl",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_GL_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-gl-transcription",
    base="commonvoice-gl",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_CY_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-cy-transcription",
    base="commonvoice-cy",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_RO_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-ro-transcription",
    base="commonvoice-ro",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_ET_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-et-transcription",
    base="commonvoice-et",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_BR_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-br-transcription",
    base="commonvoice-br",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_LT_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-lt-transcription",
    base="commonvoice-lt",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_EL_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-el-transcription",
    base="commonvoice-el",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_SK_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-sk-transcription",
    base="commonvoice-sk",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_BG_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-bg-transcription",
    base="commonvoice-bg",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_MK_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-mk-transcription",
    base="commonvoice-mk",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_FI_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-fi-transcription",
    base="commonvoice-fi",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_MR_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-mr-transcription",
    base="commonvoice-mr",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_MN_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-mn-transcription",
    base="commonvoice-mn",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_VI_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-vi-transcription",
    base="commonvoice-vi",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_DA_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-da-transcription",
    base="commonvoice-da",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_SL_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-sl-transcription",
    base="commonvoice-sl",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_SR_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-sr-transcription",
    base="commonvoice-sr",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_ML_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-ml-transcription",
    base="commonvoice-ml",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_OC_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-oc-transcription",
    base="commonvoice-oc",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)
CV_BN_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-bn-transcription",
    base="commonvoice-bn",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
    eval_config=types.EvalConfig(metric="wer"),
)

CV_EN_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-en-continuation",
    base="commonvoice-en",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_AR_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-ar-continuation",
    base="commonvoice-ar",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_DE_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-de-continuation",
    base="commonvoice-de",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_ES_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-es-continuation",
    base="commonvoice-es",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_FR_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-fr-continuation",
    base="commonvoice-fr",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_IT_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-it-continuation",
    base="commonvoice-it",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_JA_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-ja-continuation",
    base="commonvoice-ja",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_PT_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-pt-continuation",
    base="commonvoice-pt",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_RU_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-ru-continuation",
    base="commonvoice-ru",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_HI_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-hi-continuation",
    base="commonvoice-hi",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_TR_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-tr-continuation",
    base="commonvoice-tr",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_SV_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-sv-continuation",
    base="commonvoice-sv",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_UK_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-uk-continuation",
    base="commonvoice-uk",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_SW_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-sw-continuation",
    base="commonvoice-sw",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_FA_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-fa-continuation",
    base="commonvoice-fa",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_TH_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-th-continuation",
    base="commonvoice-th",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_BE_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-be-continuation",
    base="commonvoice-be",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_TA_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-ta-continuation",
    base="commonvoice-ta",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_CS_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-cs-continuation",
    base="commonvoice-cs",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_LV_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-lv-continuation",
    base="commonvoice-lv",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_KA_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-ka-continuation",
    base="commonvoice-ka",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_UR_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-ur-continuation",
    base="commonvoice-ur",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_PL_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-pl-continuation",
    base="commonvoice-pl",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_HU_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-hu-continuation",
    base="commonvoice-hu",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_NL_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-nl-continuation",
    base="commonvoice-nl",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_GL_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-gl-continuation",
    base="commonvoice-gl",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_CY_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-cy-continuation",
    base="commonvoice-cy",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_RO_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-ro-continuation",
    base="commonvoice-ro",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_ET_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-et-continuation",
    base="commonvoice-et",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_BR_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-br-continuation",
    base="commonvoice-br",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_LT_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-lt-continuation",
    base="commonvoice-lt",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_EL_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-el-continuation",
    base="commonvoice-el",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_SK_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-sk-continuation",
    base="commonvoice-sk",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_BG_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-bg-continuation",
    base="commonvoice-bg",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_MK_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-mk-continuation",
    base="commonvoice-mk",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_FI_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-fi-continuation",
    base="commonvoice-fi",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_MR_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-mr-continuation",
    base="commonvoice-mr",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_MN_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-mn-continuation",
    base="commonvoice-mn",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_VI_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-vi-continuation",
    base="commonvoice-vi",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_DA_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-da-continuation",
    base="commonvoice-da",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_SL_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-sl-continuation",
    base="commonvoice-sl",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_SR_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-sr-continuation",
    base="commonvoice-sr",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_ML_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-ml-continuation",
    base="commonvoice-ml",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_OC_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-oc-continuation",
    base="commonvoice-oc",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

CV_BN_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-bn-continuation",
    base="commonvoice-bn",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

configs = [
    CV_BASE_CONFIG,
    CV_EN_CONFIG,
    CV_AR_CONFIG,
    CV_DE_CONFIG,
    CV_ES_CONFIG,
    CV_FR_CONFIG,
    CV_IT_CONFIG,
    CV_JA_CONFIG,
    CV_PT_CONFIG,
    CV_RU_CONFIG,
    CV_HI_CONFIG,
    CV_TR_CONFIG,
    CV_SV_CONFIG,
    CV_UK_CONFIG,
    CV_SW_CONFIG,
    CV_FA_CONFIG,
    CV_TH_CONFIG,
    CV_BE_CONFIG,
    CV_TA_CONFIG,
    CV_CS_CONFIG,
    CV_LV_CONFIG,
    CV_KA_CONFIG,
    CV_UR_CONFIG,
    CV_PL_CONFIG,
    CV_HU_CONFIG,
    CV_NL_CONFIG,
    CV_GL_CONFIG,
    CV_CY_CONFIG,
    CV_RO_CONFIG,
    CV_ET_CONFIG,
    CV_BR_CONFIG,
    CV_LT_CONFIG,
    CV_EL_CONFIG,
    CV_SK_CONFIG,
    CV_BG_CONFIG,
    CV_MK_CONFIG,
    CV_FI_CONFIG,
    CV_MR_CONFIG,
    CV_MN_CONFIG,
    CV_VI_CONFIG,
    CV_DA_CONFIG,
    CV_SL_CONFIG,
    CV_SR_CONFIG,
    CV_ML_CONFIG,
    CV_OC_CONFIG,
    CV_BN_CONFIG,
    CV_EN_TRANS_CONFIG,
    CV_AR_TRANS_CONFIG,
    CV_DE_TRANS_CONFIG,
    CV_ES_TRANS_CONFIG,
    CV_FR_TRANS_CONFIG,
    CV_IT_TRANS_CONFIG,
    CV_JA_TRANS_CONFIG,
    CV_PT_TRANS_CONFIG,
    CV_RU_TRANS_CONFIG,
    CV_HI_TRANS_CONFIG,
    CV_TR_TRANS_CONFIG,
    CV_SV_TRANS_CONFIG,
    CV_UK_TRANS_CONFIG,
    CV_SW_TRANS_CONFIG,
    CV_FA_TRANS_CONFIG,
    CV_TH_TRANS_CONFIG,
    CV_BE_TRANS_CONFIG,
    CV_TA_TRANS_CONFIG,
    CV_CS_TRANS_CONFIG,
    CV_LV_TRANS_CONFIG,
    CV_KA_TRANS_CONFIG,
    CV_UR_TRANS_CONFIG,
    CV_PL_TRANS_CONFIG,
    CV_HU_TRANS_CONFIG,
    CV_NL_TRANS_CONFIG,
    CV_GL_TRANS_CONFIG,
    CV_CY_TRANS_CONFIG,
    CV_RO_TRANS_CONFIG,
    CV_ET_TRANS_CONFIG,
    CV_BR_TRANS_CONFIG,
    CV_LT_TRANS_CONFIG,
    CV_EL_TRANS_CONFIG,
    CV_SK_TRANS_CONFIG,
    CV_BG_TRANS_CONFIG,
    CV_MK_TRANS_CONFIG,
    CV_FI_TRANS_CONFIG,
    CV_MR_TRANS_CONFIG,
    CV_MN_TRANS_CONFIG,
    CV_VI_TRANS_CONFIG,
    CV_DA_TRANS_CONFIG,
    CV_SL_TRANS_CONFIG,
    CV_SR_TRANS_CONFIG,
    CV_ML_TRANS_CONFIG,
    CV_OC_TRANS_CONFIG,
    CV_BN_TRANS_CONFIG,
    CV_EN_CONT_CONFIG,
    CV_AR_CONT_CONFIG,
    CV_DE_CONT_CONFIG,
    CV_ES_CONT_CONFIG,
    CV_FR_CONT_CONFIG,
    CV_IT_CONT_CONFIG,
    CV_JA_CONT_CONFIG,
    CV_PT_CONT_CONFIG,
    CV_RU_CONT_CONFIG,
    CV_HI_CONT_CONFIG,
    CV_TR_CONT_CONFIG,
    CV_SV_CONT_CONFIG,
    CV_UK_CONT_CONFIG,
    CV_SW_CONT_CONFIG,
    CV_FA_CONT_CONFIG,
    CV_TH_CONT_CONFIG,
    CV_BE_CONT_CONFIG,
    CV_TA_CONT_CONFIG,
    CV_CS_CONT_CONFIG,
    CV_LV_CONT_CONFIG,
    CV_KA_CONT_CONFIG,
    CV_UR_CONT_CONFIG,
    CV_PL_CONT_CONFIG,
    CV_HU_CONT_CONFIG,
    CV_NL_CONT_CONFIG,
    CV_GL_CONT_CONFIG,
    CV_CY_CONT_CONFIG,
    CV_RO_CONT_CONFIG,
    CV_ET_CONT_CONFIG,
    CV_BR_CONT_CONFIG,
    CV_LT_CONT_CONFIG,
    CV_EL_CONT_CONFIG,
    CV_SK_CONT_CONFIG,
    CV_BG_CONT_CONFIG,
    CV_MK_CONT_CONFIG,
    CV_FI_CONT_CONFIG,
    CV_MR_CONT_CONFIG,
    CV_MN_CONT_CONFIG,
    CV_VI_CONT_CONFIG,
    CV_DA_CONT_CONFIG,
    CV_SL_CONT_CONFIG,
    CV_SR_CONT_CONFIG,
    CV_ML_CONT_CONFIG,
    CV_OC_CONT_CONFIG,
    CV_BN_CONT_CONFIG,
]
