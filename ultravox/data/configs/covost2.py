from ultravox.data import types

CVST_BASE_CONFIG = types.DatasetConfig(
    name="covost2",
    path="fixie-ai/covost2",
    user_template=types.TRANSLATION_USER_TEMPLATE,
    transcript_template="{{sentence}}",
    assistant_template="{{translation}}",
    eval_config=types.EvalConfig(metric="bleu"),
)

CVST_AR_EN_CONFIG = types.DatasetConfig(
    name="covost2-ar-en",
    base="covost2",
    subset="ar_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=2_283),
        types.DatasetSplitConfig(name="validation", num_samples=1_758),
        types.DatasetSplitConfig(name="test", num_samples=1_695),
    ],
    user_template_args={"target": "English"},
)

CVST_CA_EN_CONFIG = types.DatasetConfig(
    name="covost2-ca-en",
    base="covost2",
    subset="ca_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=95_854),
        types.DatasetSplitConfig(name="validation", num_samples=12_730),
        types.DatasetSplitConfig(name="test", num_samples=12_730),
    ],
    user_template_args={"target": "English"},
)

CVST_CY_EN_CONFIG = types.DatasetConfig(
    name="covost2-cy-en",
    base="covost2",
    subset="cy_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_241),
        types.DatasetSplitConfig(name="validation", num_samples=690),
        types.DatasetSplitConfig(name="test", num_samples=690),
    ],
    user_template_args={"target": "English"},
)

CVST_DE_EN_CONFIG = types.DatasetConfig(
    name="covost2-de-en",
    base="covost2",
    subset="de_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=127_834),
        types.DatasetSplitConfig(name="validation", num_samples=13_511),
        types.DatasetSplitConfig(name="test", num_samples=13_511),
    ],
    user_template_args={"target": "English"},
)

CVST_EN_AR_CONFIG = types.DatasetConfig(
    name="covost2-en-ar",
    base="covost2",
    subset="en_ar",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Arabic"},
)

CVST_EN_CA_CONFIG = types.DatasetConfig(
    name="covost2-en-ca",
    base="covost2",
    subset="en_ca",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Catalan"},
)

CVST_EN_CY_CONFIG = types.DatasetConfig(
    name="covost2-en-cy",
    base="covost2",
    subset="en_cy",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Welsh"},
)

CVST_EN_DE_CONFIG = types.DatasetConfig(
    name="covost2-en-de",
    base="covost2",
    subset="en_de",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "German"},
)

CVST_EN_ET_CONFIG = types.DatasetConfig(
    name="covost2-en-et",
    base="covost2",
    subset="en_et",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Estonian"},
)

CVST_EN_FA_CONFIG = types.DatasetConfig(
    name="covost2-en-fa",
    base="covost2",
    subset="en_fa",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Persian"},
)

CVST_EN_ID_CONFIG = types.DatasetConfig(
    name="covost2-en-id",
    base="covost2",
    subset="en_id",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Indonesian"},
)

CVST_EN_JA_CONFIG = types.DatasetConfig(
    name="covost2-en-ja",
    base="covost2",
    subset="en_ja",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Japanese"},
    eval_config=types.EvalConfig(metric="bleu", args={"tokenize": "ja-mecab"}),
)

CVST_EN_LV_CONFIG = types.DatasetConfig(
    name="covost2-en-lv",
    base="covost2",
    subset="en_lv",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Latvian"},
)

CVST_EN_MN_CONFIG = types.DatasetConfig(
    name="covost2-en-mn",
    base="covost2",
    subset="en_mn",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Mongolian"},
)

CVST_EN_SL_CONFIG = types.DatasetConfig(
    name="covost2-en-sl",
    base="covost2",
    subset="en_sl",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Slovenian"},
)

CVST_EN_SV_CONFIG = types.DatasetConfig(
    name="covost2-en-sv",
    base="covost2",
    subset="en_sv-SE",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Swedish"},
)

CVST_EN_TA_CONFIG = types.DatasetConfig(
    name="covost2-en-ta",
    base="covost2",
    subset="en_ta",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Tamil"},
)

CVST_EN_TR_CONFIG = types.DatasetConfig(
    name="covost2-en-tr",
    base="covost2",
    subset="en_tr",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Turkish"},
)

CVST_EN_ZH_CONFIG = types.DatasetConfig(
    name="covost2-en-zh",
    base="covost2",
    subset="en_zh-CN",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=289_430),
        types.DatasetSplitConfig(name="validation", num_samples=15_531),
        types.DatasetSplitConfig(name="test", num_samples=15_531),
    ],
    user_template_args={"target": "Chinese"},
    eval_config=types.EvalConfig(metric="bleu", args={"tokenize": "zh"}),
)

CVST_ES_EN_CONFIG = types.DatasetConfig(
    name="covost2-es-en",
    base="covost2",
    subset="es_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=79_015),
        types.DatasetSplitConfig(name="validation", num_samples=13_221),
        types.DatasetSplitConfig(name="test", num_samples=13_221),
    ],
    user_template_args={"target": "English"},
)

CVST_ET_EN_CONFIG = types.DatasetConfig(
    name="covost2-et-en",
    base="covost2",
    subset="et_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_782),
        types.DatasetSplitConfig(name="validation", num_samples=1_576),
        types.DatasetSplitConfig(name="test", num_samples=1_571),
    ],
    user_template_args={"target": "English"},
)

CVST_FA_EN_CONFIG = types.DatasetConfig(
    name="covost2-fa-en",
    base="covost2",
    subset="fa_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=53_949),
        types.DatasetSplitConfig(name="validation", num_samples=3_445),
        types.DatasetSplitConfig(name="test", num_samples=3_445),
    ],
    user_template_args={"target": "English"},
)

CVST_FR_EN_CONFIG = types.DatasetConfig(
    name="covost2-fr-en",
    base="covost2",
    subset="fr_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=207_374),
        types.DatasetSplitConfig(name="validation", num_samples=14_760),
        types.DatasetSplitConfig(name="test", num_samples=14_760),
    ],
    user_template_args={"target": "English"},
)

CVST_ID_EN_CONFIG = types.DatasetConfig(
    name="covost2-id-en",
    base="covost2",
    subset="id_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_243),
        types.DatasetSplitConfig(name="validation", num_samples=792),
        types.DatasetSplitConfig(name="test", num_samples=844),
    ],
    user_template_args={"target": "English"},
)

CVST_IT_EN_CONFIG = types.DatasetConfig(
    name="covost2-it-en",
    base="covost2",
    subset="it_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=31_698),
        types.DatasetSplitConfig(name="validation", num_samples=8_940),
        types.DatasetSplitConfig(name="test", num_samples=8_951),
    ],
    user_template_args={"target": "English"},
)

CVST_JA_EN_CONFIG = types.DatasetConfig(
    name="covost2-ja-en",
    base="covost2",
    subset="ja_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_119),
        types.DatasetSplitConfig(name="validation", num_samples=635),
        types.DatasetSplitConfig(name="test", num_samples=684),
    ],
    user_template_args={"target": "English"},
)

CVST_LV_EN_CONFIG = types.DatasetConfig(
    name="covost2-lv-en",
    base="covost2",
    subset="lv_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=2_337),
        types.DatasetSplitConfig(name="validation", num_samples=1_125),
        types.DatasetSplitConfig(name="test", num_samples=1_629),
    ],
    user_template_args={"target": "English"},
)

CVST_MN_EN_CONFIG = types.DatasetConfig(
    name="covost2-mn-en",
    base="covost2",
    subset="mn_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=2_067),
        types.DatasetSplitConfig(name="validation", num_samples=1_761),
        types.DatasetSplitConfig(name="test", num_samples=1_759),
    ],
    user_template_args={"target": "English"},
)

CVST_NL_EN_CONFIG = types.DatasetConfig(
    name="covost2-nl-en",
    base="covost2",
    subset="nl_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=7_108),
        types.DatasetSplitConfig(name="validation", num_samples=1_699),
        types.DatasetSplitConfig(name="test", num_samples=1_699),
    ],
    user_template_args={"target": "English"},
)

CVST_PT_EN_CONFIG = types.DatasetConfig(
    name="covost2-pt-en",
    base="covost2",
    subset="pt_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=9_158),
        types.DatasetSplitConfig(name="validation", num_samples=3_318),
        types.DatasetSplitConfig(name="test", num_samples=4_023),
    ],
    user_template_args={"target": "English"},
)

CVST_RU_EN_CONFIG = types.DatasetConfig(
    name="covost2-ru-en",
    base="covost2",
    subset="ru_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=12_112),
        types.DatasetSplitConfig(name="validation", num_samples=6_110),
        types.DatasetSplitConfig(name="test", num_samples=6_300),
    ],
    user_template_args={"target": "English"},
)

CVST_SL_EN_CONFIG = types.DatasetConfig(
    name="covost2-sl-en",
    base="covost2",
    subset="sl_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_843),
        types.DatasetSplitConfig(name="validation", num_samples=509),
        types.DatasetSplitConfig(name="test", num_samples=360),
    ],
    user_template_args={"target": "English"},
)

CVST_SV_EN_CONFIG = types.DatasetConfig(
    name="covost2-sv-en",
    base="covost2",
    subset="sv-SE_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=2_160),
        types.DatasetSplitConfig(name="validation", num_samples=1_349),
        types.DatasetSplitConfig(name="test", num_samples=1_595),
    ],
    user_template_args={"target": "English"},
)

CVST_TA_EN_CONFIG = types.DatasetConfig(
    name="covost2-ta-en",
    base="covost2",
    subset="ta_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_358),
        types.DatasetSplitConfig(name="validation", num_samples=384),
        types.DatasetSplitConfig(name="test", num_samples=786),
    ],
    user_template_args={"target": "English"},
)

CVST_TR_EN_CONFIG = types.DatasetConfig(
    name="covost2-tr-en",
    base="covost2",
    subset="tr_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=3_966),
        types.DatasetSplitConfig(name="validation", num_samples=1_624),
        types.DatasetSplitConfig(name="test", num_samples=1_629),
    ],
    user_template_args={"target": "English"},
)

CVST_ZH_EN_CONFIG = types.DatasetConfig(
    name="covost2-zh-en",
    base="covost2",
    subset="zh-CN_en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=7_085),
        types.DatasetSplitConfig(name="validation", num_samples=4_843),
        types.DatasetSplitConfig(name="test", num_samples=4_898),
    ],
    user_template_args={"target": "English"},
)

configs = [
    CVST_BASE_CONFIG,
    CVST_AR_EN_CONFIG,
    CVST_CA_EN_CONFIG,
    CVST_CY_EN_CONFIG,
    CVST_DE_EN_CONFIG,
    CVST_EN_AR_CONFIG,
    CVST_EN_CA_CONFIG,
    CVST_EN_CY_CONFIG,
    CVST_EN_DE_CONFIG,
    CVST_EN_ET_CONFIG,
    CVST_EN_FA_CONFIG,
    CVST_EN_ID_CONFIG,
    CVST_EN_JA_CONFIG,
    CVST_EN_LV_CONFIG,
    CVST_EN_MN_CONFIG,
    CVST_EN_SL_CONFIG,
    CVST_EN_SV_CONFIG,
    CVST_EN_TA_CONFIG,
    CVST_EN_TR_CONFIG,
    CVST_EN_ZH_CONFIG,
    CVST_ES_EN_CONFIG,
    CVST_ET_EN_CONFIG,
    CVST_FA_EN_CONFIG,
    CVST_FR_EN_CONFIG,
    CVST_ID_EN_CONFIG,
    CVST_IT_EN_CONFIG,
    CVST_JA_EN_CONFIG,
    CVST_LV_EN_CONFIG,
    CVST_MN_EN_CONFIG,
    CVST_NL_EN_CONFIG,
    CVST_PT_EN_CONFIG,
    CVST_RU_EN_CONFIG,
    CVST_SL_EN_CONFIG,
    CVST_SV_EN_CONFIG,
    CVST_TA_EN_CONFIG,
    CVST_TR_EN_CONFIG,
    CVST_ZH_EN_CONFIG,
]
