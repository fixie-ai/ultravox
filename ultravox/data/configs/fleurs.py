from ultravox.data import types

# 1) Base FLEURS config
FLEURS_BASE_CONFIG = types.DatasetConfig(
    name="fleurs",
    path="google/fleurs",
    transcript_template="{{transcription}}",
    assistant_template="{{transcription}}",
)

# The following languages needs special normalization (english vs languages that are split into characters)
LANG_ID_MAPPING = {
    "cmn_hans_cn": "zh",  # Mandarin Chinese (Simplified, China)
    "yue_hant_hk": "zh",  # Cantonese (Hong Kong, Traditional Chinese)
    "ja_jp": "ja",  # Japanese (Japan)
    "th_th": "th",  # Thai (Thailand)
    "lo_la": "lo",  # Lao (Laos)
    "my_mm": "my",  # Burmese (Myanmar)
    "en_us": "en",  # English (United States)
    "ar_eg": "ar",  # Arabic (Egypt)
}

# 2) Subset codes exactly as listed (with inline comments only)
FLEURS_LANG_CODES = [
    "af_za",  # Afrikaans (South Africa)
    "am_et",  # Amharic (Ethiopia)
    "ar_eg",  # Arabic (Egypt)
    "as_in",  # Assamese (India)
    "ast_es",  # Asturian (Spain)
    "az_az",  # Azerbaijani (Azerbaijan)
    "be_by",  # Belarusian (Belarus)
    "bg_bg",  # Bulgarian (Bulgaria)
    "bn_in",  # Bengali (India)
    "bs_ba",  # Bosnian (Bosnia & Herzegovina)
    "ca_es",  # Catalan (Spain)
    "ceb_ph",  # Cebuano (Philippines)
    "ckb_iq",  # Sorani Kurdish (Iraq)
    "cmn_hans_cn",  # Mandarin Chinese (Simplified, China)
    "cs_cz",  # Czech (Czech Republic)
    "cy_gb",  # Welsh (United Kingdom)
    "da_dk",  # Danish (Denmark)
    "de_de",  # German (Germany)
    "el_gr",  # Greek (Greece)
    "en_us",  # English (United States)
    "es_419",  # Spanish (Latin America)
    "et_ee",  # Estonian (Estonia)
    "fa_ir",  # Persian (Iran)
    "ff_sn",  # Fula/Fulfulde (Senegal)
    "fi_fi",  # Finnish (Finland)
    "fil_ph",  # Filipino (Philippines)
    "fr_fr",  # French (France)
    "ga_ie",  # Irish (Ireland)
    "gl_es",  # Galician (Spain)
    "gu_in",  # Gujarati (India)
    "ha_ng",  # Hausa (Nigeria)
    "he_il",  # Hebrew (Israel)
    "hi_in",  # Hindi (India)
    "hr_hr",  # Croatian (Croatia)
    "hu_hu",  # Hungarian (Hungary)
    "hy_am",  # Armenian (Armenia)
    "id_id",  # Indonesian (Indonesia)
    "ig_ng",  # Igbo (Nigeria)
    "is_is",  # Icelandic (Iceland)
    "it_it",  # Italian (Italy)
    "ja_jp",  # Japanese (Japan)
    "jv_id",  # Javanese (Indonesia)
    "ka_ge",  # Georgian (Georgia)
    "kam_ke",  # Kamba (Kenya)
    "kea_cv",  # Kabuverdianu (Cape Verde)
    "kk_kz",  # Kazakh (Kazakhstan)
    "km_kh",  # Khmer (Cambodia)
    "kn_in",  # Kannada (India)
    "ko_kr",  # Korean (South Korea)
    "ky_kg",  # Kyrgyz (Kyrgyzstan)
    "lb_lu",  # Luxembourgish (Luxembourg)
    "lg_ug",  # Ganda (Uganda)
    "ln_cd",  # Lingala (Congo - Kinshasa)
    "lo_la",  # Lao (Laos)
    "lt_lt",  # Lithuanian (Lithuania)
    "luo_ke",  # Luo (Kenya)
    "lv_lv",  # Latvian (Latvia)
    "mi_nz",  # Māori (New Zealand)
    "mk_mk",  # Macedonian (North Macedonia)
    "ml_in",  # Malayalam (India)
    "mn_mn",  # Mongolian (Mongolia)
    "mr_in",  # Marathi (India)
    "ms_my",  # Malay (Malaysia)
    "mt_mt",  # Maltese (Malta)
    "my_mm",  # Burmese (Myanmar)
    "nb_no",  # Norwegian Bokmål (Norway)
    "ne_np",  # Nepali (Nepal)
    "nl_nl",  # Dutch (Netherlands)
    "nso_za",  # Northern Sotho (South Africa)
    "ny_mw",  # Nyanja/Chichewa (Malawi)
    "oc_fr",  # Occitan (France)
    "om_et",  # Oromo (Ethiopia)
    "or_in",  # Oriya (India)
    "pa_in",  # Punjabi (India)
    "pl_pl",  # Polish (Poland)
    "ps_af",  # Pashto (Afghanistan)
    "pt_br",  # Portuguese (Brazil)
    "ro_ro",  # Romanian (Romania)
    "ru_ru",  # Russian (Russia)
    "sd_in",  # Sindhi (India)
    "sk_sk",  # Slovak (Slovakia)
    "sl_si",  # Slovenian (Slovenia)
    "sn_zw",  # Shona (Zimbabwe)
    "so_so",  # Somali (Somalia)
    "sr_rs",  # Serbian (Serbia)
    "sv_se",  # Swedish (Sweden)
    "sw_ke",  # Swahili (Kenya)
    "ta_in",  # Tamil (India)
    "te_in",  # Telugu (India)
    "tg_tj",  # Tajik (Tajikistan)
    "th_th",  # Thai (Thailand)
    "tr_tr",  # Turkish (Turkey)
    "uk_ua",  # Ukrainian (Ukraine)
    "umb_ao",  # Umbundu (Angola)
    "ur_pk",  # Urdu (Pakistan)
    "uz_uz",  # Uzbek (Uzbekistan)
    "vi_vn",  # Vietnamese (Vietnam)
    "wo_sn",  # Wolof (Senegal)
    "xh_za",  # Xhosa (South Africa)
    "yo_ng",  # Yoruba (Nigeria)
    "yue_hant_hk",  # Cantonese (Hong Kong, Traditional Chinese)
    "zu_za",  # Zulu (South Africa)
]

configs = [FLEURS_BASE_CONFIG]

for subset_code in FLEURS_LANG_CODES:
    fleurs_lang_config = types.DatasetConfig(
        name=f"fleurs-{subset_code}",
        base="fleurs",
        subset=subset_code,
        splits=[
            types.DatasetSplitConfig(name="train", num_samples=1509),
            types.DatasetSplitConfig(name="validation", num_samples=150),
            types.DatasetSplitConfig(name="test", num_samples=350),
        ],
        transcript_template="{{transcription}}",
        assistant_template="{{transcription}}",
    )

    eval_args = {}
    if subset_code in LANG_ID_MAPPING:
        eval_args["lang_id"] = LANG_ID_MAPPING[subset_code]

    fleurs_lang_trans_config = types.DatasetConfig(
        name=f"fleurs-{subset_code}-transcription",
        base=f"fleurs-{subset_code}",
        user_template=types.TRANSCRIPTION_USER_TEMPLATE,
        eval_config=types.EvalConfig(metric="wer", args=eval_args),
    )

    configs.append(fleurs_lang_config)
    configs.append(fleurs_lang_trans_config)
