import dataclasses
from typing import Dict, List, Optional

from ultravox.data import datasets
from ultravox.data import types

CONTINUATION_USER_TEMPLATE = f"Continue the following text using less than 50 words:\n\n{types.AUDIO_PLACEHOLDER}"
CONTINUATION_ASSISTANT_TEMPLATE = "{{continuation}}"
TRANSCRIPTION_USER_TEMPLATE = f"Transcribe\n{types.AUDIO_PLACEHOLDER}"

BOOLQ_CONFIG = types.DatasetConfig(
    name="boolq",
    path="fixie-ai/boolq-audio",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=10000),
        types.DatasetSplitConfig(name="validation", num_samples=1000),
    ],
    user_template="{{passage}}\n\n{AUDIO_PLACEHOLDER}",
    assistant_template="{{'True' if answer else 'False'}}",
    transcript_template="{{question}}",
)

CV_BASE_CONFIG = types.DatasetConfig(
    name="commonvoice",
    path="fixie-ai/common_voice_17_0",
    assistant_template="{{sentence}}",
    transcript_template="{{sentence}}",
)

CV_EN_CONFIG = types.DatasetConfig(
    name="commonvoice-en",
    base="commonvoice",
    subset="en",
    splits=[types.DatasetSplitConfig(name="train", num_samples=1_101_170)],
)

CV_AR_CONFIG = types.DatasetConfig(
    name="commonvoice-ar",
    base="commonvoice",
    subset="ar",
    splits=[types.DatasetSplitConfig(name="train", num_samples=28_369)],
)

CV_DE_CONFIG = types.DatasetConfig(
    name="commonvoice-de",
    base="commonvoice",
    subset="de",
    splits=[types.DatasetSplitConfig(name="train", num_samples=589_100)],
)

CV_ES_CONFIG = types.DatasetConfig(
    name="commonvoice-es",
    base="commonvoice",
    subset="es",
    splits=[types.DatasetSplitConfig(name="train", num_samples=336_846)],
)

CV_FR_CONFIG = types.DatasetConfig(
    name="commonvoice-fr",
    base="commonvoice",
    subset="fr",
    splits=[types.DatasetSplitConfig(name="train", num_samples=558_054)],
)

CV_IT_CONFIG = types.DatasetConfig(
    name="commonvoice-it",
    base="commonvoice",
    subset="it",
    splits=[types.DatasetSplitConfig(name="train", num_samples=169_771)],
)

CV_JA_CONFIG = types.DatasetConfig(
    name="commonvoice-ja",
    base="commonvoice",
    subset="ja",
    splits=[types.DatasetSplitConfig(name="train", num_samples=10_039)],
)

CV_PT_CONFIG = types.DatasetConfig(
    name="commonvoice-pt",
    base="commonvoice",
    subset="pt",
    splits=[types.DatasetSplitConfig(name="train", num_samples=21_968)],
)

CV_RU_CONFIG = types.DatasetConfig(
    name="commonvoice-ru",
    base="commonvoice",
    subset="ru",
    splits=[types.DatasetSplitConfig(name="train", num_samples=26_377)],
)

GS_XL_CONFIG = types.DatasetConfig(
    name="gigaspeech",
    path="speechcolab/gigaspeech",
    subset="xl",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_000_000),
        types.DatasetSplitConfig(name="validation", num_samples=10_000),
    ],
    assistant_template="{{text_proc.format_asr_text(text)}}",
    transcript_template="{{text_proc.format_asr_text(text)}}",
)

LS_BASE_CONFIG = types.DatasetConfig(
    name="librispeech",
    path="fixie-ai/librispeech_asr",
    assistant_template="{{text_proc.format_asr_text(text)}}",
    transcript_template="{{text_proc.format_asr_text(text)}}",
)

LS_CLEAN_CONFIG = types.DatasetConfig(
    name="librispeech-clean",
    base="librispeech",
    subset="clean",
    splits=[
        types.DatasetSplitConfig(name="train.100", num_samples=28_539),
        types.DatasetSplitConfig(name="train.360", num_samples=104_014),
    ],
)

LS_OTHER_CONFIG = types.DatasetConfig(
    name="librispeech-other",
    base="librispeech",
    subset="other",
    splits=[
        types.DatasetSplitConfig(name="train.500", num_samples=148_688),
    ],
)

PS_CLEAN_CONFIG = types.DatasetConfig(
    name="peoplespeech",
    path="fixie-ai/peoples_speech",
    subset="clean",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_000_000),
        types.DatasetSplitConfig(name="validation", num_samples=10_000),
    ],
)

# SODA_CONFIG = types.DatasetConfig(
#     name="soda",
#     path="fixie-ai/soda-audio",
#     splits=[
#         types.DatasetSplitConfig(name="train", num_samples=1_000_000),
#         types.DatasetSplitConfig(name="validation", num_samples=10_000),
#     ],
#     # Need way to specify message history.
#     audio_field="audio_second_last_turn",
#     assistant_template="{{alt_last_turn}}",
#     transcript_template="{{turns[-2]}}",
# )

VP_BASE_CONFIG = types.DatasetConfig(
    name="voxpopuli",
    path="facebook/voxpopuli",
    assistant_template="{{raw_text}}",
    transcript_template="{{raw_text}}",
)

VP_EN_CONFIG = types.DatasetConfig(
    name="voxpopuli-en",
    base="voxpopuli",
    subset="en",
    splits=[
        types.DatasetSplitConfig(name="train", num_samples=1_000_000),
        types.DatasetSplitConfig(name="validation", num_samples=10_000),
    ],
)

CV_EN_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-en-transcription",
    base="commonvoice-en",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_AR_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-ar-transcription",
    base="commonvoice-ar",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_DE_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-de-transcription",
    base="commonvoice-de",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_ES_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-es-transcription",
    base="commonvoice-es",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_FR_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-fr-transcription",
    base="commonvoice-fr",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_IT_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-it-transcription",
    base="commonvoice-it",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_JA_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-ja-transcription",
    base="commonvoice-ja",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_PT_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-pt-transcription",
    base="commonvoice-pt",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
CV_RU_TRANS_CONFIG = types.DatasetConfig(
    name="commonvoice-ru-transcription",
    base="commonvoice-ru",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)

LS_CLEAN_TRANS_CONFIG = types.DatasetConfig(
    name="librispeech-clean-transcription",
    base="librispeech-clean",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)
LS_OTHER_TRANS_CONFIG = types.DatasetConfig(
    name="librispeech-other-transcription",
    base="librispeech-other",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)

PS_CLEAN_TRANS_CONFIG = types.DatasetConfig(
    name="peoplespeech-clean-transcription",
    base="peoplespeech",
    user_template=TRANSCRIPTION_USER_TEMPLATE,
)

CV_EN_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-en-continuation",
    base="commonvoice-en",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_AR_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-ar-continuation",
    base="commonvoice-ar",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_DE_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-de-continuation",
    base="commonvoice-de",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_ES_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-es-continuation",
    base="commonvoice-es",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_FR_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-fr-continuation",
    base="commonvoice-fr",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_IT_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-it-continuation",
    base="commonvoice-it",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_JA_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-ja-continuation",
    base="commonvoice-ja",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_PT_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-pt-continuation",
    base="commonvoice-pt",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
CV_RU_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-ru-continuation",
    base="commonvoice-ru",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)

LS_CLEAN_CONT_CONFIG = types.DatasetConfig(
    name="librispeech-clean-continuation",
    base="librispeech-clean",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)
LS_OTHER_CONT_CONFIG = types.DatasetConfig(
    name="librispeech-other-continuation",
    base="librispeech-other",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)

PS_CLEAN_CONT_CONFIG = types.DatasetConfig(
    name="peoplespeech-clean-continuation",
    base="peoplespeech",
    user_template=CONTINUATION_USER_TEMPLATE,
    assistant_template=CONTINUATION_ASSISTANT_TEMPLATE,
)

INTERNAL_DATASETS = [
    BOOLQ_CONFIG,
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
    CV_EN_TRANS_CONFIG,
    CV_AR_TRANS_CONFIG,
    CV_DE_TRANS_CONFIG,
    CV_ES_TRANS_CONFIG,
    CV_FR_TRANS_CONFIG,
    CV_IT_TRANS_CONFIG,
    CV_JA_TRANS_CONFIG,
    CV_PT_TRANS_CONFIG,
    CV_RU_TRANS_CONFIG,
    CV_EN_CONT_CONFIG,
    CV_AR_CONT_CONFIG,
    CV_DE_CONT_CONFIG,
    CV_ES_CONT_CONFIG,
    CV_FR_CONT_CONFIG,
    CV_IT_CONT_CONFIG,
    CV_JA_CONT_CONFIG,
    CV_PT_CONT_CONFIG,
    CV_RU_CONT_CONFIG,
    GS_XL_CONFIG,
    LS_BASE_CONFIG,
    LS_CLEAN_CONFIG,
    LS_OTHER_CONFIG,
    LS_CLEAN_TRANS_CONFIG,
    LS_OTHER_TRANS_CONFIG,
    LS_CLEAN_CONT_CONFIG,
    LS_OTHER_CONT_CONFIG,
    PS_CLEAN_CONFIG,
    PS_CLEAN_TRANS_CONFIG,
    PS_CLEAN_CONT_CONFIG,
    VP_EN_CONFIG,
]
DATASET_MAP: Dict[str, types.DatasetConfig] = {}


def register_datasets(data_sets: List[types.DatasetConfig]):
    for config in data_sets:
        name = config.name
        assert name not in DATASET_MAP, f"Dataset {name} already registered"
        DATASET_MAP[name] = config


def unregister_datasets(datasets: List[str]):
    for name in datasets:
        del DATASET_MAP[name]


def _merge_configs(configs: List[types.DatasetConfig]) -> types.DatasetConfig:
    merged_config = dataclasses.replace(configs[0])
    for config in configs[1:]:
        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            if field.name != "base" and value is not None:
                merged_config = dataclasses.replace(
                    merged_config, **{field.name: value}
                )
    return merged_config


def create_dataset(
    name: str, args: types.VoiceDatasetArgs
) -> datasets.SizedIterableDataset:
    if name == "dummy":
        return datasets.LibriSpeechDummyDataset(args)
    assert name in DATASET_MAP, f"Unknown dataset: {name}"
    # Make a list of configs from root->base.
    configs: List[types.DatasetConfig] = []
    temp: Optional[str] = name
    while temp:
        config = DATASET_MAP[temp]
        configs.insert(0, config)
        temp = config.base
    # Set the root config, and then apply any non-None overrides from the subclasses.
    merged_config = _merge_configs(configs)
    # Sanity check.
    if not merged_config.path:
        raise ValueError(f"Dataset {name} has no path")
    if not merged_config.splits:
        raise ValueError(f"Dataset {name} has no splits")
    return datasets.GenericDataset(args, merged_config)


register_datasets(INTERNAL_DATASETS)
