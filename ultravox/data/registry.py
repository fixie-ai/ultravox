import dataclasses
from typing import Dict, List, Optional

from ultravox.data import datasets
from ultravox.data import types
from ultravox.data import datasets_boolq, datasets_commonvoice, datasets_covost2, \
    datasets_gigaspeech, datasets_librispeech, datasets_multilingual_librispeech, \
    datasets_peoplespeech, datasets_voxpopuli,datasets_wenetspeech


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


register_datasets(datasets_boolq.configs)
register_datasets(datasets_commonvoice.configs)
register_datasets(datasets_covost2.configs)
register_datasets(datasets_gigaspeech.configs)
register_datasets(datasets_librispeech.configs)
register_datasets(datasets_multilingual_librispeech.configs)
register_datasets(datasets_peoplespeech.configs)
register_datasets(datasets_voxpopuli.configs)
register_datasets(datasets_wenetspeech.configs)
