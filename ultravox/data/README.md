# Datasets

This directory contains the code for loading datasets to be used in training, validation, and evaluation.

To add a new dataset, you need to:

0. (if applicable) Create the dataset on HF Hub
   - see [ds_tool](../tools/ds_tool/README.md) for how to create or transform an existing dataset
1. Add the [dataset config](#dataset-configs) to `ultravox/data/configs`
2. [Register the dataset](#dataset-registry) in `ultravox/data/registry.py`
3. Add the dataset name to your config file of choice. See [example_config.yaml](../training/configs/example_config.yaml) for an example.

## Dataset Configs

Exising dataset configs are defined in `ultravox/data/configs`. Each dataset config specifies the dataset name, where the data is (only HF Hub supported at the moment), what are the transcript and assistant templates, and the splits to use for training, validation, and evaluation.

New datasets can also be added dynamically per training run. See `TrainConfig.data_sets` for more details. We rarely ever use this feature and instrad rely on dataset config files.

Dataset configs can inherit from other dataset configs, and can override any of the fields in the base config. For example see how the `commonvoice-en-continuation` config inherits from the `commonvoice-en` config which itself inherits from the `commonvoice` config:

```python
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

CV_EN_CONT_CONFIG = types.DatasetConfig(
    name="commonvoice-en-continuation",
    base="commonvoice-en",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)
```

## Dataset Registry

All the datasets need to be registered via [`ultravox/data/registry.py::register_datasets`](./registry.py) before being used in a training run. See the same file to see how the existing dataset configs are registered.

## GenericDataset

The `GenericDataset` class is the main dataset class that is used to load datasets. It is initialized with a `DatasetConfig` that we showed above as well a `VoiceDatasetArgs` object that is controlled via CLI args and differs slightly based on the split: train, validation, or test (aka evaluation).

In most cases, you will not need to interact with this class directly.

## Utility Dataset Classes

**InterleaveDataset:**
This class is used to interleave multiple datasets together for training. It allows for weighted interleaving of datasets. The weight uses the `num_samples` field from the `DatasetSplitConfig` object.
Note that even though weighting is involved, the dataset order is deterministic and the same across multiple runs.

**Range:**
This class is used to limit the number of samples in a dataset. It is often used to limit the number of samples in a validation or evaluation set to a smaller number.

**Dataproc:**
This class is used to wrap the proprocessor function of the model. The samples before this step are model-agnostic.

For more info, see the [`prepare_dataset` function in `training/train.py`](../training/train.py) as well as [`create_dataset` in `registry.py`](./registry.py) to see how the elements above are combined.
