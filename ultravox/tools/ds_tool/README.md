# The Dataset Tool (ds_tool)

The Dataset Tool is a command-line tool that allows you to update and transform existing datasets. It is designed to be used in conjunction with the [Hugging Face Dataset Hub](https://huggingface.co/datasets).

The tool has multiple existing tasks. The workflow is:

- load an existing dataset from the Hugging Face Hub
- run a task (e.g. tts, textgen, timestamp, dedup, or audiosplit)
- push the changes back to the Hugging Face Hub

Many of the our datasets are generated using this tool. See [our HF Hub page](https://huggingface.co/fixie-ai) for examples.

Note: We do our own chunked upload for resiliency. See [Chunked Uploads](#chunked-uploads) for details of how the uploaded version is different from the original.
Some tasks also cache the partial results such as text or audio generations to make retrying easier.

### Setup: Write-enabled token

Make sure to have a WRITE-enabled `HF_TOKEN` from the Hugging Face website. I personally have a read-only HF_TOKEN and when I need to write I do `HF_TOKEN=HF_WRITE_TOKEN just ds_tool ...`.

## Creating a new dataset

Note that this tool does not allow you to create new datasets. For that you can use the native [creating a dataset](https://huggingface.co/docs/datasets/create_dataset) workflow which allows you to create a dataset either from a local CSV/JSON/Parquet file or directory, such as [`audiofolder`](https://huggingface.co/docs/datasets/audio_dataset#audiofolder).

```python
import datasets

ds = datasets.load_dataset("csv", data_files="my_file.csv")  # same with json or parquet
# or
ds = datasets.Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"], "type": ["grass", "water"]})
# or
ds = datasets.load_dataset("audiofolder", data_dir="/path/to/data")
# audio folder expects a folder structure like this:
# /path/to/data
#   metadata.csv  # a csv with columns: file_name,transcription
#   data/audio_file_1.wav
#   data/audio_file_2.wav

ds.push_to_hub("fixie-ai/my-dataset", private=True, num_shards={"train": 16})
# make sure to have a WRITE-enabled HF_TOKEN
```

## Chunked Uploads

When uploading a large dataset, we've sometimes had issues with the upload failing. To address this, `ds_tool` applies the transformation to each chunk and then uploads it before moving on to the next chunk.

For example in [fixie-ai/common_voice_17_0](https://huggingface.co/datasets/fixie-ai/common_voice_17_0/tree/main/da/train) you can see `da/train/chunk-range-000000000-000003484` which indicates this subset was uploaded via `ds_tool` and more specifically `DatasetChunkProcessor`.

To keep track of which chunks are already uploaded, we use locally store metadata in `.cache/ds_tool/processed_datasets`. Note however that this process is not atomic and can lead to inconsistencies if the size of chunks change between runs. It is also **append-only**. As such, it will not remove old chunks when re-uploading to the same dataset.

## Tasks

Here is a list of the existing tasks:

- `tts`: Generate audio samples from text using a TTS model.
- `textgen`: Generate text samples using a text generation model. Mainly used for producing continuations, which are used for training Ultravox.
- `timestamp`: Generate timestamps for audio samples using Montreal Forced Aligner.
- `dedup`: Deduplicate the text, using embedding similarity.
- `audiosplit`: Split longer audio samples into chunks.

### Example Commands

In [ds_tool.py](ds_tool.py), you can see a list of sample commands for each task.
There are also some example scripts in the [scripts/dataset_creation](../../../scripts/dataset_creation) directory.

Let's look at an example:

```bash
just ds_tool tts \
    -d google/boolq -u fixie-ai/boolq-audio \
    -S default -s train \
    -T {{question}} -i azura -a audio
```

This command will start from the original [BoolQ dataset](https://huggingface.co/datasets/google/boolq) dataset, and adds a synthetic `audio` column to each sample to read out the `question`. The `audio` column is generated using a random voice from Azure TTS. We also suppport Eleven Labs.

The extended dataset is then uploaded to the Hugging Face Hub under the name [`fixie-ai/boolq-audio`](https://huggingface.co/datasets/fixie-ai/boolq-audio). Make sure `HF_TOKEN` and `AZURE_TTS_API_KEY` are set.

Even though this is unnecessary to specify in this case, `-S default -s train` specifies that this command will only run on the `train` split of the `default` subset. If no split is specified, it will run on all splits sequentially.

### The Continuation Task via `textgen`

Computing "continuation" for ASR datasets is very important for training Ultravox. To do so, we use the `textgen` task with a custom template.

This task processes a dataset by generating text continuations for each row using a specified language model. The generated continuations are stored in a new column, and the transformed dataset is then uploaded for further use.

#### How It Works

Following the same pattern as the other tasks, the `textgen` task will:

1. **Downloads the original dataset** specified by `dataset_name`.
2. **Generates text continuations** for each row using the selected AI model and stores them in `new_column_name`.
3. **Uploads the transformed dataset** under `upload_name`.

#### Key Parameters

- `new_column_name`: Specifies the name of the column where the generated continuations will be stored.
- `dataset_name`: The original dataset being processed.
- `upload_name`: The name of the transformed dataset after processing.
- `base_url, api_key, language_model`: Defines the AI provider and model used for text generation.

#### Example Usage

```bash
just ds_tool textgen \
 --new_column_name continuation \
 --dataset_name mozilla-foundation/common_voice_17_0 \
 --dataset_subset bn \
 --upload_name fixie-ai/common_voice_17_0 \
 --private \
 --base_url https://api.fireworks.ai/inference/v1 \
 --api_key $FIREWORKS_API_KEY \
 --token $HF_TOKEN \
 --language_model accounts/fireworks/models/llama-v3p1-8b-instruct \
 --template @ultravox/tools/ds_tool/continuation.jinja \
 --max_tokens 64 \
 --num_workers 60 \
 --writer_batch_size 100 \
 --num_shards 8 \
 --exclude_fields audio
```

#### Important Considerations

- **Ensure the continuation template is correct (`--template`)**. It should be designed to ingest the right column in the input dataset
- **For datasets with fewer than ~100,000 rows**, always set `num_shards` to 16, or else the processed dataset will be incompatible with the training pipeline.
- **Exclude large fields like audio (`--exclude_fields audio`)**, or it may cause RAM overflow. Ensure to exclude any other large data fields if present.
- **Choose the right AI provider** using base_url, api_key, and language_model. Different providers may have different model behaviors, so test accordingly.

### Creating a new task

Let's say you want to create a new task called `my_task`. To do so, first you'd probably want to create a new module under `ultravox/tools/ds_tool/tasks/` called `my_task.py` (or similar) that contains a new `dataclass` called `MyTask` inheriting from `ds_commons.DSToolTask`:

```python
# ultravox/tools/ds_tool/tasks/my_task.py

@dataclasses.dataclass
class MyTask(ds_commons.DSToolTask):
    # Define CLI parameters for your task, these are added to the common ds_tool CLI args
    param1: str = simple_parsing.field(alias="-p1")
    param2: int = simple_parsing.field(alias="-p2")

    # main logic for transforming the dataset, one split at a time
    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: List[str],
    ) -> datasets.Dataset:
        # Implement the logic for your task.
        # We regularly use `.map` and `.filter` to modify the dataset.
        # For example:
        ds_split = ds_split.map(
            self._map_sample,
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
            fn_kwargs={"exclude_fields": exclude_fields},
        )
        return ds_split
```

Next, all is left is to add the new task to the `DS_TOOL_TASKS` dictionary in `ds_tool.py`:

```python
DS_TOOL_TASKS: Dict[str, Type[ds_commons.DSToolTask]] = {
    "tts": tts_task.TtsTask,
    ...
    "my_task": my_task.MyTask,
}
```

And you're all set!
You can now use your new task by running `just ds_tool my_task -d from_dataset -u to_dataset --param1 "foo" -p2 42` from the root directory.

Once the dataset is uploaded, if you want to train with it, you need to create a new dataset config for that. See the README at [ultravox/data](../../data/README.md) for more details.
