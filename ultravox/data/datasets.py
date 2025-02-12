import abc
import logging
import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Sequence

import datasets as hf_datasets
import jinja2
import numpy as np
import streaming as mds
import transformers
from torch.utils import data

from ultravox.data import data_sample
from ultravox.data import text_proc
from ultravox.data import types

# TODO(juberti): set these in the environment so they don't need to be hard-coded here.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "fixie-training"

# Silence the spurious warnings coming from the MosaicML streaming library.
logging.getLogger("streaming.base.dataset").setLevel(logging.ERROR)


def _get_messages(
    *turns: str, sys_prompt: Optional[str] = None, assistant_last: bool = True
) -> List[Dict[str, str]]:
    """
    Convert a list of strings into a list of messages, alternating between user and assistant.
    If `sys_prompt` is set, it is prepended as a system message.
    If `assistant_last` is True, the assistant's message is the last one.
    """
    messages = []

    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})

    roles = ["user", "assistant"]

    # Make sure the last turn is the assistant's iff assistant_last is True.
    if (len(turns) + assistant_last) % 2 == 0:
        roles = roles[::-1]

    messages += [{"role": roles[i % 2], "content": c} for i, c in enumerate(turns)]

    return messages


class SizedIterableDataset(abc.ABC, data.IterableDataset):
    """
    An interface for an IterableDataset that provides a length method.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass


class VoiceDataset(SizedIterableDataset):
    """
    Base class for streaming voice datasets.
    Wraps a Hugging Face dataset or MDS-formatted dataset from GCP.
    """

    def __init__(self, args: types.VoiceDatasetArgs) -> None:
        super().__init__()
        self._args = args
        self._rng = np.random.default_rng(self._args.shuffle_seed)
        self._name = "[unset]"
        self._length = -1

    # num_samples is the total number of samples in the dataset
    def _init_dataset(
        self,
        dataset: data.Dataset,
        name: str,
        num_samples: int,
    ) -> None:
        self._dataset = dataset
        self._name = name
        self._length = num_samples

    def __len__(self):
        return self._length

    @property
    def name(self):
        return self._name

    def _load_hf_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        streaming: bool = True,
        audio_field: Optional[str] = None,
    ) -> data.Dataset:
        # HF datasets sometimes fails to download due to network issues, so retry a few times.
        dataset = hf_datasets.load_dataset(
            path,
            name,
            split=split,
            trust_remote_code=True,
            streaming=streaming,
            download_config=hf_datasets.DownloadConfig(max_retries=10),
        )
        if audio_field is not None:
            dataset = dataset.cast_column(
                audio_field, hf_datasets.Audio(sampling_rate=data_sample.SAMPLE_RATE)
            )
        if self._args.shuffle:
            if streaming:
                dataset = dataset.shuffle(
                    seed=self._args.shuffle_seed,
                    buffer_size=self._args.shuffle_buffer_size,
                )
            else:
                dataset = dataset.shuffle(seed=self._args.shuffle_seed)
        return dataset

    def _load_mds_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        batch_size: int = 1,
    ) -> data.Dataset:
        gcs_path = path.replace("/", "_")
        if name:
            gcs_path += f"/{name}"
        if split:
            gcs_path += f"/{split}"
        url = f"gs://fixie-datasets/mds/{gcs_path}"
        temp_dir = os.path.join(
            tempfile.gettempdir(), f"mds_{gcs_path.replace('/', '_')}"
        )
        return mds.StreamingDataset(
            remote=url,
            local=temp_dir,
            batch_size=batch_size,
            shuffle=self._args.shuffle,
            shuffle_seed=self._args.shuffle_seed,
        )

    def __iter__(self):
        actual_length = 0
        skipped_samples = 0
        bad_samples = 0
        dataset_iter = iter(self._dataset)
        for row in dataset_iter:
            actual_length += 1
            sample = self._get_sample(row)
            if sample is None:
                print(f"Sample is None in dataset {self._config.alias} for row {row}")
                bad_samples += 1
                continue  # Skip this sample and proceed to the next

            if self._args.include_audio:
                if sample.audio is None:
                    print(f"Audio is None for sample {sample}")
                    bad_samples += 1
                    continue  # Skip this sample
                if sample.audio.shape[-1] == 0:
                    print(f"Audio length is 0 for sample {sample}")
                    bad_samples += 1
                    continue  # Skip this sample
                if (
                    self._args.max_audio_duration_secs > 0
                    and sample.audio.shape[-1] / data_sample.SAMPLE_RATE
                    > self._args.max_audio_duration_secs
                ):
                    skipped_samples += 1
                    continue  # Skip this sample

            yield sample

        logging.info(
            f"Extracted {actual_length} samples from {self.name} (total: {len(self)}), removed {bad_samples} bad samples, and skipped {skipped_samples} samples for exceeding max audio duration ({self._args.max_audio_duration_secs}s)."
        )

    @abc.abstractmethod
    def _get_sample(
        self, row: transformers.BatchFeature
    ) -> Optional[data_sample.VoiceSample]:
        """
        Converts a row from the dataset into a VoiceSample.
        Returns None if the sample should be skipped.
        """

    def _get_audio(
        self, row: transformers.BatchFeature, column_name: Optional[str] = "audio"
    ) -> np.ndarray:
        # Hugging Face datasets have an Audio object, with array and sampling_rate fields.
        # For MDS, this object is flattened into audio_array and audio_sampling_rate fields.
        if column_name in row:
            audio = row[column_name]["array"]
            sampling_rate = row[column_name]["sampling_rate"]
        elif f"{column_name}_array" in row:
            audio = row[f"{column_name}_array"]
            sampling_rate = row[f"{column_name}_sampling_rate"]
        else:
            raise ValueError("No audio field found in row.")
        assert sampling_rate == data_sample.SAMPLE_RATE
        return audio

    def _make_messages(
        self, user_content: str, assistant_content: str
    ) -> List[Dict[str, str]]:
        return _get_messages(user_content, assistant_content)

    def _make_sample(
        self,
        messages: List[Dict[str, str]],
        audio: np.ndarray,
        audio_transcript: Optional[str] = None,
    ) -> data_sample.VoiceSample:
        if not self._args.include_audio:
            return data_sample.VoiceSample(messages)
        return data_sample.VoiceSample(
            messages, audio, audio_transcript=audio_transcript
        )


class GenericDataset(VoiceDataset):
    def __init__(
        self, args: types.VoiceDatasetArgs, config: types.DatasetConfig
    ) -> None:
        assert config.splits is not None
        assert config.path is not None
        assert config.mds_batch_size is not None
        super().__init__(args)
        self._config = config
        dsets = []
        total_samples = 0
        for split in config.splits:
            if split.split == self._args.split:
                if not config.use_mds:
                    ds = self._load_hf_dataset(
                        config.path,
                        config.subset,
                        split=split.name,
                        audio_field=config.audio_field,
                    )
                else:
                    ds = self._load_mds_dataset(
                        config.path,
                        name=config.subset,
                        split=split.name,
                        batch_size=config.mds_batch_size,
                    )
                dsets.append(ds)
                total_samples += split.num_samples
        assert (
            len(dsets) > 0
        ), f"The {config.name} dataset has no {self._args.split} splits."
        dataset = ds if len(dsets) == 1 else hf_datasets.concatenate_datasets(dsets)

        dataset_name = f"{config.name}.{self._args.split.value}"

        super()._init_dataset(dataset, dataset_name, total_samples)

    def __str__(self):
        return f"GenericDataset({self._config})"

    def _get_sample(self, row) -> Optional[data_sample.VoiceSample]:
        assert self._config.user_template is not None
        assert self._config.user_template_args is not None
        assert self._config.assistant_template is not None
        assert self._config.transcript_template is not None
        try:
            user_content = jinja2.Template(
                self._config.user_template, undefined=jinja2.StrictUndefined
            ).render(
                **row,
                text_proc=text_proc,
                **self._config.user_template_args,
            )
            assistant_content = jinja2.Template(
                self._config.assistant_template, undefined=jinja2.StrictUndefined
            ).render(**row, text_proc=text_proc)
            transcript = jinja2.Template(
                self._config.transcript_template, undefined=jinja2.StrictUndefined
            ).render(**row, text_proc=text_proc)
        except jinja2.TemplateError as e:
            print(f"Error rendering template: {e}")
            print(f"user_template: {self._config.user_template}")
            print(f"assistant_template: {self._config.assistant_template}")
            print(f"transcript_template: {self._config.transcript_template}")
            print(f"sample keys: {list(row.keys())}")
            raise ValueError(
                "Template rendering failed. Make sure all keys in the template exist in the sample."
            ) from e
        if not self._args.include_audio:
            user_content = user_content.replace(
                types.AUDIO_PLACEHOLDER, f'"{transcript}"'
            )
        messages = _get_messages(user_content, assistant_content)
        audio = self._get_audio(row, self._config.audio_field)
        return self._make_sample(messages, audio, audio_transcript=transcript)

    def get_config(self):
        return self._config


class LibriSpeechDummyDataset(GenericDataset):
    def __init__(self, args: types.VoiceDatasetArgs) -> None:
        VoiceDataset.__init__(self, args)
        # This dataset doesn't support streaming.
        dataset = self._load_hf_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            streaming=False,
        )
        self._init_dataset(dataset, "dummy", 73)

    def __str__(self):
        return "LibriSpeechDummyDataset"

    @property
    def name(self):
        return "dummy"

    def get_config(self):
        return types.DatasetConfig(
            name="dummy",
            path="hf-internal-testing/librispeech_asr_dummy",
        )

    def _get_sample(
        self, row: transformers.BatchFeature
    ) -> Optional[data_sample.VoiceSample]:
        text = text_proc.format_asr_text(row["text"])
        user_content = "Transcribe\n"
        user_content += (
            types.AUDIO_PLACEHOLDER if self._args.include_audio else f'"{text}"'
        )
        return self._make_sample(
            self._make_messages(user_content, text),
            # some of our test models that use this dataset can only handle up to 4 seconds of audio
            self._get_audio(row, "audio")[: 4 * data_sample.SAMPLE_RATE],
            audio_transcript=text,
        )


class EmptyDataset(SizedIterableDataset):
    def __init__(self, length: int = 1) -> None:
        self._length = length

    def __iter__(self):
        return iter([])

    def __len__(self):
        return self._length

    def __str__(self):
        return f"EmptyDataset(length={self._length})"

    @property
    def name(self):
        return "empty"


class InterleaveDataset(SizedIterableDataset):
    """Interleaves multiple SizedIterableDataset objects based on normalized weights."""

    def __init__(
        self,
        datasets: Sequence[SizedIterableDataset],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Args:
            datasets: A list of SizedIterableDataset objects.
            weights: An optional list of dataset weights, i.e., the number of times it should be repeated.
            seed: Optional seed for reproducibility.
        """
        self._datasets = datasets
        if weights is not None:
            assert len(weights) == len(datasets)
        else:
            weights = [1.0] * len(datasets)
        self._weights = weights
        self._weighted_samples = [int(w * len(d)) for w, d in zip(weights, datasets)]
        self._total_samples = sum(self._weighted_samples)

    def __iter__(self):
        ds_iters = [iter(ds) for ds in self._datasets]
        ds_pos = [0] * len(ds_iters)
        # Find the iterator that is least far along and vend from it.
        for i in range(self._total_samples):
            min_fraction = 1.0
            for j in range(len(ds_iters)):
                iter_fraction = ds_pos[j] / self._weighted_samples[j]
                if iter_fraction < min_fraction:
                    min_fraction = iter_fraction
                    iter_index = j
            try:
                yield next(ds_iters[iter_index])
            except StopIteration:
                ds_iters[iter_index] = iter(self._datasets[iter_index])
                try:
                    yield next(ds_iters[iter_index])
                except StopIteration:
                    warnings.warn(
                        f"Dataset {iter_index} is empty. num_workers is likely too high. Stopping iteration."
                    )
                    break
            ds_pos[iter_index] += 1

    def __len__(self):
        return self._total_samples

    def __str__(self):
        return "+".join([f"{d}:{w:.2f}" for w, d in zip(self._weights, self._datasets)])

    @property
    def name(self):
        return "+".join([ds.name for ds in self._datasets])


class Dataproc(SizedIterableDataset):
    """Base class to preprocess a dataset of VoiceSamples."""

    def __init__(self, dataset: SizedIterableDataset) -> None:
        self._dataset = dataset

    @abc.abstractmethod
    def _process(self, sample: data_sample.VoiceSample) -> Dict[str, Any]:
        pass

    def __iter__(self):
        return (self._process(sample) for sample in self._dataset)

    def __len__(self):
        return len(self._dataset)

    def __str__(self):
        return f"Dataproc({self._dataset})"

    @property
    def name(self):
        return self._dataset.name


class Range(SizedIterableDataset):
    """Limits the number of samples from another dataset."""

    def __init__(
        self, dataset: SizedIterableDataset, num_samples: Optional[int] = None
    ) -> None:
        self._dataset = dataset
        self._length = num_samples or len(dataset)
        if self._length > len(dataset):
            warnings.warn(
                f"num_samples ({self._length}) exceeds dataset length ({len(dataset)}). Truncating to {len(dataset)}."
            )
            self._length = len(dataset)
        self._name = f"{dataset.name}.{self._length}"

    def __iter__(self):
        for i, sample in enumerate(self._dataset):
            if i >= self._length:
                break
            yield sample

    def __str__(self):
        return f"Range({self._dataset}%{len(self)})"

    def __len__(self):
        return self._length

    @property
    def name(self):
        return self._name

    def get_config(self):
        if isinstance(self._dataset, GenericDataset):
            return self._dataset.get_config()
        else:
            raise ValueError("Cannot get config for non-GenericDataset")
