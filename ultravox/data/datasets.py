import abc
import dataclasses
import logging
import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Sequence

import datasets as hf_datasets
import jinja2
import numpy as np
import streaming as mds
import torch
import torch.nn.functional as F
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


@dataclasses.dataclass
class DataCollatorForSeq2SeqWithAudio(transformers.DataCollatorForSeq2Seq):
    # when enabled, the alt_input_ids, alt_attention_mask, and alt_labels fields are used for computing the KL loss in UltravoxModel
    include_alt_fields: bool = False

    def __call__(self, features, *args, **kwargs):
        audio_values = [f.pop("audio_values", None) for f in features]
        if self.include_alt_fields:
            # these fields are hard-coded in the transformer data collator, so they need special handling before calling the super method
            alt_features = [
                {
                    "input_ids": f.pop("alt_input_ids"),
                    "attention_mask": f.pop("alt_attention_mask"),
                    "labels": f.pop("alt_labels"),
                }
                for f in features
            ]
        input_ids_lens = torch.LongTensor([f["input_ids"].shape[-1] for f in features])
        batch = super().__call__(features, *args, **kwargs)
        if self.include_alt_fields:
            alt_batch = super().__call__(alt_features, *args, **kwargs)
            batch["alt_input_ids"] = alt_batch["input_ids"]
            batch["alt_attention_mask"] = alt_batch["attention_mask"]
            batch["alt_labels"] = alt_batch["labels"]

        # Pad the last dimension of all audio_values to the same length, with 0s on the right.
        if audio_values and audio_values[0] is not None:
            max_len = max([x.shape[-1] for x in audio_values])
            batch["audio_values"] = torch.stack(
                [F.pad(x, (0, max_len - x.shape[-1])) for x in audio_values]
            )
            if self.tokenizer.padding_side == "left":
                displacement = batch["input_ids"].shape[-1] - input_ids_lens
                batch["audio_token_start_idx"] += displacement.to(
                    batch["audio_token_start_idx"].device
                )

        return batch


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
    def __len__(self):
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

    def _init_dataset(self, dataset: data.Dataset, num_samples: int) -> None:
        self._dataset = dataset
        self._length = num_samples

    def __len__(self):
        return self._length

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
        for _, row in enumerate(self._dataset):
            sample = self._get_sample(row)
            if sample is None:
                raise ValueError(
                    f"Sample is None in dataset {self._config.alias} for row {row}"
                )

            if self._args.include_audio:
                # If audio_field is set, make sure the sample has audio data.
                if sample.audio is None:
                    raise ValueError(f"Audio is None for sample {sample}")
                if sample.audio.shape[-1] == 0:
                    raise ValueError(f"Audio length is 0 for sample {sample}")
                if (
                    self._args.max_audio_duration_secs is not None
                    and sample.audio.shape[-1] / data_sample.SAMPLE_RATE
                    > self._args.max_audio_duration_secs
                ):
                    duration = sample.audio.shape[-1] / data_sample.SAMPLE_RATE
                    warnings.warn(
                        f"Audio length ({duration}s) exceeds max audio duration ({self._args.max_audio_duration_secs}s), skipping sample."
                    )
                    continue

            yield sample
            actual_length += 1
            if actual_length == len(self) + 1:
                warnings.warn(
                    f"The presumed length {self._length} has been exceeded for {self._config.name}:{self._args.split.value}. Make sure to update."
                )
        if actual_length != len(self):
            warnings.warn(
                f"Mismatch between presumed length ({self._length}) and actual length ({actual_length}) for {self._config.name}:{self._args.split.value}. Make sure to update."
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
            if split.split_type == self._args.split:
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
        super()._init_dataset(dataset, total_samples)

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
                dataset=self,
                **self._config.user_template_args,
            )
            assistant_content = jinja2.Template(
                self._config.assistant_template, undefined=jinja2.StrictUndefined
            ).render(**row, text_proc=text_proc, dataset=self)
            transcript = jinja2.Template(
                self._config.transcript_template, undefined=jinja2.StrictUndefined
            ).render(**row, text_proc=text_proc, dataset=self)
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


class LibriSpeechDummyDataset(VoiceDataset):
    def __init__(self, args: types.VoiceDatasetArgs) -> None:
        super().__init__(args)
        # This dataset doesn't support streaming.
        dataset = self._load_hf_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            streaming=False,
        )
        self._init_dataset(dataset, 73)

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
            self._get_audio(row, "audio"),
            audio_transcript=text,
        )


class EmptyDataset(SizedIterableDataset):
    def __init__(self, length: int = 1) -> None:
        self._length = length

    def __iter__(self):
        return iter([])

    def __len__(self):
        return self._length


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
                yield next(ds_iters[iter_index])
            ds_pos[iter_index] += 1

    def __len__(self):
        return self._total_samples


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


class Range(SizedIterableDataset):
    """Limits the number of samples from another dataset."""

    def __init__(
        self, dataset: SizedIterableDataset, num_samples: Optional[int] = None
    ) -> None:
        self._dataset = dataset
        self._length = num_samples or len(dataset)
        if self._length > len(dataset):
            raise ValueError("num_samples exceeds dataset length.")

    def __iter__(self):
        for i, sample in enumerate(self._dataset):
            if i >= self._length:
                break
            yield sample

    def __len__(self):
        return self._length
