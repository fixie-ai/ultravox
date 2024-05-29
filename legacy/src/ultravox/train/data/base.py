import enum
import os
import random
import typing as t
from dataclasses import dataclass
from dataclasses import field

import datasets
import pyrallis
import torch
import transformers
from train.data import epd

from .gigaspeech import clean_text_for_training
from .prompts import get_random_asr_prompt


class DatasetType(enum.Enum):
    LIBRISPEECH = "librispeech"
    GIGASPEECH = "gigaspeech"
    COMMON_VOICE = "commonvoice"


pyrallis.decode.register(DatasetType, lambda x: DatasetType(x))


def get_dataset(
    dataset_name: datasets.Dataset,
    dev_env: bool = False,
    streaming: bool = True,
    shuffle: bool = True,
    sampling_rate: int = 16_000,
    max_duration_in_seconds: float = 20.0,
    val_max_num_samples: t.Optional[int] = 256,
) -> t.Tuple[datasets.IterableDataset, datasets.IterableDataset]:
    train_ds = get_dataset_split(
        dataset_name=dataset_name,
        train=True,
        streaming=streaming,
        shuffle=shuffle,
        dev_env=dev_env,
        sampling_rate=sampling_rate,
        max_duration_in_seconds=max_duration_in_seconds,
    )
    validation_ds = get_dataset_split(
        dataset_name=dataset_name,
        train=False,
        streaming=streaming,
        shuffle=shuffle,
        dev_env=dev_env,
        sampling_rate=sampling_rate,
        max_duration_in_seconds=max_duration_in_seconds,
        max_num_samples=val_max_num_samples,  # FIXME: remove this limit? I'm worried about OOM right now
    )

    return train_ds, validation_ds


def is_text_unempty(sample):
    return bool(sample.get("text", None))


def get_dataset_split(
    dataset_name: datasets.Dataset,
    train: bool = True,
    dev_env: bool = False,
    streaming: bool = True,
    shuffle: bool = True,
    sampling_rate: int = 16_000,
    max_duration_in_seconds: float = 20.0,
    max_num_samples: t.Optional[int] = None,
) -> datasets.IterableDataset:
    """
    Args:
        dataset_name: The name of the dataset to load.
        train: Whether to load the training or validation split.
        dev_env: Whether to use a development environment (e.g. for testing).
        streaming: Whether to use the streaming version of the dataset.
            This is almost always True for training.
        shuffle: Whether to shuffle the dataset.
        sampling_rate: The sampling rate to use for the audio.
        max_duration_in_seconds: The maximum duration of audio to use.
        max_num_samples: The maximum number of samples to load.
    """
    kwargs = {
        "trust_remote_code": True,
        "token": os.environ.get("HF_ACCESS_TOKEN", None),
        "streaming": streaming,
    }

    if dev_env and not train:
        max_num_samples = 10

    if dataset_name == DatasetType.LIBRISPEECH:
        kwargs["path"] = "librispeech_asr"
        kwargs["split"] = "train.clean.360" if train else "validation.clean"
        # kwargs["streaming"] = True  # TODO?
    elif dataset_name == DatasetType.GIGASPEECH:
        kwargs["path"] = "speechcolab/gigaspeech"
        # kwargs["name"] = "xs" if dev_env else "s"
        kwargs["name"] = "s"
        kwargs["split"] = "train" if train else "validation"
    elif dataset_name == DatasetType.COMMON_VOICE:
        kwargs["path"] = "mozilla-foundation/common_voice_16_1"
        # kwargs["path"] = "mozilla-foundation/common_voice_11_0"
        kwargs["name"] = "en"  # TODO: combine multiple languages
        kwargs["split"] = "train" if train else "validation"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    ds: datasets.IterableDataset = datasets.load_dataset(**kwargs)

    if shuffle:
        ds = ds.shuffle()

    if isinstance(ds, datasets.Dataset):
        ds = ds.to_iterable_dataset(num_shards=4)

    if max_num_samples:
        if isinstance(ds, datasets.IterableDataset):
            ds = ds.take(max_num_samples)
        else:
            ds = ds.select(range(max_num_samples))

    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=sampling_rate))
    ds = ds.filter(IsAudioLengthInRange(max_duration_in_seconds))

    if dataset_name == DatasetType.COMMON_VOICE:
        # Common Voice text already seems clean, but it's called "sentence"
        ds = ds.rename_column("sentence", "text")
    else:
        ds = ds.map(clean_text_for_training, input_columns=["text"])

    ds = ds.filter(is_text_unempty)

    return ds


@dataclass
class IsAudioLengthInRange:
    max_duration_in_seconds: float

    def __call__(self, sample):
        audio_len = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
        return audio_len < self.max_duration_in_seconds


@dataclass
class AudioTextTokenizerConfig:
    system_prompt: t.Optional[str] = None
    prompt: t.Optional[str] = None
    add_audio_tag_ratio: float = 0.5
    asr_label: bool = True
    inference_mode: bool = False
    late_eou_label: bool = False
    early_eou_labels: bool = False
    early_eou_count: int = 1
    early_mid_count: int = -1
    crop_audio_prob: float = 0.0
    crop_audio_band: t.Tuple[float, float] = field(default_factory=lambda: (0.2, 0.8))
    crop_silence_prob: float = 0.0
    crop_silence_band: t.Tuple[float, float] = field(default_factory=lambda: (0, 1))

    def __post_init__(self):
        if self.crop_audio_prob > 0:
            if (
                self.crop_audio_band[0] <= 0
                or self.crop_audio_band[1] >= 1
                or self.crop_audio_band[0] >= self.crop_audio_band[1]
            ):
                raise ValueError(
                    "crop_audio_band should be a tuple of two floats in (0, 1)."
                )


@dataclass
class AudioTextTokenizer:
    audio_processor: transformers.Wav2Vec2Processor
    tokenizer: transformers.LlamaTokenizer
    audio_to_tokens_ratio: int
    """Defines how many frames of audio_features correspond to one token."""
    cfg: AudioTextTokenizerConfig

    def __post_init__(self):
        if self.cfg.late_eou_label:
            self.cropped_transcriber = epd.CroppedTranscriber()
        eou = self.tokenizer.encode("END")
        mid = self.tokenizer.encode("...")
        self.eou_token_id = eou[-1]
        self.mid_token_id = mid[-1]
        if self.cfg.early_eou_labels:
            if len(eou) > 2 or len(mid) > 2:
                raise ValueError(
                    "When early_eou_labels are enabled, the tokenizer should not split the 'END' or '...' tokens. Cannot recover."
                )

    def generate_prompt_tokens(
        self,
        num_audio_tokens: int,
        transcription: str = None,
        prompt: str = None,
        remove_last_eos: bool = False,
    ) -> t.List[int]:
        # Putting unk token at the beginning of the sequence.
        # This should be replaced with audio features inside the model
        audio_placeholder = self.tokenizer.unk_token * num_audio_tokens
        if random.random() < self.cfg.add_audio_tag_ratio:
            audio_placeholder = f"<speech>{audio_placeholder}</speech>"

        if prompt is None:
            prompt = self.cfg.prompt
        if prompt is None:
            prompt = get_random_asr_prompt()

        if "{audio}" not in prompt:
            prompt = "{audio}\n" + prompt

        chat = []
        if self.cfg.system_prompt:
            chat.append({"role": "system", "content": self.cfg.system_prompt})

        chat.append({"role": "user", "content": prompt.format(audio=audio_placeholder)})

        if transcription:
            chat.append(
                {"role": "assistant", "content": f"Transcript: {transcription}"}
            )

        tokens: t.List[int] = self.tokenizer.apply_chat_template(chat, tokenize=True)

        if remove_last_eos:
            for i in range(len(tokens) - 1, -1, -1):
                if tokens[i] == self.tokenizer.eos_token_id:
                    tokens = tokens[:i]
                    break

        return tokens

    def __call__(self, sample: t.Dict[str, t.Any]):
        audio_array = sample["audio"]["array"]
        audio_sr = sample["audio"]["sampling_rate"]
        text = sample.get("text", None)
        cropped = random.random() < self.cfg.crop_audio_prob

        if cropped:
            keep_portion = random.uniform(*self.cfg.crop_audio_band)
            audio_array = audio_array[..., : int(len(audio_array) * keep_portion)]
            if random.random() < self.cfg.crop_silence_prob:
                # Add silence to the end of the audio
                sil_len = random.uniform(*self.cfg.crop_silence_band) * audio_sr
                audio_array[..., -int(sil_len) :] *= 100
            if self.cfg.late_eou_label:
                text = self.cropped_transcriber(
                    audio_array, audio_sr, full_transcript=text
                )
            else:
                # In this case we can ignore the whole text (no ASR task, just EPD-early)
                text = None

        audio_input = self.audio_processor(audio_array, sampling_rate=audio_sr)

        if "input_features" in audio_input:
            audio_feats = audio_input.input_features[0]
        else:
            audio_feats = audio_input.input_values[0]

        processed = {}
        processed["audio_features"] = audio_feats

        if isinstance(self.audio_processor, transformers.WhisperProcessor):
            # Whisper pads all inputs to the same length of 30 seconds
            num_audio_tokens = 30 * audio_sr // self.audio_to_tokens_ratio
        else:
            num_audio_tokens = audio_array.shape[-1] // self.audio_to_tokens_ratio

        suffix = ""
        if self.cfg.late_eou_label:
            suffix = " [...]" if cropped else " [END]"

        tokens = self.generate_prompt_tokens(
            num_audio_tokens,
            transcription=text + suffix if text is not None else None,
            prompt=sample.get("prompt", None),
        )
        input_tokens_only = self.generate_prompt_tokens(
            num_audio_tokens,
            prompt=sample.get("prompt", None),
        )
        tokens_without_suffix = self.generate_prompt_tokens(
            num_audio_tokens,
            transcription=text,
            prompt=sample.get("prompt", None),
            remove_last_eos=True,
        )

        if self.cfg.asr_label:
            # The input tokens are not labelled (hence -100)
            num_unlabelled = len(input_tokens_only)
        else:
            # Transcript tokens are not labelled in this case
            # Only the suffix (EOU + EOS) is labelled.
            # TODO: do we need to label EOS? Probably okay, huh?
            num_unlabelled = len(tokens_without_suffix)

        labels = [-100] * num_unlabelled + tokens[num_unlabelled:]

        if self.cfg.inference_mode:
            tokens = input_tokens_only

        # Simple way to keep track of which tokens are audio and which are text
        audio_token_mask = [
            1 if t == self.tokenizer.unk_token_id else 0 for t in input_tokens_only
        ]

        # gotta assert that they are continuous
        processed["audio_token_start_idx"] = audio_token_mask.index(1)
        processed["audio_token_len"] = audio_token_mask.count(1)

        if self.cfg.early_eou_labels:
            # Set audio labels as a bunch of ... tokens followed by a single END
            start = processed["audio_token_start_idx"]
            end = start + processed["audio_token_len"]
            # Shift everything to the right by 1 to match "next"-word prediction task
            start += 1
            end += 1

            # The last early_eou_count places are labeled as END if not cropped, otherwise ...
            for i in range(end - self.cfg.early_eou_count, end):
                labels[i] = self.mid_token_id if cropped else self.eou_token_id
            end -= self.cfg.early_eou_count

            # If early_mid_count is -1, we label all the remaining tokens as `...`
            # otherwise, we randomly select that many tokens to be labeled as `...`
            mid_labels = range(start, end)
            if 0 <= self.cfg.early_mid_count < len(mid_labels):
                mid_labels = random.sample(mid_labels, self.cfg.early_mid_count)
            for i in mid_labels:
                labels[i] = self.mid_token_id

        processed["input_ids"] = tokens
        processed["labels"] = labels
        processed["attention_mask"] = [1] * len(tokens)

        return processed


@dataclass
class DataCollatorForSeq2SeqWithAudio(transformers.DataCollatorForSeq2Seq):
    audio_dtype: torch.dtype = torch.float32

    def __call__(self, features, *args, **kwargs):
        audio_features = [f.pop("audio_features") for f in features]
        # audio_token_mask = [f.pop("audio_token_mask") for f in features]

        batch = super().__call__(features, *args, **kwargs)

        batch["audio_features"] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f, dtype=self.audio_dtype) for f in audio_features],
            batch_first=True,
        )
        # batch["audio_token_mask"] = torch.nn.utils.rnn.pad_sequence(
        #     [torch.tensor(f) for f in audio_token_mask], batch_first=True
        # )

        return batch
