import dataclasses
import glob
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Set

import datasets
import librosa
import simple_parsing
import soundfile as sf
from praatio import textgrid

from ultravox.tools.ds_tool import ds_commons

MFA_ENV_NAME = "aligner"


@dataclasses.dataclass
class TimestampGenerationTask:
    """
    This task is used to generate timestamps for the text transcription.
    It uses the Montreal Forced Aligner (MFA) to align the text with the audio. The result is a
    list of timestamps for each word in the text transcription. The timestamps are stored in a new
    column, in a list of dict format:
        [ {"start": float in seconds, "end": float in seconds, "text": first word str}, ... ]
    """

    # Jinja template for the text transcription that needs to be aligned
    template: str = simple_parsing.field(alias="-T")
    # The accoustic model to use for MFA alignment.
    # Make sure the dictionary and acoustic model are installed. See just install_mfa for an example (English).
    # Model index: https://mfa-models.readthedocs.io/en/latest/acoustic/index.html
    # For many languages there exists a {language}_mfa model that you can use, e.g. "english_mfa"
    mfa_acoustic_model: str = simple_parsing.field(alias="-m")
    # The dictionary to use for MFA alignment. Defaults to the same name as the acoustic model.
    mfa_dictionary: Optional[str] = simple_parsing.field(default=None, alias="-d")
    audio_column_name: str = simple_parsing.field(default="audio", alias="-a")
    sample_rate: int = simple_parsing.field(default=16000, alias="-r")
    # The column name to store the timestamps in
    timestamp_column_name: str = simple_parsing.field(default="timestamps", alias="-ts")
    aligned_ratio_check: float = simple_parsing.field(default=0.95, alias="-ar")

    def __post_init__(self):
        if self.mfa_dictionary is None:
            self.mfa_dictionary = self.mfa_acoustic_model

        try:
            # Make sure the MFA environment is installed
            subprocess.run(["conda", "run", "-n", MFA_ENV_NAME, "echo"], check=True)
        except subprocess.CalledProcessError:
            raise Exception(
                "Please install the MFA environment using `just install_mfa` first."
            )

        if self.template.startswith("@"):
            with open(self.template[1:], "r") as template_file:
                self.template = template_file.read()

    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: List[str],
    ) -> datasets.Dataset:
        # 0. create a temp directory to store the audio and text files
        # The files will be deleted when the with block ends or when an exception is raised
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. copy all audio-text pairs into the temp directory
            ds_split.map(
                self._store_sample_as_files,
                num_proc=num_proc,
                fn_kwargs={"exclude_fields": set(exclude_fields), "temp_dir": temp_dir},
            )

            count_wavs = len(glob.glob(os.path.join(temp_dir, "*.wav")))
            assert count_wavs == len(
                ds_split
            ), "Not all samples were stored as files. The id is likely not unique."

            # 2. run the alignment
            self._run_alignment(temp_dir, num_proc=num_proc)

            # 3. retrieve the timestamps
            ds_mapped = ds_split.map(
                self._retrieve_timestamps,
                num_proc=num_proc,
                writer_batch_size=writer_batch_size,
                fn_kwargs={"temp_dir": temp_dir},
            )

            # 4. filter out samples without timestamps (should be a small number)
            ds_mapped = ds_mapped.filter(
                lambda sample: sample[self.timestamp_column_name] is not None,
                num_proc=num_proc,
                writer_batch_size=writer_batch_size,
            )

            # 5. make sure most samples have timestamps
            if len(ds_split) * self.aligned_ratio_check > len(ds_mapped):
                raise Exception(
                    f"Found too many samples without timestamps: {len(ds_mapped)}/{len(ds_split)} aligned."
                )

        return ds_mapped

    def _retrieve_timestamps(self, sample, temp_dir: str):
        # find the timestamps for the audio and populate the timestamps column
        sample_id = self.get_id(sample)
        text_path = os.path.join(temp_dir, f"{sample_id}.TextGrid")
        if not os.path.exists(text_path):
            sample[self.timestamp_column_name] = None
            return sample

        tg = textgrid.openTextgrid(text_path, False)
        timestamps = tg.getTier("words").entries
        sample[self.timestamp_column_name] = [
            {"start": entry.start, "end": entry.end, "text": entry.label}
            for entry in timestamps
        ]
        return sample

    @staticmethod
    def get_id(sample):
        for key in ["id", "segment_id"]:
            if key in sample and isinstance(sample[key], str):
                return str(sample[key])
        for key in ["file", "path", "audio_file"]:
            if key in sample and isinstance(sample[key], str):
                return Path(sample[key]).stem
        raise ValueError("Could not find an ID in the sample")

    def _store_sample_as_files(self, sample, temp_dir: str, exclude_fields: Set[str]):
        sample_id = self.get_id(sample)
        audio_path = os.path.join(temp_dir, f"{sample_id}.wav")
        with open(audio_path, "wb") as f:
            audio = sample[self.audio_column_name]
            if audio["sampling_rate"] != self.sample_rate:
                audio["array"] = librosa.resample(
                    audio["array"],
                    orig_sr=audio["sampling_rate"],
                    target_sr=self.sample_rate,
                )
            sf.write(f, audio["array"], 16000, format="WAV", subtype="PCM_16")

        text_path = os.path.join(temp_dir, f"{sample_id}.txt")
        text = ds_commons.apply_jinja_template(self.template, sample, exclude_fields)
        with open(text_path, "w") as f:
            f.write(text)

    def _run_alignment(self, temp_dir: str, num_proc: int = 16) -> None:
        subprocess.run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                MFA_ENV_NAME,
                "mfa",
                "align",
                "--clean",
                "--single_speaker",
                "--use_mp",
                "-j",
                str(num_proc),
                temp_dir,
                self.mfa_acoustic_model,
                str(self.mfa_dictionary),
                temp_dir,
            ],
            check=True,
        )
