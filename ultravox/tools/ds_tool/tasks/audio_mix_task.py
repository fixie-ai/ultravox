import dataclasses
import random
from typing import Optional

import datasets
import librosa
import numpy as np
import simple_parsing

from ultravox.tools.ds_tool import ds_commons


@dataclasses.dataclass
class AudioMixTask(ds_commons.DSToolTask):
    mix_dataset_name: str = simple_parsing.field(alias="-m")

    # Optional parameters (with defaults) come after
    audio_field: str = simple_parsing.field(default="audio", alias="-a")
    mix_dataset_subset: Optional[str] = simple_parsing.field(
        default="default", alias="-msubset"
    )
    mix_dataset_split: Optional[str] = simple_parsing.field(
        default="train", alias="-msplit"
    )
    mix_audio_field: str = simple_parsing.field(default="audio", alias="-maudio")

    # Mixing parameters
    max_length: float = simple_parsing.field(default=15.0, alias="-L")
    shuffle: bool = simple_parsing.field(default=False, alias="-es")
    seed: int = simple_parsing.field(default=42, alias="-ss")

    # Batch size for processing
    batch_size: int = simple_parsing.field(default=8, alias="-b")

    normalize: bool = simple_parsing.field(default=True, alias="-n")
    min_volume_ratio: float = simple_parsing.field(default=0.2, alias="-minvr")
    max_volume_ratio: float = simple_parsing.field(default=0.5, alias="-maxvr")

    def __post_init__(self):
        random.seed(self.seed)

        # Load the mix dataset in standard mode (not streaming)
        print(f"Loading mix dataset: {self.mix_dataset_name} in standard mode")
        download_config = datasets.DownloadConfig(max_retries=2)
        self.mix_dataset = datasets.load_dataset(
            self.mix_dataset_name,
            self.mix_dataset_subset,
            split=self.mix_dataset_split,
            download_config=download_config,
            streaming=False,  # Changed to standard mode
        )

        # Initialize dataset
        if self.shuffle:
            self.mix_dataset = self.mix_dataset.shuffle(seed=self.seed)

        print(f"Mix dataset loaded and initialized: {self.mix_dataset}")

        # Store the dataset length for process distribution
        self.mix_dataset_length = len(self.mix_dataset)

        # Initialize process-specific state dictionaries
        self.process_states = {}

    def _get_process_state(self, process_id, num_proc):
        """Get or create a state dictionary for the current process"""
        if process_id not in self.process_states:
            # Calculate the subset range for this process
            subset_size = self.mix_dataset_length // max(1, num_proc)
            start_idx = process_id * subset_size
            end_idx = (
                start_idx + subset_size
                if process_id < num_proc - 1
                else self.mix_dataset_length
            )

            self.process_states[process_id] = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "current_idx": start_idx,
                "current_audio": None,
                "current_position": 0,
                "current_sr": None,
            }

        return self.process_states[process_id]

    def _get_audio_segment(self, length_seconds, target_sr, process_id=0, num_proc=1):
        """Get an audio segment of specified length in seconds and sample rate"""
        # Convert length from seconds to samples
        length_samples = int(length_seconds * target_sr)

        # Get or initialize process state
        state = self._get_process_state(process_id, num_proc)
        mix_array = np.array([], dtype=np.float32)

        while len(mix_array) < length_samples:
            # If we don't have current audio or have used all of it, load a new one
            if state["current_audio"] is None or state["current_position"] >= len(
                state["current_audio"]
            ):

                # Get next sample from this process's subset
                sample = self.mix_dataset[state["current_idx"]]

                # Move to next index, wrapping around within this process's subset
                state["current_idx"] = (
                    state["current_idx"] + 1
                ) % self.mix_dataset_length
                if (
                    state["current_idx"] < state["start_idx"]
                    or state["current_idx"] >= state["end_idx"]
                ):
                    state["current_idx"] = state["start_idx"]

                # Process the sample
                mix_audio = sample[self.mix_audio_field]
                current_mix_array = mix_audio["array"]
                current_mix_sr = mix_audio["sampling_rate"]

                if current_mix_sr is None:
                    raise ValueError("Sample rate is None")
                if current_mix_array is None:
                    raise ValueError("Audio array is None")

                # Resample if needed
                if current_mix_sr != target_sr:
                    current_mix_array = librosa.resample(
                        current_mix_array, orig_sr=current_mix_sr, target_sr=target_sr
                    )

                # Update state with new audio
                state["current_audio"] = current_mix_array
                state["current_position"] = 0
                state["current_sr"] = target_sr

            # Determine how much audio we need and how much we can get from current audio
            samples_needed = length_samples - len(mix_array)
            samples_available = len(state["current_audio"]) - state["current_position"]
            samples_to_take = min(samples_needed, samples_available)

            # Take the segment from current position
            segment = state["current_audio"][
                state["current_position"] : state["current_position"] + samples_to_take
            ]
            mix_array = np.append(mix_array, segment)

            # Update position
            state["current_position"] += samples_to_take

        return mix_array

    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: set[str],
    ) -> datasets.Dataset:
        print(
            f"Mixing audio from field '{self.audio_field}' with samples from {self.mix_dataset_name}..."
        )

        # Process each sample to add the mixed audio
        ds_split = ds_split.map(
            self._mix_audio,
            fn_kwargs={"num_proc": num_proc},  # Pass the number of processes
            num_proc=num_proc,
            batch_size=self.batch_size,
            writer_batch_size=min(64, writer_batch_size),
            batched=True,
        )

        # Cast the audio column to the proper datasets.Audio type
        if len(ds_split) > 0 and "audio" in ds_split.column_names:
            sampling_rate = ds_split[0]["audio"]["sampling_rate"]
            ds_split = ds_split.cast_column(
                "audio", datasets.Audio(sampling_rate=sampling_rate)
            )

        return ds_split

    def _normalize_audio(self, source_array, target_array, volume_ratio=1.0):
        """Normalize source_array to have the same RMS volume as target_array, multiplied by volume_ratio"""

        # Calculate RMS (root mean square) of both signals
        def rms(x):
            return np.sqrt(np.mean(np.square(x)))

        source_rms = rms(source_array)
        target_rms = rms(target_array)

        # Scale source_array to match target_array's volume multiplied by volume_ratio
        if source_rms > 0 and target_rms > 0:  # Avoid division by zero
            scale_factor = (target_rms / source_rms) * volume_ratio
            return source_array * scale_factor

        return source_array

    def _mix_audio(self, sample, num_proc=1):
        """Mix the audio with samples from the mix pool"""
        # Get the process ID (0 for single process)
        process_id = getattr(
            datasets.utils.logging.get_verbosity(), "_process_index", 0
        )
        if process_id is None:
            process_id = 0

        # Handle batched input
        batch_size = len(sample[self.audio_field])
        mixed_audios = []
        valid_indices = []

        # Process each sample in the batch
        for idx in range(batch_size):
            input_audio = sample[self.audio_field][idx]
            input_array = input_audio["array"]
            input_sr = input_audio["sampling_rate"]

            # Calculate input audio length in seconds
            input_length_seconds = len(input_array) / input_sr

            # Skip if input audio is longer than max_length
            if input_length_seconds >= self.max_length:
                continue

            # Determine available time for mix audio
            available_seconds = max(0, self.max_length - input_length_seconds)

            mix_length_seconds = random.uniform(
                available_seconds * 0.25, available_seconds
            )

            # Get mix audio of required length in seconds, passing process ID
            mix_array = self._get_audio_segment(
                mix_length_seconds, input_sr, process_id, num_proc
            )

            # Normalize the mix audio to have the same mean volume as input audio if requested
            if self.normalize:
                # Choose a random volume ratio between min and max
                volume_ratio = random.uniform(
                    self.min_volume_ratio, self.max_volume_ratio
                )
                mix_array = self._normalize_audio(mix_array, input_array, volume_ratio)

            # Randomly split the mix_array
            split_point = random.randint(0, len(mix_array))
            mix_prefix = mix_array[:split_point]
            mix_suffix = mix_array[split_point:]

            # Create the final mixed audio by combining prefix + input + suffix
            mixed_audio = np.concatenate([mix_prefix, input_array, mix_suffix])

            # Prevent clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val

            # Store the mixed audio
            mixed_audios.append({"array": mixed_audio, "sampling_rate": input_sr})
            valid_indices.append(idx)

        # Create a new result dictionary with only the valid samples
        result = {}
        if len(valid_indices) == batch_size:
            # If no samples were skipped, keep the original sample data
            result = {k: v for k, v in sample.items()}
            result[self.audio_field] = mixed_audios
        else:
            # If some samples were skipped, create a filtered dictionary
            for key in sample:
                if isinstance(sample[key], list) and len(sample[key]) == batch_size:
                    result[key] = [sample[key][i] for i in valid_indices]
                else:
                    # For non-list fields or lists of different length, copy as is
                    result[key] = sample[key]
            result[self.audio_field] = mixed_audios

        return result

    @classmethod
    def chunking_allowed(cls) -> bool:
        return True
