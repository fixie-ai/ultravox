import dataclasses
import random

import datasets
import simple_parsing

from ultravox.tools.ds_tool import ds_commons


@dataclasses.dataclass
class AudioSplitTask(ds_commons.DSToolTask):
    """A task that splits audio samples into smaller segments with configurable lengths.

    This task processes audio data by:
    1. Filtering audio samples based on minimum and maximum length constraints
    2. Splitting each audio sample into multiple segments with lengths randomly chosen
       between min_segment_length and max_segment_length
    3. Preserving all non-audio fields from the original samples
    4. Generating unique IDs for each segment by appending '-seg-{index}' to the original ID
    5. Renaming the audio field to "audio" to be consistent with the default audio field used by ultravox

    Args:
        audio_field (str): Name of the field containing audio data. Defaults to "audio".
        id_field (str): Name of the field containing unique identifiers. Defaults to "id".
        min_length (float): Minimum allowed length of input audio in seconds. Defaults to 2.0.
        max_length (float): Maximum allowed length of input audio in seconds. Defaults to infinity.
        min_segment_length (float): Minimum length of output segments in seconds. Defaults to 2.0.
        max_segment_length (float): Maximum length of output segments in seconds. Defaults to 10.0.
            Each segment's length is randomly chosen between min_segment_length and max_segment_length.
        random_seed (int): Seed for random segment length generation. Defaults to 42.
        max_segments_per_file (int): Maximum number of segments to create from a single audio file. Defaults to 10.
        batch_size (int): Number of samples to process at once. Defaults to 8.
    """

    # Source audio field to process
    src_audio_field: str = simple_parsing.field(default="audio", alias="-a")
    # Target audio field to store the split audio segments
    tgt_audio_field: str = simple_parsing.field(default="audio", alias="-A")
    # Unique identifier field
    id_field: str = simple_parsing.field(default="id", alias="-i")
    # Length filters (in seconds)
    min_length: float = simple_parsing.field(default=2.0, alias="-l")
    max_length: float = simple_parsing.field(default=float("inf"), alias="-L")
    # Segment length parameters (in seconds)
    min_segment_length: float = simple_parsing.field(default=2.0, alias="-s")
    max_segment_length: float = simple_parsing.field(default=10.0, alias="-S")
    # Random seed for reproducibility
    random_seed: int = simple_parsing.field(default=42, alias="-r")
    # Maximum segments per audio file (to prevent offset overflow)
    max_segments_per_file: int = simple_parsing.field(default=10, alias="-m")
    # Batch size for processing
    batch_size: int = simple_parsing.field(default=8, alias="-b")

    def __post_init__(self):
        random.seed(self.random_seed)
        if self.max_segment_length < self.min_segment_length:
            raise ValueError("max_segment_length must be >= min_segment_length")

        if self.min_length < self.min_segment_length:
            raise ValueError("min_length must be >= min_segment_length")
        if self.max_length < self.min_length:
            raise ValueError("self.max_length must be >= self.min_length")
        if self.max_segments_per_file < self.min_segment_length:
            raise ValueError("max_segments_per_file must be >= min_segment_length")

    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: set[str],
    ) -> datasets.Dataset:
        print(f"Splitting audio from field '{self.src_audio_field}' into segments...")

        # First filter by audio length
        ds_split = ds_split.filter(
            self._filter_by_length,
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
        )

        # Process each sample to get a dataset with audio segments
        ds_split = ds_split.map(
            self._split_audio,
            remove_columns=ds_split.column_names,
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

    def _filter_by_length(self, sample):
        audio = sample[self.src_audio_field]
        duration = len(audio["array"]) / audio["sampling_rate"]
        return self.min_length <= duration <= self.max_length

    def _split_audio(self, samples: dict[str, list]):
        # Handle batched input
        all_segments = []

        # Process each sample in the batch
        batch_size = len(samples[self.src_audio_field])
        for idx in range(batch_size):
            audio = samples[self.src_audio_field][idx]
            sample_dict = {k: samples[k][idx] for k in samples.keys()}

            array = audio["array"]
            sampling_rate = audio["sampling_rate"]

            # Create segments covering the entire audio
            current_pos = 0
            segment_index = 0

            while current_pos < len(array):
                # Calculate available duration from current position
                remaining_samples = len(array) - current_pos
                remaining_seconds = remaining_samples / sampling_rate

                # Determine segment length
                if remaining_seconds <= self.max_segment_length:
                    # Final segment - if it's too short, adjust starting position
                    if remaining_seconds >= self.min_segment_length:
                        segment_len_samples = remaining_samples
                    else:
                        # If remaining audio is too short, take a segment of min_segment_length
                        # from the end of the audio
                        segment_len_samples = int(
                            self.min_segment_length * sampling_rate
                        )
                        current_pos = max(0, len(array) - segment_len_samples)
                else:
                    # Randomly sized segment within specified range
                    segment_len_sec = random.uniform(
                        self.min_segment_length, self.max_segment_length
                    )
                    segment_len_samples = int(segment_len_sec * sampling_rate)

                # Extract segment
                segment_array = array[current_pos : current_pos + segment_len_samples]

                # Add segment to results
                all_segments.append(
                    {
                        **{
                            k: sample_dict[k]
                            for k in sample_dict.keys()
                            if k != self.src_audio_field
                        },
                        self.tgt_audio_field: {
                            "array": segment_array,
                            "sampling_rate": sampling_rate,
                        },
                        self.id_field: f"{sample_dict[self.id_field]}-seg-{segment_index}",
                    }
                )

                segment_index += 1
                current_pos += segment_len_samples

                # Limit segments per batch to avoid offset overflow
                # If we've hit the limit, but still have audio to process,
                # break and let subsequent processing handle it
                if segment_index >= self.max_segments_per_file and current_pos < len(
                    array
                ):
                    break

        # Handle empty segments case
        if not all_segments:
            return {k: [] for k in samples.keys()}

        # Restructure segments into a dictionary of lists
        return {k: [seg[k] for seg in all_segments] for k in all_segments[0].keys()}

    @classmethod
    def chunking_allowed(cls) -> bool:
        return True
