import dataclasses
from typing import Optional

import datasets
import openai
import simple_parsing
import yaml

from ultravox.data import text_proc
from ultravox.tools.ds_tool import caching
from ultravox.tools.ds_tool import ds_commons

chat_client: caching.CachingChatWrapper


@dataclasses.dataclass
class TextGenerationTask(ds_commons.DSToolTask):
    new_column_name: str = simple_parsing.field(alias="-c")
    template: str = simple_parsing.field(alias="-T")
    json_mode: bool = simple_parsing.field(default=False, alias="-j")

    language_model: str = simple_parsing.field(default="gpt-4o", alias="-m")
    base_url: Optional[str] = simple_parsing.field(default=None, alias="-b")
    api_key: Optional[str] = simple_parsing.field(default=None, alias="-k")
    max_tokens: int = 128
    temperature: float = 0

    def __post_init__(self):
        # The OAI client is separate from the task to avoid pickling issues when multiprocessing.
        # Caching the client to avoid repeated calls to the API if the tool fails.
        global chat_client
        chat_client = caching.CachingChatWrapper(
            openai.Client(base_url=self.base_url, api_key=self.api_key),
            unique_id=f"{self.base_url}__{self.language_model}",
        )
        if self.template.startswith("@"):
            with open(self.template[1:], "r") as template_file:
                self.template = template_file.read()

    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: set[str],
    ) -> datasets.Dataset:
        print(f'Generating "{self.new_column_name}" with template:\n{self.template}')
        ds_mapped = ds_split.map(
            lambda sample: self._map_sample(sample, exclude_fields),
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
        )

        # Filter out samples where new_column_name is None
        return ds_mapped.filter(
            lambda sample: sample[self.new_column_name] is not None,
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
        )

    def _map_sample(self, sample, exclude_fields):
        # using a Jinja template for some added flexibility, template can include variables and functions
        # e.g., {{ text }} or {{ text_proc.format_asr_text(text) }}
        try:
            rendered = ds_commons.apply_jinja_template(
                self.template, sample, exclude_fields
            )
        except text_proc.FormatASRError as e:
            print(f"Format ASR Error {e}. Filtering out sample.")
            sample[self.new_column_name] = None
            return sample

        if self.json_mode:
            turns = yaml.safe_load(rendered)
            assert isinstance(turns, list)
            assert all(isinstance(turn, dict) for turn in turns)
            assert len(turns) > 0
            assert turns[-1].get("role", None) == "user"
        else:
            turns = [{"role": "user", "content": rendered}]

        sample[self.new_column_name] = chat_client.chat_completion(
            model=self.language_model,
            messages=turns,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return sample
