import dataclasses
from typing import Any, Dict, Set

import annoy
import datasets
import openai
import simple_parsing

from ultravox.tools.ds_tool import caching
from ultravox.tools.ds_tool import ds_commons

VECTOR_DB = None
embedding_client: caching.CachingEmbeddingWrapper


# Example usage:
#   just ds_tool dedup -d fixie-ai/proper-noun-challenge -T "\"{{user}}\"" -t 0.6 -u fixie-ai/proper-noun-challenge-filtered --check_empty_columns --chunk_split_threshold 1000000
@dataclasses.dataclass
class DeduplicationTask(ds_commons.DSToolTask):
    """
    This task is used to deduplicate a dataset based on a text template.
    It uses a vector database to store the embeddings of the text and then checks for duplicates based on the cosine similarity of the embeddings.
    """

    text_template: str = simple_parsing.field(alias="-T")
    # The threshold for the cosine similarity of the embeddings.
    # This is task dependent and might need to be tuned. When in doubt, use a number close to 0.5.
    threshold: float = simple_parsing.field(alias="-t")
    # The OpenAI model to use for the embeddings.
    model: str = simple_parsing.field(default="text-embedding-3-small", alias="-m")
    # The dimensions of the embeddings (for models allowing shortened embeddings)
    model_dimensions: int = simple_parsing.field(default=1024, alias="-md")
    # The number of trees to build in the Annoy index.
    num_trees: int = simple_parsing.field(default=100, alias="-nt")

    @classmethod
    def chunking_allowed(cls) -> bool:
        return False

    def __post_init__(self):
        if self.text_template.startswith("@"):
            with open(self.text_template[1:], "r") as template_file:
                self.text_template = template_file.read()

        global embedding_client
        embedding_client = caching.CachingEmbeddingWrapper(
            openai.Client(), unique_id=self.model
        )

    def _add_dedup_text_and_embedding_columns(self, sample, exclude_fields: Set[str]):
        sample["dedup_text"] = ds_commons.apply_jinja_template(
            self.text_template, sample, exclude_fields
        )
        sample["dedup_embedding"] = embedding_client.embed(
            input=sample["dedup_text"],
            model=self.model,
            dimensions=self.model_dimensions,
        )
        return sample

    def _filter_dups(
        self, sample: Dict[str, Any], idx: int, vector_db: annoy.AnnoyIndex
    ):
        indices, distances = vector_db.get_nns_by_item(idx, 2, include_distances=True)
        if distances[1] < self.threshold and indices[1] < idx:
            return False
        return True

    def map_split(
        self,
        ds_split: datasets.Dataset,
        num_proc: int,
        writer_batch_size: int,
        exclude_fields: set[str],
    ) -> datasets.Dataset:
        # 1. create the index
        global VECTOR_DB

        if VECTOR_DB is None:
            VECTOR_DB = annoy.AnnoyIndex(self.model_dimensions, "angular")

        # 2. add the dedup_text column
        ds_split_with_dedup = ds_split.map(
            self._add_dedup_text_and_embedding_columns,
            num_proc=num_proc,
            fn_kwargs={"exclude_fields": exclude_fields},
        )

        for i, sample in enumerate(ds_split_with_dedup):
            VECTOR_DB.add_item(i, sample["dedup_embedding"])

        VECTOR_DB.build(self.num_trees, n_jobs=num_proc)

        ds_split = ds_split.filter(
            self._filter_dups,
            with_indices=True,
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
            fn_kwargs={"vector_db": VECTOR_DB},
        )

        # TODO: log some of the duplicates so that we can see what is being removed

        return ds_split
