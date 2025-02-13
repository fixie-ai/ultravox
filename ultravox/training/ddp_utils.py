import contextlib
from typing import Any, Generator, List, Tuple, TypeVar

import torch.distributed
from torch.utils import data


@contextlib.contextmanager
def run_on_master_first(is_master: bool):
    """
    If using DDP, allows the master process to run the enclosed code first.
    This is useful when only one process should download a model or other resources first to avoid race conditions.
    """
    if is_master:
        yield
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    else:
        # All other processes wait for the master to download the model first
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        yield


T = TypeVar("T")


def flatten(data: List[List[T]]) -> List[T]:
    return [item for sublist in data for item in sublist]


def all_gather_list(data: List[T]) -> List[T]:
    if not torch.distributed.is_initialized():
        return data
    world_size = torch.distributed.get_world_size()
    data_list = [None] * world_size
    torch.distributed.all_gather_object(data_list, data)
    return flatten(data_list)  # type: ignore


def sharded_iterator(ds: data.IterableDataset, num_shards: int, shard_index: int):
    # TODO: handle drop last gracefully
    for i, sample in enumerate(ds):
        if i % num_shards == shard_index:
            yield sample


def sharded_batch_iterator(
    ds: data.IterableDataset, batch_size: int, num_shards: int, shard_index: int
) -> Generator[List[Tuple[int, Any]], None, None]:
    batch = []  # List of tuples
    for idx, sample in enumerate(ds):
        if idx % num_shards == shard_index:
            batch.append((idx, sample))  # Tuple of (int, sample)
            if len(batch) == batch_size:
                yield batch  # Yields List[Tuple[int, sample]]
                batch = []
    # Yield any remaining samples in the last incomplete batch
    if batch:
        yield batch
