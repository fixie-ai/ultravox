import contextlib
import logging
from typing import Any, Generator, List, Tuple, TypeVar

import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
import transformers
from torch.utils import data

from ultravox.utils import device_helpers


@contextlib.contextmanager
def run_on_master_first(is_master: bool | None = None):
    """
    If using DDP, allows the master process to run the enclosed code first.
    This is useful when only one process should download a model or other resources first to avoid race conditions.
    """
    if is_master is None:
        is_master = device_helpers.is_local_master()

    if is_master:
        yield
        if dist.is_initialized():
            dist.barrier()
    else:
        # All other processes wait for the master to download the model first
        if dist.is_initialized():
            dist.barrier()
        yield


T = TypeVar("T")


def flatten(data: List[List[T]]) -> List[T]:
    return [item for sublist in data for item in sublist]


def all_gather_list(data: List[T]) -> List[T]:
    if not dist.is_initialized():
        return data
    world_size = dist.get_world_size()
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
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


def fsdp_basic_wrap(
    model: transformers.PreTrainedModel, modules_to_wrap: list[str] | None = None
) -> transformers.PreTrainedModel:
    if modules_to_wrap is None:
        if isinstance(model._no_split_modules, list):
            modules_to_wrap = model._no_split_modules
        else:
            raise ValueError("modules_to_wrap is not provided and not found in model")
        # assert modules_to_wrap is not None

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    torch.cuda.set_device(torch.device(rank))

    logging.info(f"[rank {rank}] wrapping model with FSDP, world_size: {world_size}")
    mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))

    for submodule in model.modules():
        if submodule.__class__.__name__ in modules_to_wrap:
            fsdp.fully_shard(submodule, mesh=mesh)

    model = fsdp.fully_shard(model, mesh=mesh)

    # make sure the generate method knows to load the weights from the FSDP wrapper
    fsdp.register_fsdp_forward_method(model, "generate")

    return model


def model_to_device(
    model: transformers.PreTrainedModel,
    device: str,
    use_fsdp: bool = False,
    use_tp: bool = False,
) -> transformers.PreTrainedModel:
    if use_fsdp:
        return fsdp_basic_wrap(model)
    elif use_tp:
        return model
    else:
        return model.to(device)
