import os

import pytest
import torch.distributed
from torch import multiprocessing as mp

from ultravox.training import ddp_utils

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"


@pytest.mark.skipif(True, reason="Temporarily disabled for debugging purposes")
def test_all_gather_list():
    # Test without DDP
    verify_all_gather(0, 1)
    # Test with DDP: world_size = 2, 4
    mp.spawn(verify_all_gather, args=(2,), nprocs=2, join=True)
    mp.spawn(verify_all_gather, args=(4,), nprocs=4, join=True)


def verify_all_gather(rank: int, world_size: int, k: int = 4):
    if world_size > 1:
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    d = [rank * k + i for i in range(k)]
    all_d = ddp_utils.all_gather_list(d)
    assert all_d == list(range(world_size * k))
    if world_size > 1:
        torch.distributed.destroy_process_group()
