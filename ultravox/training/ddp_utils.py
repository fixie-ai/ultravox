import contextlib

import torch.distributed


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
