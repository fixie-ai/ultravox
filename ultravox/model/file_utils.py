from ultravox.model import hf_hub_utils
from ultravox.model import wandb_utils


def download_file_if_needed(path: str, file_name: str) -> str:
    if wandb_utils.is_wandb_url(path):
        path = wandb_utils.download_file_from_wandb(path, file_name)
    elif hf_hub_utils.is_hf_model(path):
        path = hf_hub_utils.download_file_from_hf_hub(path, file_name)
    return path


def download_dir_if_needed(load_path: str) -> str:
    if wandb_utils.is_wandb_url(load_path):
        # We assume that the weights are already downloaded via prefetch_weights.py
        # and hence this is just resolving the path. If the weights are not downloaded,
        # we might see a race condition here when using DDP.
        load_path = wandb_utils.download_model_from_wandb(load_path)
    elif hf_hub_utils.is_hf_model(load_path):
        load_path = hf_hub_utils.download_hf_model(load_path)
    return load_path
