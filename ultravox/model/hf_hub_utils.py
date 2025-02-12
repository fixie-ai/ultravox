import logging
import os

import huggingface_hub

logger = logging.getLogger(__name__)

ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt"]


def is_hf_model(model_id: str) -> bool:
    return model_id.startswith("hf://")


def get_hf_model_id(model_id: str) -> str:
    if is_hf_model(model_id):
        return model_id.split("hf://")[1]
    return model_id


def download_hf_model(model_id: str, use_hf_transfer: bool = False) -> str:
    """
    Download the model from HF Hub.

    The model_id can be of format "hf://<repo_id>" to disambiguate from a local path, but <repo_id> is also accepted.
    """
    model_id = get_hf_model_id(model_id)

    if use_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    return huggingface_hub.snapshot_download(
        repo_id=model_id, allow_patterns=ALLOW_PATTERNS
    )


def download_file_from_hf_hub(model_id: str, file_path: str) -> str:
    model_id = get_hf_model_id(model_id)
    return huggingface_hub.hf_hub_download(repo_id=model_id, filename=file_path)
