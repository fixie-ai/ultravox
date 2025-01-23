import os
from typing import Tuple, Type, Union

import huggingface_hub
import requests
from huggingface_hub.utils._typing import HTTP_METHOD_T

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


def patch_hf_hub_http_backoff():
    """
    Monkey patch the huggingface_hub http_backoff implementation to include the ChunkedEncodingError exception.
    """
    # We first tried patching the original http_backoff function from utils._http but due to directly importing
    # the function, the hf_file_system's version of http_backoff would not get updated, hence we're updating it
    # directly in the target module even if http_backoff itself is defined elsewhere.
    from huggingface_hub import hf_file_system

    original_http_backoff = hf_file_system.http_backoff

    def http_backoff(
        method: HTTP_METHOD_T,
        url: str,
        *,
        max_retries: int = 10,
        retry_on_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ),
        **kwargs,
    ) -> requests.Response:
        return original_http_backoff(
            method=method,
            url=url,
            max_retries=max_retries,
            retry_on_exceptions=retry_on_exceptions,
            **kwargs,
        )

    hf_file_system.http_backoff = http_backoff


patch_hf_hub_http_backoff()
