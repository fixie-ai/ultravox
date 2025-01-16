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


def patch_hf_hub_http_backoff():
    """
    Monkey patch the huggingface_hub http_backoff implementation to include the ChunkedEncodingError exception.
    """
    from huggingface_hub.utils import _http

    original_http_backoff = _http.http_backoff

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

    _http.http_backoff = http_backoff


patch_hf_hub_http_backoff()
