import logging
from functools import wraps
from typing import Any, Type

import datasets
import huggingface_hub
import requests
import tenacity

logger = logging.getLogger(__name__)

IS_PATCHED = False


def patch_with_retry(cls: Type[Any], method_name: str, max_attempts: int = 10) -> None:
    """
    Generic function to patch any method with retry capability.

    Args:
        cls: The class containing the method to patch
        method_name: The name of the method to patch
        max_attempts: Maximum number of retry attempts (default: 10)
    """
    original_method = getattr(cls, method_name)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(max_attempts),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    @wraps(original_method)
    def method_with_retry(self, *args, **kwargs):
        return original_method(self, *args, **kwargs)

    # Apply the patch
    setattr(cls, method_name, method_with_retry)
    logger.info(
        f"Applied retry patch to {cls.__name__}.{method_name} with max_attempts={max_attempts}"
    )


def patch_hf_hub_http_backoff():
    """
    Monkey patch the huggingface_hub http_backoff implementation to include the ChunkedEncodingError exception.
    """
    original_http_backoff = huggingface_hub.hf_file_system.http_backoff

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
    )
    def http_backoff(
        method: huggingface_hub.utils._typing.HTTP_METHOD_T,
        url: str,
        *,
        max_retries: int = 10,
        retry_on_exceptions: tuple = (
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

    huggingface_hub.hf_file_system.http_backoff = http_backoff
    logger.info(
        "Applied retry patch to huggingface_hub http_backoff with ChunkedEncodingError support"
    )


def patch_audio_decoder():
    """
    Monkey-patch the datasets.Audio.decode_example method to handle errors gracefully.
    When decoding fails, returns a dict with None for array and original path.
    """
    # Store the original decode_example method
    original_decode_example = datasets.Audio.decode_example

    def safe_decode_example(self, value, token_per_repo_id=None):
        try:
            # Try to decode using the original method
            return original_decode_example(self, value, token_per_repo_id)
        except Exception as e:
            logger.warning(f"Error decoding audio at path {value.get('path')}: {e}")
            return {
                "array": None,
                "path": value.get("path", None),
                "sampling_rate": self.sampling_rate or 16000,
            }

    # Replace the original decode_example with our safe version
    datasets.Audio.decode_example = safe_decode_example
    logger.info(
        "Applied patch to datasets.Audio.decode_example for graceful error handling"
    )


def apply_all_patches():
    """
    Apply all patches at once.
    """
    global IS_PATCHED
    if IS_PATCHED:
        logger.info("Patches already applied, skipping")
        return

    logger.info("Starting to apply patches...")

    # Patch HF Hub methods
    patch_with_retry(huggingface_hub.HfApi, "dataset_info")
    patch_with_retry(huggingface_hub.HfApi, "model_info")
    patch_with_retry(huggingface_hub.HfApi, "repo_info")

    # Patch http_backoff
    patch_hf_hub_http_backoff()

    # Patch audio decoder
    patch_audio_decoder()

    # Patch datasets methods
    patch_with_retry(
        datasets.load.HubDatasetModuleFactoryWithParquetExport, "get_module"
    )

    IS_PATCHED = True
    logger.info("Successfully applied all patches")
