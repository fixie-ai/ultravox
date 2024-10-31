import huggingface_hub
from huggingface_hub import file_download
from huggingface_hub import hf_file_system


def _fetch_range(self, start: int, end: int) -> bytes:
    """
    This is a copy of the original _fetch_range method from HfFileSystemFile.
    The only modification is the addition of the 500 status code to the retry_on_status_codes tuple.

    Original source code:
    https://github.com/huggingface/huggingface_hub/blob/c0fd4e0f7519a4e3659c836081cc7e38c0d14b35/src/huggingface_hub/hf_file_system.py#L717
    """
    headers = {
        "range": f"bytes={start}-{end - 1}",
        **self.fs._api._build_hf_headers(),
    }
    url = file_download.hf_hub_url(
        repo_id=self.resolved_path.repo_id,
        revision=self.resolved_path.revision,
        filename=self.resolved_path.path_in_repo,
        repo_type=self.resolved_path.repo_type,
        endpoint=self.fs.endpoint,
    )
    r = hf_file_system.http_backoff(
        "GET",
        url,
        headers=headers,
        retry_on_status_codes=(500, 502, 503, 504),  # add 500 to retry on server errors
        timeout=huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT,
    )
    hf_file_system.hf_raise_for_status(r)
    return r.content


def monkey_patch_fetch_range():
    import huggingface_hub

    huggingface_hub.HfFileSystemFile._fetch_range = _fetch_range