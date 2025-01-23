import os
from typing import Optional

import wandb

WANDB_PREFIX = "wandb://"
IGNORE_PATHS = ["pytorch_model_fsdp.bin"]


def is_wandb_url(model_path: str) -> bool:
    return model_path.startswith(WANDB_PREFIX)


def get_artifact(model_url: str) -> wandb.Artifact:
    assert is_wandb_url(model_url)
    api = wandb.Api()
    # example artifact name: "fixie/ultravox/model-llama2_asr_gigaspeech:v0"
    model_url = model_url[len(WANDB_PREFIX) :]
    return api.artifact(model_url)


def download_file_from_wandb(model_url: str, file_path: str) -> str:
    artifact = get_artifact(model_url)
    path = artifact.download(path_prefix=file_path)
    return os.path.join(path, file_path)


def download_model_from_wandb(model_url: str) -> str:
    artifact = get_artifact(model_url)

    if any(
        file.name.endswith(path) for file in artifact.files() for path in IGNORE_PATHS
    ):
        # downloading one by one to avoid downloading the ignored files
        for file in artifact.files():
            if not any(file.name.endswith(path) for path in IGNORE_PATHS):
                print("downloading", file.name)
                model_path = artifact.download(path_prefix=file.name)
    else:
        model_path = artifact.download()

    if model_path is None:
        raise ValueError(f"No files to be downloaded.")
    return model_path


def get_run_config_from_artifact(model_url: str) -> Optional[wandb.sdk.Config]:
    artifact = get_artifact(model_url)
    run = artifact.logged_by()
    if run is None:
        return None
    return run.config
