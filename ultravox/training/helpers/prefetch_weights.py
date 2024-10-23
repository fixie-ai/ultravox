from datetime import datetime
from typing import List, Optional

import huggingface_hub
import transformers

from ultravox.model import wandb_utils
from ultravox.training import config_base

ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt"]


def main(override_sys_args: Optional[List[str]] = None):
    start = datetime.now()
    print("Downloading weights ...")

    args = config_base.get_train_args(override_sys_args)

    download_weights([args.text_model, args.audio_model], args.model_load_dir)

    end = datetime.now()
    print(f"Weights downloaded in {end - start} seconds")


def download_weights(
    model_ids: List[str],
    model_load_dir: Optional[str] = None,
    include_models_from_load_dir: bool = False,
) -> Optional[str]:
    """
    Fast download for the weights of the given model_ids, and the model_load_dir if it's a wandb url.
    It uses HF Hub's snapshot_download to download multiple files in parallel which is faster than .from_pretrained.

    It can potentially try to download the weights for the text_ and audio_ models used in the model_load_dir as well.

    Args:
        model_ids: List of model ids to download
        model_load_dir: The model load directory. Can be a wandb url:
            Ex: "wandb://fixie/ultravox/<model_path>"
        include_models_from_load_dir: If True, include the models from the load directory
            For training this behaviour is not needed since the parent models can be swapped out in some cases,
            but for inference we need the exact same models.

    Returns:
        model_path: The path to the model if it was downloaded from wandb
    """
    model_path: Optional[str] = None

    if model_load_dir and wandb_utils.is_wandb_url(model_load_dir):
        model_path = wandb_utils.download_model_from_wandb(model_load_dir)
        if include_models_from_load_dir:
            run_config = wandb_utils.get_run_config_from_artifact(model_load_dir)
            if run_config is not None:
                for key in ["text_model", "audio_model"]:
                    if key in run_config:
                        model_ids.append(run_config[key])

    for model_id in model_ids:
        try:
            # Download all model files that match ALLOW_PATTERNS
            # This is faster than .from_pretrained due to parallel downloads
            huggingface_hub.snapshot_download(
                repo_id=model_id, allow_patterns=ALLOW_PATTERNS
            )
        except huggingface_hub.utils.GatedRepoError as e:
            raise e
        except huggingface_hub.utils.RepositoryNotFoundError as e:
            # We assume that the model is local if it's not found on HF Hub.
            # The `.from_pretrained` call will verify the local case.
            print(
                f"Model {model_id} not found on HF Hub. Skipping download. Error: {e}"
            )

        # A backstop to make sure the model is fully downloaded. Scenarios to consider:
        # - ALLOW_PATTERNS is not enough to download all files needed
        # - The model is local, this will verify that everything is in order
        # Using `device_map="meta"` to avoid loading the weights into memory or device
        transformers.AutoModel.from_pretrained(model_id, device_map="meta")

    return model_path


if __name__ == "__main__":
    main()
