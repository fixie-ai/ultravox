from datetime import datetime
from typing import List, Optional

import huggingface_hub
import transformers

from ultravox.model import wandb_utils
from ultravox.training import config_base

ALLOW_PATTERNS = ["*.safetensors", "*.json"]


def main(override_sys_args: Optional[List[str]] = None):
    start = datetime.now()
    print("Downloading weights ...")

    args = config_base.get_train_args(override_sys_args)

    for model_id in [args.text_model, args.audio_model]:
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
                f"Model {args.text_model} not found on HF Hub. Skipping download. Error: {e}"
            )

        # A backstop to make sure the model is fully downloaded. Scenarios to consider:
        # - ALLOW_PATTERNS is not enough to download all files needed
        # - The model is local, this will verify that everything is in order
        # Using `device_map="meta"` to avoid loading the weights into memory or device
        transformers.AutoModel.from_pretrained(model_id, device_map="meta")

    if args.model_load_dir and wandb_utils.is_wandb_url(args.model_load_dir):
        wandb_utils.download_model_from_wandb(args.model_load_dir)

    end = datetime.now()
    print(f"Weights downloaded in {end - start} seconds")


def raise_on_weights_not_downloaded(model_ids: List[str]):
    """
    This function checks to see if the model weights are downloaded and available locally.
    If they are not, it raises an error.
    """
    for model_id in model_ids:
        transformers.AutoModel.from_pretrained(
            model_id, device_map="meta", local_files_only=True
        )


if __name__ == "__main__":
    main()
