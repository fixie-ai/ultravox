from datetime import datetime
from typing import List, Optional

import huggingface_hub

from ultravox.model import wandb_utils
from ultravox.training import config_base

ALLOW_PATTERNS = ["*.safetensors", "*.json"]


def main(args: Optional[List[str]] = None):
    start = datetime.now()
    print("Downloading weights ...")

    args = config_base.get_train_args(args)

    for model_id in [args.text_model, args.audio_model]:
        try:
            huggingface_hub.snapshot_download(
                repo_id=model_id, allow_patterns=ALLOW_PATTERNS
            )
        except huggingface_hub.utils.GatedRepoError as e:
            raise e
        except huggingface_hub.utils.RepositoryNotFoundError as e:
            print(
                f"Model {args.text_model} not found on HF Hub. Skipping download. Error: {e}"
            )

    if args.model_load_dir and wandb_utils.is_wandb_url(args.model_load_dir):
        wandb_utils.download_model_from_wandb(args.model_load_dir)

    end = datetime.now()
    print(f"Weights are downloaded in {end - start} seconds")


if __name__ == "__main__":
    main()
