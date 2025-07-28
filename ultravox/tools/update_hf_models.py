import dataclasses
import json
import logging
import os
import tempfile
from typing import Literal

import huggingface_hub
import simple_parsing

from ultravox.model import ultravox_tokenizer

logging.basicConfig(level=logging.INFO)


@dataclasses.dataclass
class UpdateHfCodeArgs:
    """Arguments for updating Hugging Face code tool."""

    # Local path of the files to upload/update in HF Hub.
    # Usually this contains ultravox_model.py, ultravox_config.py, etc.
    local_paths: list[str] = simple_parsing.list_field(
        "ultravox/model/ultravox_model.py",
        "ultravox/model/ultravox_config.py",
        "ultravox/model/ultravox_processing.py",
        "ultravox/model/ultravox_pipeline.py",
        "ultravox/model/ultravox_tokenizer.py",
        alias="-l",
    )
    # List of config changes, e.g. "torch_dtype=bfloat16 text_model_id=..."
    config_changes: list[str] = simple_parsing.list_field(alias="-c")
    # Whether to add <|audio|> token to the tokenizer vocab
    add_audio_token: bool = simple_parsing.field(default=True, alias="-at")
    # Search query for models
    repo_path_query: str = simple_parsing.field(default="ultravox-v0_", alias="-q")
    # Whether to only upload to public models
    public_only: bool = simple_parsing.field(default=True, alias="-p")
    # Author of the models
    user_or_org: str = simple_parsing.field(default="fixie-ai", alias="-u")
    # Commit message
    commit_message: str = simple_parsing.field(default="Update code", alias="-m")
    # Sort field for models list.
    # sort and direction are to make sure the order of the models in HF Hub does not change.
    sort: str = simple_parsing.field(default="created_at")
    # Sort direction (None for ascending, -1 for descending)
    direction: Literal[-1] | None = simple_parsing.field(default=None)


def main(args: UpdateHfCodeArgs):
    api = huggingface_hub.HfApi()

    models = api.list_models(
        search=args.repo_path_query,
        author=args.user_or_org,
        sort=args.sort,
        direction=args.direction,
    )
    if args.public_only:
        models = [x for x in models if not x.private]

    for model in models:
        with tempfile.TemporaryDirectory() as temp_dir:
            operations = [
                huggingface_hub.CommitOperationAdd(
                    path_in_repo=os.path.basename(local_path),
                    path_or_fileobj=local_path,
                )
                for local_path in args.local_paths
            ]
            if args.config_changes or args.add_audio_token:
                config_operations = get_updated_config(
                    api,
                    model.id,
                    config_changes=args.config_changes,
                    temp_dir=temp_dir,
                    add_audio_token=args.add_audio_token,
                )
                operations.extend(config_operations)

            logging.info(
                f"Updating the following files for {model.id}: {', '.join([o.path_in_repo for o in operations])}"
            )

            api.create_commit(
                repo_id=model.id,
                operations=operations,
                commit_message=args.commit_message,
            )


def get_updated_config(
    api: huggingface_hub.HfApi,
    model_id: str,
    config_changes: list[str],
    temp_dir: str,
    add_audio_token: bool = True,
) -> list[huggingface_hub.CommitOperationAdd]:
    operations = []

    # download the config.json
    config_path = api.hf_hub_download(
        repo_id=model_id,
        filename="config.json",
        local_dir=temp_dir,
    )

    # load, update, and save the config.json
    with open(config_path, "r") as f:
        config = json.load(f)

    # make changes to the config and possibly the tokenizer
    for change in config_changes:
        key, value = change.split("=")
        config[key] = value

    if add_audio_token:
        print("Adding <|audio|> token to the tokenizer vocab...")
        tokenizer = ultravox_tokenizer.from_pretrained_text_tokenizer(model_id)
        config["audio_token_index"] = ultravox_tokenizer.get_audio_token_id(tokenizer)
        # save the tokenizer and add it to the operations
        files_saved = tokenizer.save_pretrained(temp_dir)
        operations.extend(
            [
                huggingface_hub.CommitOperationAdd(
                    path_in_repo=os.path.basename(file), path_or_fileobj=file
                )
                for file in files_saved
            ]
        )

    # save the config.json
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # add the config.json to the operations
    operations.append(
        huggingface_hub.CommitOperationAdd(
            path_in_repo="config.json", path_or_fileobj=config_path
        )
    )

    return operations


if __name__ == "__main__":
    args = simple_parsing.parse(UpdateHfCodeArgs)
    main(args)
