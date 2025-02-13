import dataclasses
import json
import os
import tempfile

import simple_parsing
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download


@dataclasses.dataclass
class ModelModifierArgs:
    # Original model ID on Hugging Face
    model_id: str
    # New text_model_id to set in config.json
    new_text_model_id: str
    # Hugging Face model ID to upload to
    new_model_id: str
    # Whether to push to hub (default: True)
    push_to_hub: bool = True
    # Private model (default: True)
    private: bool = True


def modify_and_reupload_model(args: ModelModifierArgs):
    """
    Downloads a model from Hugging Face, modifies its config.json, and reuploads it

    Args:
        model_id: Original model ID on Hugging Face (e.g., "organization/model-name")
        new_text_model_id: New text_model_id to set in config.json
        new_model_id: New model ID to upload to (e.g., "organization/new-model-name")
    """
    # Get token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable must be set")

    # Initialize Hugging Face API
    api = HfApi()

    # Use tempfile instead of manual directory management
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Download the model
            print(f"Downloading model {args.model_id}...")
            model_path = snapshot_download(repo_id=args.model_id, local_dir=temp_dir)

            # Modify config.json
            config_path = os.path.join(model_path, "config.json")
            print("Modifying config.json...")

            with open(config_path, "r") as f:
                config = json.load(f)

            # Update the text_model_id
            config["text_model_id"] = args.new_text_model_id

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            if args.push_to_hub:
                # Upload the modified model
                print(f"Uploading modified model to {args.new_model_id}...")
                api.create_repo(
                    repo_id=args.new_model_id, private=args.private, exist_ok=True
                )

                api.upload_folder(
                    folder_path=model_path,
                    repo_id=args.new_model_id,
                    commit_message="Updated text_model_id in config.json",
                )

                print("Success! Model has been modified and uploaded.")
            else:
                print(f"Model modified and saved to {model_path}")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise


def main():
    args = simple_parsing.parse(ModelModifierArgs)
    modify_and_reupload_model(args)


if __name__ == "__main__":
    main()
