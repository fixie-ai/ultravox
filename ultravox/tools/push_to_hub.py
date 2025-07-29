import dataclasses
import logging
import os
import shutil
import tempfile
from typing import Optional

import simple_parsing
import transformers
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

from ultravox.model import file_utils
from ultravox.model import ultravox_model
from ultravox.model import ultravox_pipeline
from ultravox.utils import device_helpers


# This script is used to upload a model to the HuggingFace Hub, for either internal or external consumption.
# Ex: python -m ultravox.tools.push_to_hub -m wandb://fixie/ultravox/<model_path> -u fixie-ai/ultravox-vXYZ
@dataclasses.dataclass
class UploadToHubArgs:
    # Model ID to use for the model
    model: str = simple_parsing.field(alias="-m")
    # HuggingFace Hub model_id to push to
    hf_upload_model: str = simple_parsing.field(alias="-u")
    # Only the llm for finetuned models
    text_only: bool = simple_parsing.field(default=False, alias="-t")
    # Device to use for the model
    device: Optional[str] = simple_parsing.field(
        default=device_helpers.default_device(), alias="-D"
    )
    # Data type to use for the model
    data_type: Optional[str] = None
    # Public or private (default)
    private: bool = True
    # Verify the model after uploading
    verify: bool = False
    chat_template: Optional[str] = simple_parsing.field(default=None, alias="-c")

    def __post_init__(self):
        if self.chat_template and self.chat_template.startswith("file://"):
            file_path = self.chat_template[7:].strip()  # Remove "file://" prefix
            try:
                with open(file_path, "r") as f:
                    self.chat_template = f.read()
            except Exception as e:
                raise ValueError(
                    f"Failed to load chat template from file {file_path}: {e}"
                )


def main(args: UploadToHubArgs):
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load the model and tokenizer, then merge LoRA weights if they exist
    model_path = file_utils.download_dir_if_needed(args.model)
    dtype = device_helpers.get_dtype(args.data_type)
    model = ultravox_model.UltravoxModel.from_pretrained(model_path, torch_dtype=dtype)
    model.merge_and_unload()

    if args.text_only:
        text_llm = model.language_model
        tokenizer_repo = model.config.text_model_id

        print("Preparing text language model with tokenizer for upload...")
        with tempfile.TemporaryDirectory() as temp_dir:
            text_llm.save_pretrained(temp_dir)

            tokenizer_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ]

            for file in tokenizer_files:
                try:
                    downloaded_file = hf_hub_download(
                        repo_id=tokenizer_repo, filename=file
                    )
                    target_path = os.path.join(temp_dir, file)
                    shutil.copy2(downloaded_file, target_path)
                except Exception as e:
                    print(
                        f"Warning: Could not download {file} from {tokenizer_repo}: {e}"
                    )

            # Upload the combined model
            print("Uploading text model with tokenizer to HuggingFace Hub...")
            api = HfApi()
            # Create the repository if it doesn't exist
            api.create_repo(
                repo_id=args.hf_upload_model, private=args.private, exist_ok=True
            )
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=args.hf_upload_model,
                commit_message=f"Upload text model with tokenizer from {tokenizer_repo}",
            )
        return

    pipe = ultravox_pipeline.UltravoxPipeline(
        model=model,
        device=args.device,
        chat_template=args.chat_template,
    )

    print("Uploading model to HuggingFace Hub...")
    pipe.push_to_hub(args.hf_upload_model, private=args.private)

    if args.verify:
        from ultravox import data as datasets

        print("Model uploaded. Testing model...")
        loaded_pipe = transformers.pipeline(
            model=args.hf_upload_model, trust_remote_code=True
        )
        ds = datasets.create_dataset("boolq", datasets.VoiceDatasetArgs())
        sample = next(iter(ds))
        generated = loaded_pipe(
            {"audio": sample.audio, "turns": sample.messages[:-1]}, max_new_tokens=10
        )
        print(f"Generated (max 10 tokens): {generated}")


if __name__ == "__main__":
    main(simple_parsing.parse(UploadToHubArgs))
