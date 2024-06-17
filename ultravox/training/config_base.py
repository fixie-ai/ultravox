import dataclasses
import datetime
import logging
import os
from pathlib import Path
from typing import List, Optional

import simple_parsing
import torch

from ultravox.model import ultravox_config


@dataclasses.dataclass
class TrainConfig:
    data_sets: List[str]
    # language model to use
    text_model: str
    # audio encoder model to use
    audio_model: str

    do_train: bool = True
    do_eval: bool = True

    # In InterleaveDataset, if one dataset runs out, should we repeat it to keep
    # the ratio of samples from each dataset fixed?
    repeat_data: bool = False
    data_dir: Optional[str] = None
    mds: bool = False
    num_samples: Optional[int] = None
    val_num_samples: int = 100
    eval_num_samples: int = 100
    eval_max_new_tokens: Optional[int] = None
    eval_num_procs: int = 8
    num_prompts: int = 1
    # number of data loader workers
    num_workers: int = 8 if torch.cuda.is_available() else 1
    train_on_inputs: bool = False
    shuffle_data: bool = False
    # Maximum audio duration in seconds. Samples with longer audio will be skipped.
    # This is usually due to GPU memory constraints and also dependends on the dataset.
    max_audio_duration_secs: Optional[float] = None

    verbose: bool = False

    device: str = "cuda"
    data_type: str = "bfloat16"
    # Path to load the model from. Can be local path, HF hub model_id, or W&B artifact
    model_load_dir: Optional[str] = None
    text_model_lora_config: Optional[ultravox_config.LoraConfigSimplified] = None
    audio_model_lora_config: Optional[ultravox_config.LoraConfigSimplified] = None
    disable_layerdrop: bool = False

    # The experiment name
    exp_name: Optional[str] = None
    output_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    optimizer: str = "adamw_torch"
    num_epochs: int = 1
    max_steps: int = 0
    val_steps: Optional[int] = None
    save_steps: float = 0
    logging_steps: int = 1
    grad_accum_steps: int = 1
    val_accum_steps: int = 1
    batch_size: int = 2
    lr: float = 1e-5
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 0
    weight_decay: float = 0.0
    seed: int = 42
    shuffle_seed: int = 42
    # Experiment logging destinations: tensorboard, wandb, neptune, mlflow, etc
    report_logs_to: List[str] = simple_parsing.list_field("tensorboard")

    def __post_init__(self):
        assert self.data_type in ["bfloat16", "float16", "float32"]
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device != "cuda":
            if self.data_type == "bfloat16":
                self.data_type = "float32"
            if self.optimizer == "adamw_bnb_8bit":
                logging.warning(
                    "Using CPU with adamw_bnb_8bit is not supported. Switching to adamw_torch"
                )
                self.optimizer = "adamw_torch"

        if self.exp_name is None:
            self.exp_name = datetime.datetime.now().strftime("exp--%Y-%m-%d--%H-%M-%S")
        if self.output_dir is None:
            self.output_dir = Path("runs") / self.exp_name
        if self.logs_dir is None:
            self.logs_dir = self.output_dir / "logs"

        if (
            self.audio_model_lora_config is not None
            and self.audio_model_lora_config.r > 0
            and os.environ.get("WORLD_SIZE", None) is not None
            and self.disable_layerdrop is False
        ):
            logging.warning(
                "LayerDrop cannot be used in DDP when encoder is not frozen. Disabling LayerDrop."
            )
            self.disable_layerdrop = True
