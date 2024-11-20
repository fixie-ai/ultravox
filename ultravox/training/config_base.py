import dataclasses
import datetime
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import simple_parsing
import torch

from ultravox import data as datasets
from ultravox.model import ultravox_config


@dataclasses.dataclass
class DatasetOptions:
    name: str
    weight: float = 1.0
    include_audio: bool = True


@dataclasses.dataclass
class TrainConfig:
    # Language model to use
    text_model: str
    # Audio encoder model to use
    audio_model: str

    # Workaround for simple_parsing not supporting lists of dataclasses; we need to
    # define these as lists of dicts and convert them manually in helpers.

    # Data-defined datasets (datasets.DatasetConfig)
    data_sets: List[Dict[str, Any]] = simple_parsing.list_field()
    # Training sets and weights (DatasetOptions)
    train_sets: List[Dict[str, Any]] = simple_parsing.list_field()
    # Validation sets and weights (DatasetOptions)
    val_sets: List[Dict[str, Any]] = simple_parsing.list_field()

    def get_data_sets(self) -> List[datasets.DatasetConfig]:
        return [datasets.DatasetConfig.from_dict(ds) for ds in self.data_sets]

    def get_train_sets(self) -> List[DatasetOptions]:
        return [DatasetOptions(**ds) for ds in self.train_sets]

    def get_val_sets(self) -> List[DatasetOptions]:
        return [DatasetOptions(**ds) for ds in self.val_sets]

    do_train: bool = True
    do_eval: bool = True

    num_samples: Optional[int] = None
    val_num_samples: int = 100
    eval_num_samples: int = 100
    eval_max_new_tokens: Optional[int] = None
    eval_num_procs: int = 8
    eval_text_only: bool = False
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
    # Whether to use FSDP (Fully Sharded Data Parallelism) for training
    # needed for large model training (e.g. 70B+)
    use_fsdp: bool = False
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
    # Run an evaluation every X steps. If smaller than 1, will be interpreted as ratio of total training steps.
    val_steps: Optional[float] = None
    # Save checkpoint every X steps. If smaller than 1, will be interpreted as ratio of total training steps.
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
    # A list of tags for filtering runs. Only used for wandb.
    run_tags: List[str] = simple_parsing.list_field()

    # loss function to use
    loss_config: Optional[ultravox_config.LossConfig] = None

    # To simulate audio streaming with masking. None for non-causal, 100 for 1s, 200 for 2s, and so on.
    audio_latency_block_size: Optional[int] = None

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

        # HF Pipeline gets tripped up if the path has a "." in it
        self.output_dir = Path(str(self.output_dir).replace(".", "--"))

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

        if self.use_fsdp and self.save_steps:
            logging.warning(
                "FSDP is enabled: Saving checkpoints is going to be extremely slow and results in a full save."
                " Consider setting save_steps=0."
            )

        if self.use_fsdp and self.do_eval:
            logging.warning(
                "FSDP is enabled: Evaluation is not supported with FSDP. Disabling evaluation."
            )
            self.do_eval = False


def fix_hyphens(arg: str):
    return re.sub(r"^--([^=]+)", lambda m: "--" + m.group(1).replace("-", "_"), arg)


def get_train_args(
    override_sys_args: Optional[List[str]] = None, config_file="meta_config.yaml"
) -> TrainConfig:
    """
    Parse the command line arguments and return a TrainConfig object.

    Args:
        override_sys_args: The command line arguments. If None, sys.argv[1:] is used.
            This is mainly useful for testing.
    """
    args = sys.argv[1:] if override_sys_args is None else override_sys_args

    return simple_parsing.parse(
        config_class=TrainConfig,
        config_path=os.path.join(os.path.dirname(__file__), "configs", config_file),
        add_config_path_arg=True,
        args=[fix_hyphens(arg) for arg in args],
    )
