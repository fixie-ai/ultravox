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
    # ---------------------------------------------------------------------------
    # Model parameters
    # ---------------------------------------------------------------------------
    # Text and audio models
    text_model: str
    audio_model: str
    model_type: str = "ultravox"
    # Path to load model checkpoint (local/HF/W&B)
    model_load_dir: Optional[str] = None
    # If True, the optimizer and scheduler states are also loaded from model_load_dir
    # and training is resumed. o/w only the model weights are loaded if present.
    resume_from_load_dir: bool = False
    # When resuming from a checkpoint, we can skip the same number of samples as in the previous run.
    # If False, this makes sure that we get the same data loading as if we had not interrupted the training.
    # If True, this allows the training to begin faster (as that skipping step can take a long time). In this case,
    # it's recommended to change the train dataset seed, which train.py will do automatically.
    ignore_data_skip: bool = True

    # LoRA configs
    text_model_lora_config: Optional[ultravox_config.LoraConfigSimplified] = None
    audio_model_lora_config: Optional[ultravox_config.LoraConfigSimplified] = None

    # ---------------------------------------------------------------------------
    # Ultravox-specific parameters
    # ---------------------------------------------------------------------------
    # Ultravox up to v0.4.1 had layer_norm after the second (final) linear layer in the projector.
    # v0.5.0 and above have layer_norm after the first linear layer in the projector.
    projector_ln_mid: bool = True
    # Disable layerdrop
    disable_layerdrop: bool = False
    # Audio latency block size (e.g., 100 for 1s, etc.)
    audio_latency_block_size: Optional[int] = None

    # ---------------------------------------------------------------------------
    # LSM-specific parameters
    # ---------------------------------------------------------------------------
    # Expected audio length in seconds
    expected_audio_length_seconds: float = 10

    # ---------------------------------------------------------------------------
    # Dataset parameters
    # ---------------------------------------------------------------------------
    # Datasets as dicts (converted to list of DatasetConfig later)
    data_sets: List[Dict[str, Any]] = simple_parsing.list_field()
    # Train/Val/Eval sets as dicts (converted to list of DatasetOptions later)
    train_sets: List[Dict[str, Any]] = simple_parsing.list_field()
    val_sets: List[Dict[str, Any]] = simple_parsing.list_field()
    eval_sets: List[Dict[str, Any]] = simple_parsing.list_field()

    # Dataset args
    train_dataset_args: datasets.TrainDatasetArgs = simple_parsing.field(
        default_factory=datasets.TrainDatasetArgs
    )
    val_dataset_args: datasets.ValDatasetArgs = simple_parsing.field(
        default_factory=datasets.ValDatasetArgs
    )
    eval_dataset_args: datasets.EvalDatasetArgs = simple_parsing.field(
        default_factory=datasets.EvalDatasetArgs
    )

    # ---------------------------------------------------------------------------
    # Experiment and output parameters
    # ---------------------------------------------------------------------------
    exp_name: Optional[str] = None
    output_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None

    # Run modes
    do_train: bool = True
    do_eval: bool = True

    # Dataloader workers
    num_workers: int = 8 if torch.cuda.is_available() else 1
    # Training sample control
    train_on_inputs: bool = False
    # assistant response is truncated to avoid OOM errors
    max_response_tokens: Optional[int] = 50

    # Device and dtype
    device: str = "cuda"
    data_type: str = "bfloat16"
    # Seed for reproducibility
    seed: int = 42

    # Use Fully Sharded Data Parallelism
    use_fsdp: bool = False

    # ---------------------------------------------------------------------------
    # Training parameters
    # ---------------------------------------------------------------------------
    # Batch/step settings
    batch_size: int = 4
    grad_accum_steps: int = 1
    num_epochs: int = 1
    max_steps: int = 0  # overrides num_epochs if > 0

    # Optimizer and scheduler
    optimizer: str = "adamw_torch"
    lr: float = 1e-5
    lr_scheduler: str = "cosine"  # options: linear, cosine, cosine_with_restarts, etc.
    lr_scheduler_kwargs: Dict[str, Any] = simple_parsing.field(default_factory=dict)
    lr_warmup_steps: int = 0
    weight_decay: float = 0.0

    # Loss config
    loss_config: Optional[ultravox_config.LossConfig] = None

    # ---------------------------------------------------------------------------
    # Validation, saving, and logging
    # ---------------------------------------------------------------------------
    # If val_steps < 1, treated as fraction of total steps
    val_steps: Optional[float] = None
    val_accum_steps: int = 1
    # If save_steps < 1, treated as fraction of total steps
    save_steps: float = 0
    # Log steps
    logging_steps: int = 1

    # Logging destinations: tensorboard, wandb, etc.
    report_logs_to: List[str] = simple_parsing.list_field("tensorboard")
    # Tags for wandb
    run_tags: List[str] = simple_parsing.list_field()

    verbose: bool = False

    # ---------------------------------------------------------------------------
    # Evaluation parameters
    # ---------------------------------------------------------------------------
    eval_batch_size: int = 4
    eval_max_tokens: int = 512
    eval_temperature: float = 0.0

    # ---------------------------------------------------------------------------
    # Methods to convert dataset configs
    # ---------------------------------------------------------------------------
    def get_data_sets(self) -> List[datasets.DatasetConfig]:
        return [datasets.DatasetConfig.from_dict(ds) for ds in self.data_sets]

    def get_train_sets(self) -> List[datasets.DatasetOptions]:
        return [datasets.DatasetOptions(**ds) for ds in self.train_sets]

    def get_val_sets(self) -> List[datasets.DatasetOptions]:
        return [datasets.DatasetOptions(**ds) for ds in self.val_sets]

    def get_eval_sets(self) -> List[datasets.DatasetOptions]:
        return [datasets.DatasetOptions(**ds) for ds in self.eval_sets]

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

        if self.resume_from_load_dir:
            assert bool(
                self.model_load_dir
            ), "model_load_dir must be set if resume_from_load_dir is True"

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

        if self.use_fsdp and self.do_eval:
            logging.warning(
                "FSDP is enabled: Evaluation is not supported with FSDP. Disabling evaluation."
            )
            self.do_eval = False


def fix_hyphens(arg: str):
    return re.sub(r"^--([^=]+)", lambda m: "--" + m.group(1).replace("-", "_"), arg)


def get_train_config(
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
