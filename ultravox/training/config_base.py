import dataclasses
import datetime
import logging
import os
from pathlib import Path
from typing import List, Optional

import simple_parsing

from ultravox.data import datasets
from ultravox.model import ultravox_config
from ultravox.utils import device_helpers


@dataclasses.dataclass
class TrainConfig:
    """Configuration class for training settings."""

    text_model: str
    """Language model to use; could be a huggingface model id, wandb path, or local path."""

    audio_model: str
    """Audio encoder model to use; could be a huggingface model id, wandb path, or local path."""

    # train_dataset_configs: List[Dict[str, Any]] = dataclasses.field(
    #     default_factory=list
    # )
    train_dataset_configs: List[datasets.DatasetConfig] = dataclasses.field(
        default_factory=list
    )
    """
    List of training dataset configurations.
    Initially parsed as dictionaries (due to limitations of simple_parsing), then converted to DatasetConfig objects in __post_init__.
    """

    val_dataset_configs: List[datasets.DatasetConfig] = dataclasses.field(
        default_factory=list
    )
    """
    List of validation dataset configurations.
    Initially parsed as dictionaries (due to limitations of simple_parsing), then converted to DatasetConfig objects in __post_init__.
    """

    eval_dataset_configs: List[datasets.DatasetConfig] = dataclasses.field(
        default_factory=list
    )
    """
    List of evaluation dataset configurations.
    Initially parsed as dictionaries (due to limitations of simple_parsing), then converted to DatasetConfig objects in __post_init__.
    """

    train_dataset_args: datasets.VoiceDatasetArgs = dataclasses.field(
        default_factory=datasets.VoiceDatasetArgs
    )
    """Global arguments for the training dataset."""

    val_dataset_args: datasets.VoiceDatasetArgs = dataclasses.field(
        default_factory=datasets.VoiceDatasetArgs
    )
    """Global arguments for the validation dataset."""

    eval_dataset_args: datasets.VoiceDatasetArgs = dataclasses.field(
        default_factory=datasets.VoiceDatasetArgs
    )
    """Global Arguments for the evaluation dataset."""

    do_train: bool = True
    """Whether to perform training."""

    do_eval: bool = True
    """Whether to perform evaluation."""

    stop_strategy: datasets.StopStrategy = datasets.StopStrategy.LAST_EXHAUSTED
    """
    The stop strategy for InterleaveDataset when combining multiple datasets for training, 
    choose from last_exhausted (default), first_exhausted, or never_stop (rely on max_steps or num_epochs to stop, and should be used as the default.
    """

    num_workers: int = device_helpers.get_world_size()
    """Number of data loader workers."""

    train_on_inputs: bool = False
    """Whether to train on inputs."""

    verbose: bool = False
    """Whether to enable verbose output."""

    device: str = device_helpers.default_device()
    """Device to use for training (e.g., 'cuda', 'cpu', 'mps')."""

    data_type: str = device_helpers.default_dtype_str()
    """Data type to use for training (e.g., 'bfloat16', 'float16', 'float32')."""

    model_load_dir: Optional[str] = None
    """
    Path to load pretrained ultravox model from. Can be local path, HF hub model_id, or W&B artifact.
    """

    text_model_lora_config: Optional[ultravox_config.LoraConfigSimplified] = None
    """LoRA configuration for the text model."""

    audio_model_lora_config: Optional[ultravox_config.LoraConfigSimplified] = None
    """LoRA configuration for the audio model."""

    disable_layerdrop: bool = False
    """Whether to disable layerdrop."""

    exp_name: Optional[str] = None
    """The experiment name."""

    output_dir: Optional[Path] = None
    """Directory to save output files."""

    logs_dir: Optional[Path] = None
    """Directory to save log files."""

    optimizer: str = "adamw_torch"
    """Optimizer to use for training."""

    num_epochs: int = 1
    """Number of training epochs, only used when max_steps is 0."""

    max_steps: int = 0
    """Maximum number of training steps, if 0, use num_epochs."""

    val_steps: Optional[int] = None
    """Number of steps between validations."""

    save_steps: float = 0
    """Number of steps between model saves."""

    logging_steps: int = 1
    """Number of steps between logging."""

    grad_accum_steps: int = 1
    """Number of gradient accumulation steps."""

    val_accum_steps: int = 1
    """Number of validation accumulation steps."""

    lr: float = 1e-5
    """Learning rate."""

    lr_scheduler: str = "cosine"
    """Learning rate scheduler type."""

    lr_warmup_steps: int = 0
    """Number of learning rate warmup steps."""

    weight_decay: float = 0.0
    """Weight decay for optimizer."""

    seed: int = 42
    """Random seed for reproducibility."""

    report_logs_to: List[str] = simple_parsing.list_field("tensorboard")
    """Experiment logging destinations: tensorboard, wandb, etc."""

    run_tags: List[str] = simple_parsing.list_field()
    """A list of tags for filtering runs. Only used for wandb."""

    loss_config: ultravox_config.LossConfig = dataclasses.field(
        default_factory=ultravox_config.LossConfig
    )
    """Configuration for the loss function."""

    def __post_init__(self):
        self.train_dataset_configs = [
            datasets.DatasetConfig(**data_dict)
            for data_dict in self.train_dataset_configs
        ]
        self.val_dataset_configs = [
            datasets.DatasetConfig(**data_dict)
            for data_dict in self.val_dataset_configs
        ]
        self.eval_dataset_configs = [
            datasets.DatasetConfig(**data_dict)
            for data_dict in self.eval_dataset_configs
        ]

        assert self.data_type in ["bfloat16", "float16", "float32"]
        assert self.device in ["cuda", "cpu", "mps"]
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
