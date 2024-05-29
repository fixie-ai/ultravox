import logging
import typing as t
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import torch

from . import data
from .models import multimodal as multimodal_models


@dataclass
class TrainConfig:
    """Training config for Machine Learning"""

    model: multimodal_models.SpeechLMConfig
    lr: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    model_load_path: t.Optional[Path] = None
    freezing_config: multimodal_models.FreezingConfig = field(
        default_factory=multimodal_models.FreezingConfig
    )
    num_epochs: int = 0
    max_audio_duration_in_seconds: float = 20.0
    lr_warmup_ratio: float = 0.0
    lr_warmup_steps: int = 0
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: t.Dict[str, float] = None
    weight_decay: float = 0.0
    optimizer_type: str = "adamw_torch"
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: int = 1
    dataset_name: data.DatasetType = data.DatasetType.GIGASPEECH
    dataset_streaming: bool = True
    audio_tokenizer_config: data.AudioTextTokenizerConfig = field(
        default_factory=data.AudioTextTokenizerConfig
    )
    num_workers: int = 0  # The number of workers for data loader
    max_steps: t.Optional[int] = None
    use_cpu: t.Optional[bool] = None
    output_dir: Path = None
    resume_from_checkpoint: bool = False
    allow_tf32: bool = True
    fp16: bool = False
    # WARNING: currently the code cannot handle FP16 in some cases
    # Left as TODO since on A100s we'd use BF16 anyway and it's not a priority
    bf16: bool = False
    seed: int = 42
    deepspeed: t.Optional[Path] = None
    exp_name: str = "test"  # The experiment name
    eval_steps: t.Union[int, float] = 100
    logging_steps: int = 5
    save_steps: t.Union[int, float] = 0.1
    report_logs_to: t.List[str] = ("tensorboard", "wandb", "clearml")

    def __post_init__(self):
        # A builtin method of dataclasses, used for post-processing our configuration.
        if self.use_cpu is None:
            self.use_cpu = not torch.cuda.is_available()

        # Note: it's possible to do BF16 on CPU too, but I don't think we care
        self.bf16 = self.bf16 and not self.use_cpu and torch.cuda.is_bf16_supported()

        # FP16 is not supported right now
        # if self.fp16:
        #     logging.warning(
        #         "Currently the code cannot handle FP16 in some cases, hence disabling it."
        #     )
        #     self.fp16 = False
        if self.bf16 and self.fp16:
            logging.warning(
                "Using BF16 and FP16 at the same time is not supported. Disabling FP16."
            )
            self.fp16 = False

        if self.output_dir is None:
            self.output_dir = Path("runs") / self.exp_name

        if self.per_device_eval_batch_size is None:
            self.per_device_eval_batch_size = self.per_device_train_batch_size

        if self.optimizer_type == "adamw_bnb_8bit" and not torch.cuda.is_available():
            logging.warning(
                "Using CPU with adamw_bnb_8bit is not supported. Switching to adamw_torch"
            )
            self.optimizer_type = "adamw_torch"
