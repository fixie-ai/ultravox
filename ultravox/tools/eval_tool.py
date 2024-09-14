#!/usr/bin/env python
import dataclasses
import datetime
import os
import sys
from pathlib import Path
from typing import List, Optional
import logging

import simple_parsing
import torch
import torch.distributed as dist
import wandb

from ultravox.data import datasets
from ultravox.evaluation import eval
from ultravox.inference import ultravox_infer
from ultravox.training import ddp_utils
from ultravox.utils import device_helpers
from ultravox.utils import string_helpers

logging.basicConfig(level=logging.INFO)

@dataclasses.dataclass
class EvalArgs:
    model: str = simple_parsing.field()
    """Model ID to use for the model"""

    eval_dataset_configs: List[datasets.DatasetConfig] = dataclasses.field(
        default_factory=list
    )
    """List of evaluation dataset configurations"""

    eval_dataset_args: datasets.VoiceDatasetArgs = dataclasses.field(
        default_factory=datasets.VoiceDatasetArgs
    )
    """Global arguments for the evaluation dataset"""

    device: str = device_helpers.default_device()
    """Device to use for training (e.g., 'cuda', 'cpu', 'mps')"""

    data_type: str = device_helpers.default_dtype_str()
    """Data type to use for training (e.g., 'bfloat16', 'float16', 'float32')"""

    exp_name: Optional[str] = simple_parsing.field(default=None)
    """The experiment name"""

    output_dir: Optional[Path] = simple_parsing.field(default=None)
    """Output directory"""

    report_logs_to: List[str] = simple_parsing.field(default_factory=list)
    """Whether to report results to wandb"""

    def __post_init__(self):
        self.eval_dataset_configs = [
            datasets.DatasetConfig(**config) for config in self.eval_dataset_configs
        ]

        if self.data_type not in ["bfloat16", "float16", "float32", None]:
            raise ValueError(
                f"Invalid data type: {self.data_type}. Please specify one of the following: bfloat16, float16, float32."
            )
        if self.device is None:
            self.device = device_helpers.default_device()

        if self.output_dir is None:
            self.output_dir = Path("runs") / self.exp_name
        self.output_dir = Path(self.output_dir)

        if self.exp_name is None:
            self.exp_name = datetime.datetime.now().strftime("exp--%Y-%m-%d--%H-%M-%S")


def main():
    args = simple_parsing.parse(
        config_class=EvalArgs,
        add_config_path_arg=True,
        args=[string_helpers.fix_hyphens(arg) for arg in sys.argv[1:]],
    )

    world_size = device_helpers.get_world_size()
    local_rank = device_helpers.get_local_rank()
    device = torch.device(args.device, index=local_rank)
    if world_size > 1:
        dist.init_process_group(backend="gloo")

    if local_rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if "wandb" in args.report_logs_to:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "ultravox"),
                config=dataclasses.asdict(args),
                name=args.exp_name,
                dir="runs",
            )

    with ddp_utils.run_on_master_first(local_rank == 0):
        inference = ultravox_infer.UltravoxInference(
            args.model,
            device=device,
            data_type=device_helpers.get_dtype(args.data_type),
        )

    metrics, output_files = eval.run_infer(
        inference,
        args.eval_dataset_args,
        args.eval_dataset_configs,
        world_size,
        local_rank,
    )

    if local_rank == 0:
        if "wandb" in args.report_logs_to:
            wandb.log(metrics)
            for output_file in output_files:
                wandb.save(output_file)
            wandb.finish()
        else:
            logging.info(f"Evaluation Scores:\n")
            for metric, score in metrics.items():
                logging.info(f"  {metric}: {score}")
            logging.info("Output Files:\n")
            for output_file in output_files:
                logging.info(f"  {output_file}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
