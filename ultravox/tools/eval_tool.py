#!/usr/bin/env python
import dataclasses
import json
import os
import time
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
import numpy as np
import simple_parsing
from torch.utils import data as data_utils
import torch
import torch.distributed as dist
import wandb
import datetime

from ultravox.data import datasets, dataset_config
from ultravox.evaluation import eval, eval_types
from ultravox.inference import ultravox_infer
from ultravox.utils import device_helpers, string_helpers
from ultravox.training import ddp_utils
from ultravox.evaluation import inference

@dataclasses.dataclass
class EvalArgs:
    # Model ID to use for the model
    model: str = simple_parsing.field()
    # datasets
    dataset_configs: List[Dict[str, Any]] = simple_parsing.field()
    # Experiment name
    exp_name: str = simple_parsing.field(default=None)
    # Device to use for inference
    device: Optional[str] = simple_parsing.field(default=None)
    # Data type to use for the model
    data_type: Optional[str] = simple_parsing.field(default=None)
    # Temperature for sampling
    temperature: Optional[float] = simple_parsing.field(default=0.0)
    # Maximum tokens to generateexp
    max_tokens: Optional[int] = simple_parsing.field(default=1024)
    # Batch size
    batch_size: Optional[int] = simple_parsing.field(default=1)
    # Output directory
    output_dir: Optional[Path] = simple_parsing.field(default=None)
    # report results to wandb
    use_wandb: bool = simple_parsing.field(default=False)
    
    def __post_init__(self):
        self.dataset_configs = [dataset_config.DatasetConfig(**config) for config in self.dataset_configs]

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
        if args.use_wandb:
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

    inference.run_infer(inference, args, args.dataset_configs, world_size, local_rank)

    if args.use_wandb and local_rank == 0:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()