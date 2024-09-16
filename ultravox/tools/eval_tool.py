#!/usr/bin/env python
import dataclasses
import logging
import os
import sys

import simple_parsing
import torch
import torch.distributed as dist
import wandb

from ultravox.evaluation import eval
from ultravox.inference import ultravox_infer
from ultravox.training import config_base
from ultravox.training import ddp_utils
from ultravox.utils import device_helpers
from ultravox.utils import string_helpers

logging.basicConfig(level=logging.INFO)


def main():
    args = simple_parsing.parse(
        config_class=config_base.TrainConfig,
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
            args.model_load_dir,
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
