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

@dataclasses.dataclass
class GenericInferArgs:
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
    # Logs directory
    logs_dir: Optional[Path] = simple_parsing.field(default=None)

    # report logs to wandb
    report_logs_to: List[str] = simple_parsing.list_field()
    
    def __post_init__(self):
        self.dataset_configs = [dataset_config.DatasetConfig(**config) for config in self.dataset_configs]

        if self.data_type not in ["bfloat16", "float16", "float32", None]:
            raise ValueError(
                f"Invalid data type: {self.data_type}. Please specify one of the following: bfloat16, float16, float32."
            )
        if self.device is None:
            self.device = device_helpers.default_device()

        if self.exp_name is None:
            self.exp_name = datetime.datetime.now().strftime("exp--%Y-%m-%d--%H-%M-%S")

        if self.output_dir is None:
            self.output_dir = Path("runs") / self.exp_name

        if self.logs_dir is None:
            self.logs_dir = self.output_dir / "logs"

def dataset_infer(
    inference: ultravox_infer.UltravoxInference,
    dataset: datasets.VoiceDataset,
    batch_size: Optional[int] = 1,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    world_size: int = 1,
    local_rank: int = 0
) -> List[eval_types.Sample]:
    results = []
    for batch_input in ddp_utils.sharded_batch_iterator(dataset, batch_size, world_size, local_rank):
        batch_indices = [idx for idx, _ in batch_input]
        batch_samples = [sample for _, sample in batch_input]
        batch_references = []
        for sample in batch_samples:
            assistant_message = sample.messages.pop()
            if assistant_message['role'] != 'assistant':
                raise ValueError(f"Expected assistant message but got: role={assistant_message['role']}, content={assistant_message['content']}")
            batch_references.append(assistant_message["content"])
        
        batch_output = inference.infer_batch(
            batch_samples,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for index, sample, output, reference in zip(
            batch_indices, batch_samples, batch_output, batch_references
        ):
            
            results.append(
                eval_types.Sample(
                    index=index,
                    question=sample.messages[-1]['content'],
                    reference=reference,
                    hypothesis=output.text,
                )
            )
    return results

def run_evaluation(inference: ultravox_infer.UltravoxInference, args: GenericInferArgs, dataset_configs: List[dataset_config.DatasetConfig], world_size: int, local_rank: int):
    use_wandb = "wandb" in args.report_logs_to

    dataset_args = datasets.VoiceDatasetArgs()
    for config in dataset_configs:
        dataset = datasets.GenericVoiceDataset(dataset_args, config)
        results = dataset_infer(inference, dataset, batch_size=args.batch_size, max_tokens=args.max_tokens, temperature=args.temperature, world_size=world_size, local_rank=local_rank)
        results = ddp_utils.all_gather_list(results)
        if local_rank == 0:
            # Sort results based on index
            results.sort(key=lambda x: x.index)
            dataset_alias = config.alias
            if config.eval_config:
                eval_result: eval_types.Result = eval.evaluate_answers(results, config.eval_config)
                print(f"Dataset: {dataset_alias}, Metric: {config.eval_config.metric}, Score: {eval_result.score:.2f}")
                
                # Log to wandb
                if use_wandb:
                    wandb.log({
                        f"eval/{dataset_alias}-{config.eval_config.metric}": eval_result.score,
                    })

            filename = f"{dataset_alias}-{config.eval_config.metric}.results.json"
            output_file = args.output_dir / filename
            
            with open(output_file, "w") as f:   
                results_json = [result.to_dict() for result in results]
                json.dump(results_json, f, ensure_ascii=False,indent=2)
            print(f"Results saved to {output_file}")
            
            # Log results file to wandb
            if use_wandb:
                wandb.save(str(output_file))

def main():
    args = simple_parsing.parse(
        config_class=GenericInferArgs,
        add_config_path_arg=True,
        args=[string_helpers.fix_hyphens(arg) for arg in sys.argv[1:]],
    )

    world_size = device_helpers.get_world_size()
    local_rank = device_helpers.get_local_rank()
    device = torch.device(args.device, index=local_rank)
    if world_size > 1:
        timeout = datetime.timedelta(seconds=600)  # 10 minutes
        dist.init_process_group(backend="gloo", timeout=timeout)

    output_dir = Path(args.output_dir)
    # Initialize wandb if it's in report_logs_to
    use_wandb = "wandb" in args.report_logs_to
    if local_rank == 0:
            # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        if use_wandb:
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

    run_evaluation(inference, args, args.dataset_configs, world_size, local_rank)

    # Finish wandb run
    if use_wandb and local_rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()