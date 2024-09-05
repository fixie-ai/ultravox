#!/usr/bin/env python
import dataclasses
import datetime
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import numpy as np
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


@dataclasses.dataclass
class EvalArgs:
    # Model ID to use for the model
    model: str = simple_parsing.field()
    # datasets
    dataset_configs: List[Dict[str, Any]] = simple_parsing.field()
    # Experiment name
    exp_name: Optional[str] = simple_parsing.field(default=None)
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
        self.dataset_configs = [
            datasets.DatasetConfig(**config) for config in self.dataset_configs
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

    # report logs to wandb
    report_logs_to: List[str] = simple_parsing.list_field()
    
    def __post_init__(self):
        self.dataset_configs = [dataset_config.DatasetConfig(**config) for config in self.dataset_configs]

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

    metrics, output_files = eval.run_infer(
        inference, args, args.dataset_configs, world_size, local_rank
    )

    if args.use_wandb and local_rank == 0:
        wandb.log(metrics)
        for output_file in output_files:
            wandb.save(output_file)
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()

        if self.exp_name is None:
            self.exp_name = datetime.datetime.now().strftime("exp--%Y-%m-%d--%H-%M-%S")

        if self.output_dir is None:
            self.output_dir = Path("runs") / self.exp_name

        if self.logs_dir is None:
            self.logs_dir = self.output_dir / "logs"

def dataset_infer(
    inference: ultravox_infer.UltravoxInference,
    data_loader: data_utils.DataLoader,
    batch_size: Optional[int] = 1,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    world_size: int = 1,
    local_rank: int = 0
) -> List[eval_types.Sample]:
    batch_index = 0
    results = []
    for batch_input in data_loader:
        batch_references = []
        for sample in batch_input:
            assistant_message = sample.messages.pop()
            if assistant_message['role'] != 'assistant':
                raise ValueError(f"Expected assistant message but got: role={assistant_message['role']}, content={assistant_message['content']}")
            batch_references.append(assistant_message["content"])
        
        batch_output = inference.infer_batch(
            batch_input,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # set the starting index for the samples
        sample_index = batch_size * (batch_index * world_size + local_rank)        
        for sample, generated, expected in zip(
            batch_input, batch_output, batch_references
        ):
            
            results.append(
                eval_types.Sample(
                    index=sample_index,
                    question=sample.messages[-1]['content'],
                    expected_answer=expected,
                    generated_answer=generated.text,
                )
            )
            sample_index += 1

        batch_index += 1

    return results

def run_evaluation(inference: ultravox_infer.UltravoxInference, args: GenericInferArgs, dataset_configs: List[dataset_config.DatasetConfig], world_size: int, local_rank: int):
    use_wandb = "wandb" in args.report_logs_to
    world_size = device_helpers.get_world_size()
    
    dataset_args = datasets.VoiceDatasetArgs()
    for config in dataset_configs:
        dataset = datasets.GenericVoiceDataset(dataset_args, config)
        data_loader = data_utils.DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda x: x)
        data_loader = ddp_utils.sharded_dataloader(data_loader, world_size, local_rank)
        results = dataset_infer(inference, data_loader, batch_size=args.batch_size, max_tokens=args.max_tokens, temperature=args.temperature, world_size=world_size, local_rank=local_rank)
        
        if world_size > 1:
            dist.barrier()
            gathered_results = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_results, results)
            if dist.get_rank() == 0:
                # Interleave results from all workers
                results = []
                max_items = max(len(sublist) for sublist in gathered_results)
                for i in range(max_items):
                    for worker_results in gathered_results:
                        if i < len(worker_results):
                            results.extend(worker_results[i])

        if world_size == 1 or (world_size > 1 and dist.get_rank() == 0):
            dataset_alias = config.alias
            if config.eval_config:
                eval_result: eval_types.Result = eval.evaluate_answers(results, config.eval_config)
                print(f"Dataset: {dataset_alias}, Metric: {config.eval_config.metric}, Score: {eval_result.score:.2f}")
                
                # Log to wandb
                if use_wandb:
                    wandb.log({
                        f"eval/{dataset_alias}/{config.eval_config.metric}": eval_result.score,
                    })

            filename = f"{dataset_alias}.results.json"
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
        dist.init_process_group(backend="nccl")

    # Initialize wandb if it's in report_logs_to
    use_wandb = "wandb" in args.report_logs_to
    if use_wandb and local_rank == 0:
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

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_evaluation(inference, args, args.dataset_configs, world_size, local_rank)

    # Finish wandb run
    if use_wandb and (world_size == 1 or (world_size > 1 and local_rank == 0)):
        wandb.finish()

if __name__ == "__main__":
    main()
