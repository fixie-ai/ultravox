import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import simple_parsing
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

from ultravox import data
from ultravox.evaluation import eval_metrics
from ultravox.evaluation import eval_types
from ultravox.inference import infer
from ultravox.inference import ultravox_infer
from ultravox.training import ddp_utils
from ultravox.utils import device_helpers
from ultravox.utils import monkey_patches

logging.basicConfig(level=logging.INFO)


@dataclasses.dataclass
class EvalConfig:
    model: str
    eval_sets: List[Dict[str, Any]] = simple_parsing.list_field()
    eval_dataset_args: data.EvalDatasetArgs = simple_parsing.field(
        default_factory=data.EvalDatasetArgs
    )

    def get_eval_sets(self) -> List[data.DatasetOptions]:
        return [data.DatasetOptions(**ds) for ds in self.eval_sets]

    eval_batch_size: int = 4
    eval_max_tokens: int = 512
    eval_temperature: float = 0.0

    device: str = "cuda"
    data_type: str = "bfloat16"

    exp_name: Optional[str] = None
    output_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    report_logs_to: List[str] = simple_parsing.list_field()

    def __post_init__(self):
        assert self.data_type in ["bfloat16", "float16", "float32"]

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device != "cuda":
            if self.data_type == "bfloat16":
                self.data_type = "float32"

        if self.exp_name is None:
            self.exp_name = datetime.datetime.now().strftime("exp--%Y-%m-%d--%H-%M-%S")
        if self.output_dir is None:
            self.output_dir = Path("runs") / self.exp_name


def infer_dataset_shard(
    inference: infer.LocalInference,
    dataset_shard_iterator: Generator[List[Tuple[int, data.VoiceSample]], None, None],
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    progress_bar: Optional[tqdm] = None,
) -> List[eval_types.Sample]:
    results: List[eval_types.Sample] = []
    for batch_input in dataset_shard_iterator:
        batch_indices = [idx for idx, _ in batch_input]
        batch_samples = [sample for _, sample in batch_input]
        batch_references = []
        for sample in batch_samples:
            assistant_message = sample.messages.pop()
            if assistant_message["role"] != "assistant":
                raise ValueError(
                    f"Expected assistant message but got: role={assistant_message['role']}, content={assistant_message['content']}"
                )
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
                    question=sample.messages[-1]["content"],
                    transcript=sample.audio_transcript or "",
                    expected_answer=reference,
                    generated_answer=output.text,
                )
            )

        if progress_bar:
            progress_bar.update(1)
    return results


def infer_dataset(
    inference: infer.LocalInference,
    dataset: data.SizedIterableDataset,
    batch_size: int = 1,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
) -> List[eval_types.Sample]:

    world_size = device_helpers.get_world_size()
    local_rank = device_helpers.get_local_rank()

    progress_bar = None
    if local_rank == 0:
        total_batches = len(dataset) // (batch_size * world_size)
        progress_bar = tqdm(total=total_batches)

    dataset_shard_iterator = ddp_utils.sharded_batch_iterator(
        dataset, batch_size, world_size, local_rank
    )
    results: List[eval_types.Sample] = infer_dataset_shard(
        inference, dataset_shard_iterator, max_tokens, temperature, progress_bar
    )
    results = ddp_utils.all_gather_list(results)

    if local_rank == 0:
        if progress_bar:
            progress_bar.close()
        results.sort(key=lambda x: x.index)
        return results
    else:
        return []


def eval_datasets(
    inference: infer.LocalInference,
    dataset_options: List[data.DatasetOptions],
    dataset_args: data.EvalDatasetArgs,
    batch_size: int,
    max_tokens: Optional[int],
    temperature: float,
    output_dir: Optional[Path],
):
    metrics = []
    output_files = []
    local_rank = device_helpers.get_local_rank()
    for dataset_opt in dataset_options:
        dataset: Union[data.GenericDataset, data.Range] = data.create_dataset(
            dataset_opt.name, dataset_args, verbose=local_rank == 0
        )
        if dataset_args.max_samples:
            dataset = data.Range(dataset, dataset_args.max_samples)
        results = infer_dataset(
            inference,
            dataset,
            batch_size=batch_size,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # compute metrics, if specified, and save results only on the first process
        if local_rank == 0:
            # ensure same order
            dataset_config = dataset.get_config()
            if dataset_config.eval_config:
                eval_result: eval_types.Result = eval_metrics.evaluate_answers(
                    results, dataset_config.eval_config
                )
                logging.info(
                    f"Eval: {dataset.name}, {dataset_config.eval_config.metric}: {eval_result.score:.2f}"
                )
                metrics.append(
                    (
                        f"{dataset.name}.{dataset_config.eval_config.metric}",
                        eval_result.score,
                    )
                )
                if wandb.run:
                    wandb.run.log(
                        {
                            "eval_results": wandb.Table(
                                columns=["metric", "score"], data=metrics
                            )
                        }
                    )

            if output_dir:
                output_file = os.path.join(output_dir, f"{dataset.name}.json")
                with open(output_file, "w") as f:
                    results_json = [result.to_dict() for result in results]
                    json.dump(results_json, f, ensure_ascii=False, indent=2)
                print(f"Results saved to {output_file}")
                output_files.append(output_file)
                if wandb.run:
                    wandb.run.save(output_file)

    return metrics, output_files


def print_results(metrics: List[Tuple[str, float]], output_files: List[str]):
    print(f"Evaluation Scores:\n")
    for metric_name, score in metrics:
        print(f"  {metric_name}: {score}")
    print("Output Files:\n")
    for output_file in output_files:
        print(f"  {output_file}")


def main(override_sys_args: Optional[List[str]] = None):
    monkey_patches.apply_all_patches()

    config = simple_parsing.parse(
        EvalConfig, add_config_path_arg=True, args=override_sys_args
    )

    world_size = device_helpers.get_world_size()
    local_rank = device_helpers.get_local_rank()

    if world_size > 1:
        # use gloo instead of nccl as the gathering opration is on cpu; need to double check if nccl is supported previously.
        dist.init_process_group(backend="gloo")

    if local_rank == 0:
        if config.output_dir:
            config.output_dir.mkdir(parents=True, exist_ok=True)
        if "wandb" in config.report_logs_to:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "ultravox"),
                config=dataclasses.asdict(config),
                name=config.exp_name,
                dir="runs",
                save_code=True,
            )

    with ddp_utils.run_on_master_first(local_rank == 0):
        inference = ultravox_infer.UltravoxInference(
            config.model,
            device=(
                f"{config.device}:{local_rank}" if world_size > 1 else config.device
            ),
            data_type=config.data_type,
        )

    metrics, output_files = eval_datasets(
        inference,
        config.get_eval_sets(),
        config.eval_dataset_args,
        config.eval_batch_size,
        config.eval_max_tokens,
        config.eval_temperature,
        config.output_dir,
    )

    if local_rank == 0:
        print_results(metrics, output_files)

        if wandb.run:
            wandb.run.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
