import dataclasses
import logging
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

from ultravox import data
from ultravox.training import config_base
from ultravox.training import ddp_utils
from ultravox.training import model_types
from ultravox.utils import device_helpers
from ultravox.utils import monkey_patches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_batch(
    model: model_types.ModelPack,
    batch: List[Tuple[int, data.VoiceSample]],
) -> Tuple[float, int]:
    """Process a single batch and return the loss and number of valid labels."""
    # Extract just the processed samples from the batch (ignore indices)
    processed_samples = [sample for _, sample in batch]

    # Process batch using the same data collator as training
    collated_batch = model.data_collator(processed_samples)
    collated_batch = {k: v.to(model.model.device) for k, v in collated_batch.items()}

    # Forward pass
    outputs = model.model(**collated_batch)
    loss = outputs.loss

    # Count number of valid labels (non-padding tokens)
    labels = collated_batch["labels"]
    num_labels = (labels != -100).sum().item()

    return loss.item() * num_labels, num_labels


def validate_dataset(
    model: model_types.ModelPack,
    dataset: data.SizedIterableDataset,
    batch_size: int = 1,
) -> float:
    """
    Compute validation loss on a dataset using the same procedure as training.
    The loss is averaged over the number of valid labels (non-padding tokens).
    """
    local_rank = device_helpers.get_local_rank()
    is_distributed = device_helpers.is_distributed()

    # Each rank will get a different shard of the dataset
    data_world_size = device_helpers.get_world_size()
    data_rank = local_rank

    # Wrap the dataset with the data processor
    processed_dataset = model.wrap_with_data_proc(dataset)
    dataset_shard_iterator = ddp_utils.sharded_batch_iterator(
        processed_dataset, batch_size, data_world_size, data_rank
    )

    total_batches = len(dataset) // (batch_size * data_world_size)
    progress_bar = None
    if local_rank == 0:
        progress_bar = tqdm(
            total=total_batches, desc=f"Validating {dataset.name}", unit="batch"
        )

    total_loss = 0.0
    total_labels = 0

    try:
        model.model.eval()
        with torch.no_grad():
            for batch in dataset_shard_iterator:
                batch_loss, batch_labels = process_batch(model, batch)
                total_loss += batch_loss
                total_labels += batch_labels

                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "loss": (
                                f"{total_loss/total_labels:.4f}"
                                if total_labels > 0
                                else "0.0000"
                            )
                        }
                    )

    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        raise

    finally:
        if local_rank == 0 and progress_bar:
            progress_bar.close()

    if is_distributed:
        # Sum the losses and label counts across all processes
        total_loss_tensor = torch.tensor(total_loss, device=model.model.device)
        total_labels_tensor = torch.tensor(total_labels, device=model.model.device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_labels_tensor, op=dist.ReduceOp.SUM)
        total_loss = float(total_loss_tensor.item())
        total_labels = int(total_labels_tensor.item())

    return total_loss / total_labels if total_labels > 0 else 0.0


def validate_datasets(
    model: model_types.ModelPack,
    dataset_options: List[data.DatasetOptions],
    dataset_args: data.VoiceDatasetArgs,
    batch_size: int,
) -> List[Tuple[str, float]]:
    """Validate multiple datasets and return a list of (dataset_name, loss) tuples."""
    metrics: List[Tuple[str, float]] = []

    for dataset_opt in dataset_options:
        try:
            dataset: Union[data.GenericDataset, data.Range] = data.create_dataset(
                dataset_opt.name, dataset_args, verbose=device_helpers.is_local_master()
            )
            if dataset_args.max_samples != -1:
                dataset = data.Range(dataset, dataset_args.max_samples)

            loss = validate_dataset(
                model,
                dataset,
                batch_size=batch_size,
            )

            # compute metrics and save results only on the first process
            if device_helpers.is_global_master():
                logger.info(f"Validation: {dataset.name}, loss: {loss:.4f}")

                metrics.append(
                    (
                        f"{dataset.name}.loss",
                        loss,
                    )
                )

                if wandb.run:
                    wandb.run.log(
                        {
                            "val_results": wandb.Table(
                                columns=["metric", "score"],
                                data=metrics,
                            )
                        }
                    )

        except Exception as e:
            logger.error(f"Error validating dataset {dataset_opt.name}: {str(e)}")
            if device_helpers.is_global_master():
                metrics.append((f"{dataset_opt.name}.loss", float("inf")))

    return metrics


def print_results(metrics: List[Tuple[str, float]]) -> None:
    """Print validation results in a formatted way."""
    print("\nValidation Loss:")
    print("-" * 50)
    for metric_name, score in metrics:
        print(f"{metric_name:40s}: {score:.4f}")
    print("-" * 50)


def main(override_sys_args: Optional[List[str]] = None) -> None:
    """Main entry point for validation."""
    monkey_patches.apply_all_patches()

    config = config_base.get_train_config(override_sys_args)

    # Initialize distributed process group if needed
    if device_helpers.is_distributed():
        if not dist.is_initialized():
            dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        torch.cuda.set_device(device_helpers.get_local_rank())

    if device_helpers.is_global_master():
        if "wandb" in config.report_logs_to:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "ultravox"),
                config=dataclasses.asdict(config),
                name=config.exp_name,
                dir="runs",
                save_code=True,
            )

    try:
        model_pack = model_types.create_model_pack(config)

        metrics = validate_datasets(
            model=model_pack,
            dataset_options=config.get_eval_sets(),
            dataset_args=config.eval_dataset_args,
            batch_size=config.eval_batch_size,
        )

        if device_helpers.is_global_master():
            print_results(metrics)

    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        raise

    finally:
        if device_helpers.is_global_master() and wandb.run:
            wandb.run.finish()

        if device_helpers.is_distributed():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
