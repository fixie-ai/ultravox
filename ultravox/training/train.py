import contextlib
import dataclasses
import datetime
import logging
import os
import random
import traceback
from typing import Dict, List

import accelerate
import datasets as hf_datasets
import torch
import torch.distributed
import torch.distributed.fsdp as fsdp
import transformers
import wandb
import wandb.sdk

from ultravox import data as datasets
from ultravox.evaluation import eval
from ultravox.inference import infer
from ultravox.model import file_utils
from ultravox.training import config_base
from ultravox.training import ddp_utils
from ultravox.training import model_types
from ultravox.training import trainer as ultra_trainer
from ultravox.training.helpers import prefetch_weights
from ultravox.utils import device_helpers
from ultravox.utils import monkey_patches


def patch_trainer_save_fsdp_model():
    """
    When using FSDP, the trainer._save_checkpoint first calls self.save_model and then accelerator.save_fsdp_model.
    This leads to the model being saved twice when in FULL_STATE_DICT mode as save_model refuses to save the model otherwise.
    To make matters worse, the second save is going to be produce a huge `pytorch_model_fsdp.bin` file which is not what we want.
    This function skips the second save if the state_dict_type is FULL_STATE_DICT.
    We currently only use FULL_STATE_DICT (default) for training checkpoints.
    """

    def save_fsdp_model_if_not_full_state_dict(
        fsdp_plugin, accelerator, model, output_dir, **kwargs
    ):
        if "FULL_STATE_DICT" in str(fsdp_plugin.state_dict_type):
            return
        original_save_fsdp_model(fsdp_plugin, accelerator, model, output_dir, **kwargs)

    original_save_fsdp_model = transformers.trainer.save_fsdp_model
    transformers.trainer.save_fsdp_model = save_fsdp_model_if_not_full_state_dict


def prepare_dataset(
    train_args: config_base.TrainConfig,
    model_pack: model_types.ModelPack,
    data_opts: List[datasets.DatasetOptions],
    data_args: datasets.VoiceDatasetArgs,
    verbose: bool = False,
) -> datasets.SizedIterableDataset:
    data_names = [ds.name for ds in data_opts]
    data_weights = [ds.weight for ds in data_opts]
    data_sets = [
        datasets.create_dataset(
            ds,
            data_args,
            verbose=verbose,
        )
        for ds in data_names
    ]
    # If we're using epochs to train, validate the dataset length is appropriate.
    if train_args.max_steps == 0:
        for ds in data_sets:
            assert (
                len(ds) > 1
            ), f"Dataset {ds} has length {len(ds)} which is too short for epoch training"

    interleave = datasets.InterleaveDataset(data_sets, data_weights)
    ds_with_proc = model_pack.wrap_with_data_proc(interleave)
    if data_args.max_samples != -1:
        return datasets.Range(ds_with_proc, data_args.max_samples)
    else:
        return ds_with_proc


def main() -> None:
    monkey_patches.apply_all_patches()

    # Disable parallelism to avoid deadlocks in DataLoader, apparently
    # multiple processes are forked when using multiple datasets.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Log model checkpoints to W&B: we can reduce to model if storage is an issue
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_PROJECT"] = "ultravox"

    config = config_base.get_train_config()

    patch_trainer_save_fsdp_model()
    transformers.set_seed(config.seed)

    train(config)


def train(config: config_base.TrainConfig):
    world_size = device_helpers.get_world_size()
    local_world_size = device_helpers.get_local_world_size()
    local_rank = device_helpers.get_local_rank()
    is_local_master = device_helpers.is_local_master()
    is_global_master = device_helpers.is_global_master()
    is_distributed = device_helpers.is_distributed()
    logging.info(
        f"world_size: {world_size}, local_world_size: {device_helpers.get_local_world_size()}"
        f" local_rank: {local_rank}, group_rank: {device_helpers.get_group_rank()}"
        f" is_local_master: {is_local_master}, is_global_master: {is_global_master}"
        f" is_distributed: {is_distributed}"
    )

    # DDP blows up logging, so this is an attempt to suppress it to only logs from the master process
    logging.basicConfig(level=logging.INFO if is_local_master else logging.ERROR)
    # os.environ["TORCH_LOGS"] = "ERROR" if is_local_master else "WARNING"
    transformers.logging.set_verbosity(
        logging.WARNING if is_local_master else logging.ERROR
    )
    hf_datasets.logging.set_verbosity(
        logging.WARNING if is_local_master else logging.ERROR
    )

    if is_distributed:
        # It seems gloo breaks on multi-node
        torch.distributed.init_process_group(
            backend="cpu:gloo,cuda:nccl" if local_world_size == world_size else "nccl"
        )

    with ddp_utils.run_on_master_first():
        # For larger models, we assume that the weights are already downloaded via prefetch_weights.py
        # Otherwise the barrier call can timeout.
        # This call is only here as a backstop in case prefetch_weights.py was not run, for example in a local/test run.
        weights_to_download = (
            [config.audio_model, config.text_model]
            if config.audio_model
            else [config.text_model]
        )
        prefetch_weights.download_weights(weights_to_download, config.model_load_dir)

    logging.info("Instantiating model and processor...")

    model_load_context = (
        accelerate.init_empty_weights()
        if config.use_fsdp and not is_local_master
        else contextlib.nullcontext()
    )
    # If we're using FSDP, we can just initialize the model on the main process
    # and use sync_module_states to distribute the weights to the other processes.
    # Otherwise we'd be loading the model on every process, which uses too much CPU memory.
    with model_load_context:
        model_pack = model_types.create_model_pack(config)
        model = model_pack.model

        # Print _no_split_modules
        if config.use_fsdp:
            logging.info(
                f"[Rank {local_rank}] _no_split_modules: {model._no_split_modules}"
            )

    logging.info("Model and processor instantiated.")

    # Starting W&B. HF Trainer can also do this, but this way we can include the config.
    # Initializing sooner also means more of the stdout logs are captured by W&B.
    if "wandb" in config.report_logs_to and is_global_master:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "ultravox"),
            config=dataclasses.asdict(config),
            name=config.exp_name,
            dir="runs",
            tags=config.run_tags,
            save_code=True,
        )

    if config.ignore_data_skip and config.resume_from_load_dir:
        new_shuffle_seed = random.randint(1000, 1999)
        logging.info(
            "Since data skipping is ignored when resuming from a checkpoint,"
            f" randomly setting the train dataset seed to {new_shuffle_seed}."
        )
        config.train_dataset_args.shuffle_seed = new_shuffle_seed
        if wandb.run:
            wandb.run.config.update(
                {"train_dataset_args": dataclasses.asdict(config.train_dataset_args)},
                allow_val_change=True,
            )

    if config.ignore_data_skip and config.resume_from_load_dir:
        new_shuffle_seed = random.randint(1000, 1999)
        logging.info(
            "Since data skipping is ignored when resuming from a checkpoint,"
            f" randomly setting the train dataset seed to {new_shuffle_seed}."
        )
        config.train_dataset_args.shuffle_seed = new_shuffle_seed
        if wandb.run:
            wandb.run.config.update(
                {"train_dataset_args": dataclasses.asdict(config.train_dataset_args)},
                allow_val_change=True,
            )

    model.print_trainable_parameters()

    if not config.use_fsdp:
        # Moving to device in FSDP is handled by the Trainer
        model.to(device=torch.device(config.device, index=local_rank))
        logging.info(f"Using device (world_size): {model.device} ({world_size})")

    # Register custom datasets
    datasets.register_datasets(config.get_data_sets())

    # Prepare dataset, subsetting if needed
    train_dataset: datasets.SizedIterableDataset
    val_datasets: Dict[str, datasets.SizedIterableDataset] = {}

    train_dataset = prepare_dataset(
        train_args=config,
        model_pack=model_pack,
        data_opts=config.get_train_sets(),
        data_args=config.train_dataset_args,
        verbose=is_local_master,
    )
    if is_global_master:
        for val_opt in config.get_val_sets():
            val_dataset = prepare_dataset(
                train_args=config,
                model_pack=model_pack,
                data_opts=[val_opt],
                data_args=config.val_dataset_args,
                verbose=is_local_master,
            )
            val_datasets[val_opt.name] = val_dataset
        logging.info(
            f"Loaded {len(config.train_sets)}) data sets, sample limit: {config.train_dataset_args.max_samples} (val sample limit: {config.val_dataset_args.max_samples})"
        )
    else:
        # When using DDP with split_batches=True, the primary process will distribute the batches to the workers
        # The point of this is to avoid unnecessary data processing/downloading in the workers.
        # When using epochs to train, emptydataset must have a length equal to the training set
        train_dataset = datasets.EmptyDataset(len(train_dataset))
        for val_opts in config.get_val_sets():
            val_datasets[val_opts.name] = datasets.EmptyDataset(
                config.val_dataset_args.max_samples
                if config.val_dataset_args.max_samples > 0
                else 1
            )

    logging.info(f"Config Params: {config}")
    trainer = ultra_trainer.UltraTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_datasets,
        data_collator=model_pack.data_collator,
        processing_class=model_pack.processor,
        args=transformers.Seq2SeqTrainingArguments(
            dataloader_num_workers=config.num_workers if is_global_master else 0,
            output_dir=str(config.output_dir),
            run_name=config.exp_name,
            optim=config.optimizer,
            num_train_epochs=config.num_epochs,
            max_steps=config.max_steps,
            eval_strategy="steps" if config.val_steps else "no",
            eval_steps=config.val_steps,
            save_strategy="steps" if config.save_steps else "no",
            save_steps=config.save_steps,
            logging_first_step=True,
            logging_dir=str(config.logs_dir),
            logging_steps=config.logging_steps,
            # TODO (Farzad): reconsider for multi-node
            # In DDP world_size is set to num_gpus and we want process-0 to split the batches
            # If use use dynamic_batch, the batch size here has to be 1
            per_device_train_batch_size=(
                config.batch_size * world_size if not config.use_dynamic_batch else 1
            ),
            per_device_eval_batch_size=(
                config.val_batch_size * world_size
                if not config.use_dynamic_batch
                else 1
            ),
            dataloader_drop_last=config.dataloader_drop_last,
            accelerator_config={
                "split_batches": True,
            },
            gradient_accumulation_steps=config.grad_accum_steps,
            eval_accumulation_steps=config.val_accum_steps,
            # tf32=dtype == torch.float32 and device.type == "cuda",  # TODO: check for Ampere GPU not just CUDA
            ddp_find_unused_parameters=config.find_unused_parameters,
            learning_rate=config.lr,
            lr_scheduler_type=config.lr_scheduler,
            lr_scheduler_kwargs=config.lr_scheduler_kwargs,
            warmup_steps=0 if config.lr_warmup_steps < 1 else config.lr_warmup_steps,
            warmup_ratio=config.lr_warmup_steps if config.lr_warmup_steps < 1 else 0,
            weight_decay=config.weight_decay,
            fp16=config.fp16_training,
            bf16=config.bf16_training,
            use_cpu=config.device == "cpu",
            seed=config.seed,
            report_to=config.report_logs_to,
            # torch_compile=True,
            fsdp="full_shard auto_wrap" if config.use_fsdp else "",
            fsdp_config={
                "backward_prefetch": "backward_pre",
                "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            },
        ),
    )

    caught_exception = None
    if config.do_train:
        # Training loop
        logging.info("Starting training...")
        t_start = datetime.datetime.now()
        logging.info(f"train start time: {t_start}")

        if config.val_steps:
            if config.use_fsdp:
                logging.warning(
                    "FSDP is enabled: Skipping initial validation since model is not initialized."
                )
            else:
                trainer.evaluate()

        try:
            resume_from_checkpoint = (
                file_utils.download_dir_if_needed(config.model_load_dir)
                if config.model_load_dir and config.resume_from_load_dir
                else None
            )
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        except Exception as e:
            logging.error(f"[rank: {local_rank}] Training failed with error: {e}")
            logging.error(f"[rank: {local_rank}] {traceback.format_exc()}")
            caught_exception = e

        t_end = datetime.datetime.now()
        logging.info(f"train end time: {t_end}")
        logging.info(f"elapsed: {t_end - t_start}")

    # save_final_model(trainer, model_pack, config)

    # use fixie-ai/evals for evaluation if in use_fsdp mode
    if config.do_eval:
        if config.model_type == "lsm":
            logging.warning("Evaluation is not supported for LSM models, skipping")
        elif config.use_fsdp:
            logging.warning("Evaluation is not supported in FSDP mode, skipping")
        else:
            logging.info("Starting evaluation...")
            t_start = datetime.datetime.now()
            logging.info(f"eval start time: {t_start}")

            # Merge LoRA weights for better inference performance.
            # Note: this is irreversible and changes model saving format
            model.merge_and_unload()
            # changing padding side to left for inference
            model_pack.change_text_padding_side("left")
            inference = infer.LocalInference(
                model=model,
                processor=model_pack.processor,
                tokenizer=model_pack.get_text_tokenizer(),
                dtype=device_helpers.get_dtype(config.data_type),
            )

            fsdp.register_fsdp_forward_method(model, "generate")

            metrics, output_files = eval.eval_datasets(
                inference=inference,
                dataset_options=config.get_eval_sets(),
                dataset_args=config.eval_dataset_args,
                augmentation_configs=config.get_eval_augmentations(),
                batch_size=config.eval_batch_size,
                max_tokens=config.eval_max_tokens,
                temperature=config.eval_temperature,
                output_dir=config.output_dir,
            )
            if is_local_master:
                eval.print_results(metrics, output_files)

            t_end = datetime.datetime.now()
            logging.info(f"eval end time: {t_end}")
            logging.info(f"elapsed: {t_end - t_start}")

    # finish wandb run if it exists
    if wandb.run and is_local_master:
        wandb.run.finish(exit_code=1 if caught_exception else 0)
    # destroy process group if distributed training
    if is_distributed:
        torch.cuda.synchronize()
        torch.distributed.destroy_process_group()

    if caught_exception:
        logging.error(
            f"[rank: {local_rank}] Training failed earlier, exiting and raising error."
        )
        raise caught_exception


def save_final_model(
    trainer: transformers.Trainer,
    model_pack: model_types.ModelPack,
    config: config_base.TrainConfig,
):
    if config.use_fsdp:
        # For training checkpoints, even if we decide to use SHARDED_STATE_DICT (which is faster),
        # we still want the final save to be with FULL_STATE_DICT so it can be serialized properly.
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    # saves the model weights correctly (FSDP or otherwise)
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
