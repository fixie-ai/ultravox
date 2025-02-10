import contextlib
import dataclasses
import gc
import glob
import logging
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

import accelerate
import datasets as hf_datasets
import pandas as pd
import safetensors.torch
import torch
import torch.distributed
import transformers
import wandb
import wandb.sdk

from ultravox import data as datasets
from ultravox.model import data_processing
from ultravox.model import ultravox_config
from ultravox.model import ultravox_model
from ultravox.model import ultravox_pipeline
from ultravox.model import ultravox_processing
from ultravox.model import wandb_utils
from ultravox.training import config_base
from ultravox.training import ddp_utils
from ultravox.training.helpers import prefetch_weights
from ultravox.utils import monkey_patches


def prepare_dataset(
    train_args: config_base.TrainConfig,
    data_opts: List[config_base.DatasetOptions],
    data_args: datasets.VoiceDatasetArgs,
    processor: ultravox_processing.UltravoxProcessor,
    train_on_inputs: bool,
    num_samples: Optional[int] = None,
    include_alt_fields: bool = False,  # whether to generate tensors for text-only input (e.g., used for KD training)
) -> datasets.SizedIterableDataset:
    data_names = [ds.name for ds in data_opts]
    data_weights = [ds.weight for ds in data_opts]
    data_sets = [datasets.create_dataset(ds, data_args) for ds in data_names]
    # If we're using epochs to train, validate the dataset length is appropriate.
    if train_args.max_steps == 0:
        for ds in data_sets:
            assert (
                len(ds) > 1
            ), f"Dataset {ds} has length {len(ds)} which is too short for epoch training"

    interleave = datasets.InterleaveDataset(data_sets, data_weights)
    ds_with_proc = data_processing.UltravoxDataproc(
        interleave,
        processor=processor,
        train_on_inputs=train_on_inputs,
        include_alt_fields=include_alt_fields,
    )
    limited_ds = datasets.Range(ds_with_proc, num_samples=num_samples)
    return limited_ds


def main() -> None:
    monkey_patches.apply_all_patches()

    # Disable parallelism to avoid deadlocks in DataLoader, apparently
    # multiple processes are forked when using multiple datasets.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Log model checkpoints to W&B: we can reduce to model if storage is an issue
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_PROJECT"] = "ultravox"

    args = config_base.get_train_args()

    transformers.set_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_master = local_rank == 0

    train(args)

    if args.do_eval and is_master:
        gc.collect()
        torch.cuda.empty_cache()
        evaluate(args)


def train(args: config_base.TrainConfig):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_master = local_rank == 0
    is_distributed = world_size > 1

    # DDP blows up logging, so this is an attempt to suppress it to only logs from the master process
    logging.basicConfig(level=logging.INFO if is_master else logging.ERROR)
    # os.environ["TORCH_LOGS"] = "ERROR" if is_master else "WARNING"
    transformers.logging.set_verbosity(logging.WARNING if is_master else logging.ERROR)
    hf_datasets.logging.set_verbosity(logging.WARNING if is_master else logging.ERROR)

    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")

    with ddp_utils.run_on_master_first(is_master):
        # For larger models, we assume that the weights are already downloaded via prefetch_weights.py
        # Otherwise the barrier call can timeout.
        # This call is only here as a backstop in case prefetch_weights.py was not run, for example in a local/test run.
        prefetch_weights.download_weights(
            [args.text_model, args.audio_model], args.model_load_dir
        )

    logging.info("Instantiating processor...")
    text_tokenizer: transformers.PreTrainedTokenizerFast = (
        transformers.AutoTokenizer.from_pretrained(args.text_model)
    )
    text_tokenizer.padding_side = "right"
    text_tokenizer.pad_token = text_tokenizer.eos_token
    audio_processor = transformers.AutoProcessor.from_pretrained(args.audio_model)
    processor = ultravox_processing.UltravoxProcessor(audio_processor, text_tokenizer)

    # Instantiate the model and processor
    config = ultravox_config.UltravoxConfig(
        audio_model_id=args.audio_model,
        text_model_id=args.text_model,
        text_model_lora_config=args.text_model_lora_config,
        audio_model_lora_config=args.audio_model_lora_config,
        torch_dtype=args.data_type,
        pad_token_id=text_tokenizer.eos_token_id,
        audio_latency_block_size=args.audio_latency_block_size,
    )

    logging.info("Instantiating model...")

    model_load_context = (
        accelerate.init_empty_weights()
        if args.use_fsdp and not is_master
        else contextlib.nullcontext()
    )
    # If we're using FSDP, we can just initialize the model on the main process
    # and use sync_model_states to distribute the weights to the other processes.
    # Otherwise we'd be loading the model on every process, which uses too much CPU memory.
    with model_load_context:
        model = ultravox_model.UltravoxModel(config)

    assert model.get_input_embeddings().num_embeddings == len(
        text_tokenizer
    ), f"Model and tokenizer mismatch: {model.get_input_embeddings().num_embeddings} != {len(text_tokenizer)}"

    model.language_model.config.use_cache = False
    if args.disable_layerdrop and hasattr(model.audio_tower.config, "layerdrop"):
        # layerdrop causes issues when training with DDP
        # https://github.com/huggingface/transformers/issues/17116#issuecomment-1121340890
        model.audio_tower.config.layerdrop = 0.0

    # loss_config needs to be passed separately just for model training
    if args.loss_config is not None:
        model.set_loss_config(args.loss_config)

    logging.info("Model and processor instantiated.")

    # Starting W&B. HF Trainer can also do this, but this way we can include the config.
    # Initializing sooner also means more of the stdout logs are captured by W&B.
    if "wandb" in args.report_logs_to and is_master:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "ultravox"),
            config=dataclasses.asdict(args),
            name=args.exp_name,
            dir="runs",
            tags=args.run_tags,
            save_code=True,
        )

    if args.model_load_dir:
        logging.info(f"Loading model state dict from {args.model_load_dir}")
        load_path = args.model_load_dir
        if wandb_utils.is_wandb_url(load_path):
            # We assume that the weights are already downloaded via prefetch_weights.py
            # and hence this is just resolving the path. If the weights are not downloaded,
            # we might see a race condition here when using DDP.
            load_path = wandb_utils.download_model_from_wandb(load_path)
        if os.path.isdir(load_path):
            load_path = os.path.join(load_path, "model*.safetensors")
        paths = glob.glob(load_path)
        assert len(paths) > 0, f"No model files found at {load_path}"
        for path in glob.glob(load_path):
            state_dict = safetensors.torch.load_file(path)
            mismatch = model.load_state_dict(state_dict, strict=False)
            if mismatch.unexpected_keys:
                raise ValueError(
                    f"Unexpected keys in state dict: {mismatch.unexpected_keys}"
                )

    model.print_trainable_parameters()

    if not args.use_fsdp:
        # Moving to device in FSDP is handled by the Trainer
        model.to(device=torch.device(args.device, index=local_rank))
        logging.info(f"Using device (world_size): {model.device} ({world_size})")

    # Register custom datasets
    datasets.register_datasets(args.get_data_sets())

    # Prepare dataset, subsetting if needed
    train_dataset: datasets.SizedIterableDataset
    val_datasets: Dict[str, datasets.SizedIterableDataset] = {}

    train_dataset = prepare_dataset(
        train_args=args,
        data_opts=args.get_train_sets(),
        train_on_inputs=args.train_on_inputs,
        processor=processor,
        num_samples=args.num_samples,
        data_args=datasets.VoiceDatasetArgs(
            shuffle=args.shuffle_data,
            shuffle_seed=args.shuffle_seed,
            max_audio_duration_secs=args.max_audio_duration_secs,
        ),
        include_alt_fields=model.loss_config.requires_alt_fields,
    )
    if is_master:
        val_ds_args = datasets.VoiceDatasetArgs(
            split=datasets.DatasetSplit.VALIDATION,
            shuffle=False,
            max_audio_duration_secs=16,
        )
        for val_opt in args.get_val_sets():
            val_dataset = prepare_dataset(
                train_args=args,
                data_opts=[val_opt],
                train_on_inputs=args.train_on_inputs,
                processor=processor,
                num_samples=args.val_num_samples,
                data_args=val_ds_args,
                include_alt_fields=model.loss_config.requires_alt_fields,
            )
            val_datasets[val_opt.name] = val_dataset
        logging.info(
            f"Loaded {len(args.train_sets)}) data sets, sample limit: {args.num_samples} (val sample limit: {args.val_num_samples})"
        )
    else:
        # When using DDP with split_batches=True, the primary process will distribute the batches to the workers
        # The point of this is to avoid unnecessary data processing/downloading in the workers.
        # When using epochs to train, emptydataset must have a length equal to the training set
        train_dataset = datasets.EmptyDataset(len(train_dataset))
        for val_opts in args.get_val_sets():
            val_datasets[val_opts.name] = datasets.EmptyDataset()

    # Set up the data loader
    data_collator = datasets.DataCollatorForSeq2SeqWithAudio(
        tokenizer=text_tokenizer,
        include_alt_fields=model.loss_config.requires_alt_fields,
    )

    logging.info(f"Config Params: {args}")
    trainer = transformers.Seq2SeqTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=val_datasets,
        data_collator=data_collator,
        tokenizer=text_tokenizer,
        args=transformers.Seq2SeqTrainingArguments(
            dataloader_num_workers=args.num_workers if is_master else 0,
            output_dir=args.output_dir,
            run_name=args.exp_name,
            optim=args.optimizer,
            num_train_epochs=args.num_epochs,
            max_steps=args.max_steps,
            eval_strategy="steps" if args.val_steps else "no",
            eval_steps=args.val_steps,
            save_strategy="steps" if args.save_steps else "no",
            save_steps=args.save_steps,
            logging_first_step=True,
            logging_dir=args.logs_dir,
            logging_steps=args.logging_steps,
            # TODO (Farzad): reconsider for multi-node
            # In DDP world_size is set to num_gpus and we want process-0 to split the batches
            per_device_train_batch_size=args.batch_size * world_size,
            accelerator_config={"split_batches": True},
            gradient_accumulation_steps=args.grad_accum_steps,
            eval_accumulation_steps=args.val_accum_steps,
            # tf32=dtype == torch.float32 and device.type == "cuda",  # TODO: check for Ampere GPU not just CUDA
            ddp_find_unused_parameters=False,
            learning_rate=args.lr,
            lr_scheduler_type=args.lr_scheduler,
            warmup_steps=args.lr_warmup_steps,
            weight_decay=args.weight_decay,
            # fp16=dtype == torch.float16,
            # bf16=dtype == torch.bfloat16,
            use_cpu=args.device == "cpu",
            seed=args.seed + local_rank,
            report_to=args.report_logs_to,
            # torch_compile=True,
            fsdp="full_shard auto_wrap" if args.use_fsdp else "",
            fsdp_config={
                "backward_prefetch": "backward_pre",
                "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "state_dict_type": "SHARDED_STATE_DICT",
                "sync_module_states": "true",
            },
        ),
    )

    if args.do_train:
        # Training loop
        logging.info("Starting training...")
        t_start = datetime.now()
        logging.info(f"train start time: {t_start}")

        if args.val_steps:
            if args.use_fsdp:
                logging.warning(
                    "FSDP is enabled: Skipping initial validation since model is not initialized."
                )
            else:
                trainer.evaluate()

        trainer.train()
        t_end = datetime.now()
        logging.info(f"train end time: {t_end}")
        logging.info(f"elapsed: {t_end - t_start}")

    if args.use_fsdp:
        # For training checkpoints, we want to use SHARDED_STATE_DICT which should be faster,
        # but for the final save we want FULL_STATE_DICT so it can be serialized properly.
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    # We use both pipeline.save_pretrained and trainer.save_model to save everything.
    # This is because pipeline.save_pretrained knows how to save the pipeline (code and config),
    # but it doesn't know how to save FSDP models correctly (the final tensors could be flattened).
    # on the other hand, trainer.save_model knows how to save FSDP models correctly, but it won't save the pipeline.
    # Saving FSDP models is already quite slow though, so we don't want to save the model twice.
    pipeline = ultravox_pipeline.UltravoxPipeline(
        model, tokenizer=text_tokenizer, device=model.device
    )
    old_save_pretrained = model.save_pretrained
    model.save_pretrained = lambda *_, **__: None  # type: ignore[method-assign]
    # saves the pipeline code and populates the config
    pipeline.save_pretrained(args.output_dir)
    model.save_pretrained = old_save_pretrained  # type: ignore[method-assign]

    # saves the model weights correctly (FSDP or otherwise)
    trainer.save_model(args.output_dir)


def evaluate(args: config_base.TrainConfig):
    """
    Evaluate the model on the audio and text datasets.

    NOTE: This function must be run only on the primary process.
    """
    logging.info("Starting evaluation...")
    t_start = datetime.now()
    logging.info(f"eval start time: {t_start}")

    if args.text_model_lora_config and args.text_model_lora_config.r:
        logging.warn(
            "Model has unmerged LoRA config. This can lead to slower inference."
        )

    logs_dir = wandb.run.dir if wandb.run else str(args.logs_dir)

    # Run audio-based evaluations and log to W&B
    audio_metrics_df = run_oaievalset(
        log_dir=os.path.join(logs_dir, "oaieval/audio"),
        model_dir=str(args.output_dir),
        eval_set="audio-core",
        num_samples=args.eval_num_samples,
    )
    # TODO: it would be best to do trainer.log, but then we'd risk keeping parts of the model
    # in GPU memory, which could cause OOM errors.
    if wandb.run:
        wandb.run.log({"eval_audio": wandb.Table(data=audio_metrics_df)})

    if args.eval_text_only:
        # Run text-only evaluations and log to W&B
        text_metrics_df = run_oaievalset(
            log_dir=os.path.join(logs_dir, "oaieval/text"),
            model_dir=str(args.output_dir),
            eval_set="transcript-core",
            num_samples=args.eval_num_samples,
        )
        if wandb.run:
            wandb.run.log({"eval_text": wandb.Table(data=text_metrics_df)})

    t_end = datetime.now()
    logging.info(f"eval end time: {t_end}")
    logging.info(f"elapsed: {t_end - t_start}")


def run_oaievalset(
    log_dir: str, model_dir: str, eval_set: str, num_samples: Optional[int] = None
) -> pd.DataFrame:
    env = os.environ.copy()

    # num_gpus = max(1, torch.cuda.device_count())
    env["EVALS_THREADS"] = "64"

    # TODO: currently running this on a single GPU is faster than multiple GPUs :facepalm:
    env["CUDA_VISIBLE_DEVICES"] = "0"

    command = [
        "oaievalset",
        "--record_dir",
        log_dir,
        "generation/gpu/ultravox-dev",
        eval_set,
        f"--completion_args=model={model_dir}",
    ]
    if num_samples:
        command.append(f"--max_samples={num_samples}")

    # Run the evaluation set
    subprocess.run(command, check=True, env=env)

    # Extract the results from the log directory
    subprocess.run(
        [
            "python",
            "-m",
            "evals.elsuite.audio.make_table",
            "--out_dir",
            log_dir,
            "--log_dir",
            log_dir,
        ],
        check=True,
    )

    df = pd.read_csv(os.path.join(log_dir, "results.csv"))

    return df


if __name__ == "__main__":
    main()
