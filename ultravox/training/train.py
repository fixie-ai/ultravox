import dataclasses
import glob
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

import datasets as hf_datasets
import safetensors.torch
import simple_parsing
import torch
import torch.distributed
import transformers
import wandb
import wandb.sdk
from torch.utils import data

from ultravox.data import datasets
from ultravox.evaluation import eval
from ultravox.inference import infer
from ultravox.model import data_processing
from ultravox.model import ultravox_config
from ultravox.model import ultravox_model
from ultravox.model import ultravox_pipeline
from ultravox.model import ultravox_processing
from ultravox.model import wandb_utils
from ultravox.training import config_base
from ultravox.training import ddp_utils
from ultravox.utils import string_helpers

INPUT_EXAMPLE = {"text": "Transcribe\n<|audio|>", "audio": b"\x00\x00" * 16000}
OUTPUT_EXAMPLE = {"text": "Hello, world!"}


def prepare_dataset(
    train_args: config_base.TrainConfig,
    dataset_configs: List[datasets.DatasetConfig],
    data_args: datasets.VoiceDatasetArgs,
    processor: ultravox_processing.UltravoxProcessor,
    train_on_inputs: bool,
    stop_strategy: datasets.StopStrategy,
    include_alt_fields: bool = False,  # whether to generate tensors for text-only input (e.g., used for KD training)
) -> datasets.SizedIterableDataset:
    data_sets = [
        datasets.create_dataset(data_args, dataset_config)
        for dataset_config in dataset_configs
    ]
    # If we're using epochs to train, validate the dataset length is appropriate.
    if train_args.max_steps == 0:
        for ds in data_sets:
            assert (
                len(ds) > 1
            ), f"Dataset {ds} has length {len(ds)} which is too short for epoch training"

    interleaved_dataset = datasets.InterleaveDataset(
        data_sets, stop_strategy=stop_strategy
    )
    dataset_with_proc = data_processing.UltravoxDataproc(
        interleaved_dataset,
        processor=processor,
        train_on_inputs=train_on_inputs,
    )
    return dataset_with_proc


def main() -> None:
    # Disable parallelism to avoid deadlocks in DataLoader, apparently
    # multiple processes are forked when using multiple datasets.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Log model checkpoints to W&B: we can reduce to model if storage is an issue
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_PROJECT"] = "ultravox"

    args = simple_parsing.parse(
        config_class=config_base.TrainConfig,
        config_path="ultravox/training/configs/meta_config.yaml",  # base config file
        add_config_path_arg=True,
        args=[string_helpers.fix_hyphens(arg) for arg in sys.argv[1:]],
    )

    transformers.set_seed(args.seed)

    train(args)


def train(args: config_base.TrainConfig):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_master = local_rank == 0

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")

    # DDP blows up logging, so this is an attempt to suppress it to only logs from the master process
    logging.basicConfig(level=logging.INFO if is_master else logging.ERROR)
    # os.environ["TORCH_LOGS"] = "ERROR" if is_master else "WARNING"
    transformers.logging.set_verbosity(logging.WARNING if is_master else logging.ERROR)
    hf_datasets.logging.set_verbosity(logging.WARNING if is_master else logging.ERROR)

    logging.info("Instantiating processor...")
    text_tokenizer: transformers.PreTrainedTokenizerFast = (
        transformers.AutoTokenizer.from_pretrained(args.text_model)
    )
    text_tokenizer.padding_side = "right"
    text_tokenizer.pad_token = text_tokenizer.eos_token

    # Instantiate the model and processor
    config = ultravox_config.UltravoxConfig(
        audio_model_id=args.audio_model,
        text_model_id=args.text_model,
        text_model_lora_config=args.text_model_lora_config,
        audio_model_lora_config=args.audio_model_lora_config,
        adapter_type=args.adapter_type,
        adapter_config=args.adapter_config,
    )

    logging.info("Instantiating model...")

    # Since the model downloads the language model and audio encoder weights, we want one process to finish up
    # downloading before the others start in order to avoid race conditions.
    with ddp_utils.run_on_master_first(is_master):
        model = ultravox_model.UltravoxModel(config)

    # loss_config needs to be passed separately just for model training
    if args.loss_config is not None:
        model.set_loss_config(args.loss_config)

    audio_processor = transformers.AutoProcessor.from_pretrained(args.audio_model)
    processor = ultravox_processing.UltravoxProcessor(
        audio_processor=audio_processor, tokenizer=text_tokenizer
    )

    assert model.get_input_embeddings().num_embeddings == len(
        text_tokenizer
    ), f"Model and tokenizer mismatch: {model.get_input_embeddings().num_embeddings} != {len(text_tokenizer)}"

    model.language_model.config.use_cache = False
    if args.disable_layerdrop and hasattr(model.audio_tower.config, "layerdrop"):
        # layerdrop causes issues when training with DDP
        # https://github.com/huggingface/transformers/issues/17116#issuecomment-1121340890
        model.audio_tower.config.layerdrop = 0.0

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

    resume_from_checkpoint = None
    if args.model_load_dir:
        logging.info(f"Loading model state dict from {args.model_load_dir}")
        load_path = args.model_load_dir
        if wandb_utils.is_wandb_url(load_path):
            # Download the model from W&B. The main process should do the download while the others wait.
            with ddp_utils.run_on_master_first(is_master):
                load_path = wandb_utils.download_model_from_wandb(load_path)
        if args.resume_from_checkpoint:
            resume_from_checkpoint = load_path
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

    # Move the model to GPU and enable bfloat16
    dtype = getattr(torch, args.data_type)
    device = torch.device(args.device, index=local_rank)
    logging.info(
        f"Using dtype and device (world_size): {dtype}, {device} ({world_size})"
    )
    model.to(device=device, dtype=dtype)

    # Prepare dataset, subsetting if needed
    train_dataset: data.IterableDataset
    val_datasets: Dict[str, data.IterableDataset]

    train_dataset = prepare_dataset(
        train_args=args,
        dataset_configs=args.train_dataset_configs,
        train_on_inputs=args.train_on_inputs,
        stop_strategy=args.stop_strategy,
        processor=processor,
        data_args=args.train_dataset_args,
    )
    if is_master:
        val_datasets = {
            dataset_config.alias: prepare_dataset(
                train_args=args,
                dataset_configs=[dataset_config],
                train_on_inputs=args.train_on_inputs,
                stop_strategy=args.stop_strategy,
                processor=processor,
                data_args=args.val_dataset_args,
            )
            for dataset_config in args.val_dataset_configs
        }
        logging.info(
            f"Loaded {len(args.train_dataset_configs)} train data sets, {len(args.val_dataset_configs)} val data sets"
        )
    else:
        # When using DDP with split_batches=True, the primary process will distribute the batches to the workers
        # The point of this is to avoid unnecessary data processing/downloading in the workers.
        # When using epochs to train, emptydataset must have a length equal to the training set
        train_dataset = datasets.EmptyDataset()
        val_datasets = {
            dataset_config.alias: datasets.EmptyDataset()
            for dataset_config in args.val_dataset_configs
        }

    # Set up the data loader
    data_collator = datasets.DataCollatorForSeq2SeqWithAudio(
        tokenizer=text_tokenizer,
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
            evaluation_strategy="steps",
            eval_steps=args.val_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            logging_first_step=True,
            logging_dir=args.logs_dir,
            logging_steps=args.logging_steps,
            # TODO (Farzad): reconsider for multi-node
            # In DDP world_size is set to num_gpus and we want process-0 to split the batches
            per_device_train_batch_size=args.train_dataset_args.batch_size * world_size,
            per_device_eval_batch_size=args.val_dataset_args.batch_size * world_size,
            accelerator_config={"split_batches": True},
            gradient_accumulation_steps=args.grad_accum_steps,
            eval_accumulation_steps=args.val_accum_steps,
            # tf32=dtype == torch.float32 and device.type == "cuda",  # TODO: check for Ampere GPU not just CUDA
            ddp_find_unused_parameters=False,
            learning_rate=args.lr,
            lr_scheduler_type=args.lr_scheduler,
            warmup_steps=args.lr_warmup_steps,
            weight_decay=args.weight_decay,
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            use_cpu=args.device == "cpu",
            seed=args.seed + local_rank,
            report_to=args.report_logs_to,
            resume_from_checkpoint=resume_from_checkpoint,
            # torch_compile=True,
            # fsdp="full_shard auto_wrap",
            # fsdp_transformer_layer_cls_to_wrap='LlamaDecoderLayer',
        ),
    )

    if args.do_train:
        # Training loop
        logging.info("Starting training...")
        t_start = datetime.now()
        logging.info(f"train start time: {t_start}")
        if args.val_steps:
            trainer.evaluate()
        trainer.train()
        t_end = datetime.now()
        logging.info(f"train end time: {t_end}")
        logging.info(f"elapsed: {t_end - t_start}")

    if args.do_eval:
        logging.info("Starting evaluation...")
        t_start = datetime.now()
        logging.info(f"eval start time: {t_start}")

        # Merge LoRA weights for better inference performance.
        # Note: this is irreversible and changes model saving format
        model.merge_and_unload()
        # changing padding side to left for inference
        text_tokenizer.padding_side = "left"
        inference = infer.LocalInference(
            model=model,
            processor=processor,
            tokenizer=text_tokenizer,
            device=args.device,
            dtype=dtype,
        )

        metrics, output_files = eval.run_infer(
            inference,
            args.eval_dataset_args,
            args.eval_dataset_configs,
            world_size,
            local_rank,
        )
        if is_master:
            trainer.log(metrics)
            for output_file in output_files:
                wandb.save(output_file)

        t_end = datetime.now()
        logging.info(f"eval end time: {t_end}")
        logging.info(f"elapsed: {t_end - t_start}")

    if is_master:
        # Saving the model using pipeline to ensure its code is saved
        pipeline = ultravox_pipeline.UltravoxPipeline(
            model, tokenizer=text_tokenizer, device=device
        )
        pipeline.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
