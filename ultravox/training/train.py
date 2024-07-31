import copy
import dataclasses
import glob
import logging
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

import datasets as hf_datasets
import safetensors.torch
import simple_parsing
import torch
import torch.distributed
import transformers
import wandb
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils import data

from ultravox.data import datasets
from ultravox.inference import infer
from ultravox.model import data_processing
from ultravox.model import ultravox_config
from ultravox.model import ultravox_model
from ultravox.model import ultravox_processing
from ultravox.model import wandb_utils
from ultravox.training import config_base
from ultravox.training import ddp_utils
from ultravox.training import evaluation

INPUT_EXAMPLE = {"text": "Transcribe\n<|audio|>", "audio": b"\x00\x00" * 16000}
OUTPUT_EXAMPLE = {"text": "Hello, world!"}


def fix_hyphens(arg: str):
    return re.sub(r"^--([^=]+)", lambda m: "--" + m.group(1).replace("-", "_"), arg)


def prepare_dataset(
    dataset_names: List[str],
    data_args: datasets.VoiceDatasetArgs,
    processor: ultravox_processing.UltravoxProcessor,
    train_on_inputs: bool,
    repeat_data: bool,
    num_samples: Optional[int] = None,
    include_alt_input: bool = False,  # whether to generate tensors for text-only input (e.g., used for KD training)
) -> data.IterableDataset:

    data_sets = [datasets.create_dataset(ds, data_args) for ds in dataset_names]
    interleave = datasets.InterleaveDataset(data_sets)
    ds_with_proc = data_processing.UltravoxDataproc(
        interleave, processor=processor, train_on_inputs=train_on_inputs, include_alt_input=include_alt_input
    )
    limited_ds = datasets.Range(ds_with_proc, num_samples=num_samples)
    return limited_ds


@record
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
        args=[fix_hyphens(arg) for arg in sys.argv[1:]],
    )

    transformers.set_seed(args.seed)

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
    audio_processor = transformers.AutoProcessor.from_pretrained(args.audio_model)
    processor = ultravox_processing.UltravoxProcessor(audio_processor, text_tokenizer)

    # Instantiate the model and processor
    config = ultravox_config.UltravoxConfig(
        audio_model_id=args.audio_model,
        text_model_id=args.text_model,
        text_model_lora_config=args.text_model_lora_config,
        audio_model_lora_config=args.audio_model_lora_config,
    )

    logging.info("Instantiating model...")

    # Since the model downloads the language model and audio encoder weights, we want one process to finish up
    # downloading before the others start in order to avoid race conditions.
    with ddp_utils.run_on_master_first(is_master):
        model = ultravox_model.UltravoxModel(config)

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
        )

    if args.model_load_dir:
        logging.info(f"Loading model state dict from {args.model_load_dir}")
        load_path = args.model_load_dir
        if wandb_utils.is_wandb_url(load_path):
            # Download the model from W&B. The main process should do the download while the others wait.
            with ddp_utils.run_on_master_first(is_master):
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
    # loss_config needs to be passed separately just for model training
    model.set_loss_config(args.loss_config)

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
    # We use multiple validation sets here so that the results are comparable even when training set changes
    # To make sure we can compare training and validation loss (e.g. for fine-tuning), we keep a special set
    # called "matchtrain" that uses the same data as the training set.
    val_sets = dict(
        # [("matchtrain", args.data_sets)]  # FIXME: see issue https://github.com/fixie-ai/ultravox/issues/58
        [(x, [x]) for x in args.val_sets]
        + [(f"text_{x}", [x]) for x in args.val_sets]
    )
    if is_master:
        train_dataset = prepare_dataset(
            dataset_names=args.data_sets,
            train_on_inputs=args.train_on_inputs,
            repeat_data=args.repeat_data,
            processor=processor,
            num_samples=args.num_samples,
            data_args=datasets.VoiceDatasetArgs(
                num_prompts=args.num_prompts,
                data_dir=args.data_dir,
                shuffle=args.shuffle_data,
                shuffle_seed=args.shuffle_seed,
                max_audio_duration_secs=args.max_audio_duration_secs,
                use_mds=args.mds,
                mds_batch_size=args.batch_size,
            ),
            include_alt_input=model.loss_config.require_alt_input()
        )
        val_ds_args = datasets.VoiceDatasetArgs(
            num_prompts=1,
            split=datasets.DatasetSplit.VALIDATION,
            data_dir=args.data_dir,
            shuffle=False,
            max_audio_duration_secs=16,
            use_mds=args.mds,
            mds_batch_size=args.batch_size,
        )
        val_ds_args_text = copy.copy(val_ds_args)
        val_ds_args_text.include_audio = False
        val_datasets = {
            k: prepare_dataset(
                dataset_names=val_sets[k],
                train_on_inputs=args.train_on_inputs,
                repeat_data=args.repeat_data,
                processor=processor,
                num_samples=args.val_num_samples,
                data_args=val_ds_args_text if k.startswith("text_") else val_ds_args,
                include_alt_input=model.loss_config.require_alt_input()
            )
            for k in val_sets
        }
        logging.info(
            f"Loaded {args.data_sets} data sets, sample limit: {args.num_samples} (val sample limit: {args.val_num_samples})"
        )
    else:
        # When using DDP with split_batches=True, the primary process will distribute the batches to the workers
        # The point of this is to avoid unnecessary data processing/downloading in the workers.
        train_dataset = datasets.EmptyDataset()
        val_datasets = {k: datasets.EmptyDataset() for k in val_sets}

    # Set up the data loader
    data_collator = datasets.DataCollatorForSeq2SeqWithAudio(tokenizer=text_tokenizer, include_alt_input=model.loss_config.require_alt_input())

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
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            use_cpu=args.device == "cpu",
            seed=args.seed + local_rank,
            report_to=args.report_logs_to,
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
        trainer.save_model(args.output_dir)
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
        inference = infer.LocalInference(
            model=model,
            processor=processor,
            tokenizer=text_tokenizer,
            device=args.device,
            dtype=dtype,
        )
        metrics = evaluation.evaluate(
            inference,
            data_dir=args.data_dir,
            num_procs=args.eval_num_procs,
            num_samples=args.eval_num_samples,
            max_new_tokens=args.eval_max_new_tokens,
            verbose=True,
        )
        if is_master:
            trainer.log(metrics)

        t_end = datetime.now()
        logging.info(f"eval end time: {t_end}")
        logging.info(f"elapsed: {t_end - t_start}")


if __name__ == "__main__":
    main()
