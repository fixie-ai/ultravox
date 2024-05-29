import glob
import logging
import os
import typing as t
from dataclasses import asdict
from dataclasses import dataclass

import evaluate
import numpy as np
import pyrallis
import safetensors.torch
import torch
import torch.distributed.elastic.multiprocessing.errors
import transformers
import transformers.models
import wandb

from . import configs_base
from . import data
from . import env
from .models import multimodal as multimodal_models


def training_function(config: configs_base.TrainConfig):
    # set seed
    transformers.set_seed(config.seed)
    np.random.seed(config.seed)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dtype = (
        torch.bfloat16
        if config.bf16
        else (torch.float16 if config.fp16 else torch.float32)
    )

    # config.model.torch_dtype = dtype

    model = multimodal_models.SpeechLM(config.model)
    model.apply_lora_configs(config=config.freezing_config)
    model.print_trainable_parameters()
    processor = multimodal_models.SpeechLMProcessor.from_config(config.model)

    if not config.use_cpu:
        model = model.to("cuda")

    model.llm.config.use_cache = False
    # set pad token to unk. we want this to be different from the eos token
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"  # to allow for fast generation

    # TODO: move inside model
    sampling_rate = 16_000  # 16kHz

    print(f"Audio token freq: {round(sampling_rate / processor.total_audio_stride)} Hz")

    if config.model_load_path:
        logging.info(f"Loading model state dict from {config.model_load_path}")
        for path in glob.glob(str(config.model_load_path)):
            state_dict = safetensors.torch.load_file(path)
            mismatch = model.load_state_dict(state_dict, strict=False)
            if mismatch.unexpected_keys:
                raise ValueError(
                    f"Unexpected keys in state dict: {mismatch.unexpected_keys}"
                )

    train_val_dataset = data.get_dataset(
        dataset_name=config.dataset_name,
        sampling_rate=sampling_rate,
        dev_env=config.use_cpu,
        streaming=config.dataset_streaming,
        max_duration_in_seconds=config.max_audio_duration_in_seconds,
        val_max_num_samples=128,
        # val_max_num_samples=8,
    )
    preproc = data.AudioTextTokenizer(
        processor.audio_processor,
        processor.tokenizer,
        processor.total_audio_stride,
        cfg=config.audio_tokenizer_config,
        # inference_mode=True,
    )
    train_ds, val_ds = [ds.map(preproc) for ds in train_val_dataset]

    data_collator = data.DataCollatorForSeq2SeqWithAudio(
        processor.tokenizer,
        pad_to_multiple_of=8,  # This won't be needed when we move back to concat model
        return_tensors="pt",
        padding=True,
        audio_dtype=dtype,
    )

    # eval_dataset = data.get_dataset_librispeech(train=False)
    # TODO: wrap with pytorch dataloader and sampler?

    # Define training args
    training_args = transformers.Seq2SeqTrainingArguments(
        # torch_compile=True,
        optim=config.optimizer_type,
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        seed=config.seed,
        use_cpu=config.use_cpu,
        fp16=config.fp16,  # Be careful when using FP16
        bf16=config.bf16,  # Use BF16 if available
        learning_rate=config.lr,
        warmup_ratio=config.lr_warmup_ratio,
        warmup_steps=config.lr_warmup_steps,
        weight_decay=config.weight_decay,
        # TODO: couldn't find "exponential decaying schedule" in HF trainer.
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        deepspeed=config.deepspeed,
        # gradient_checkpointing=config.gradient_checkpointing,
        # logging & evaluation strategies
        logging_dir=config.output_dir / "logs",
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        dataloader_pin_memory=False,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        eval_accumulation_steps=config.eval_accumulation_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        run_name=config.exp_name,
        dataloader_num_workers=config.num_workers,
        report_to=config.report_logs_to,
        predict_with_generate=True,
        generation_config=transformers.GenerationConfig(max_new_tokens=1),
    )

    # TODO: are we shuffling the data?

    # Create Trainer instance
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=ComputeMetrics(processor.tokenizer),
    )

    # Start training

    if "wandb" in config.report_logs_to:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "ultravox"),
            config=asdict(config),
            name=config.exp_name,
            # TODO: run name, etc from HF callback
        )

    print("Initial evaluation:", trainer.evaluate())
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    trainer.save_model(config.output_dir)


@dataclass
class ComputeMetrics:
    tokenizer: transformers.LlamaTokenizer
    preamble: str = "Transcript:"

    def __post_init__(self):
        self.cer_metric = evaluate.load("cer")
        self.wer_metric = evaluate.load("wer")

    def _remove_preamble(self, text: str) -> str:
        """
        If a preamble is present, remove it and return the text after it.
        Will only remove the first occurence of the preamble.
        Will not modify the text (other than stripping spaces) if the preamble is not found.

        Examples:
            >>> _remove_preamble("hello world")
            "hello world"
            >>> _remove_preamble("Transcript: hello")
            "hello"
            >>> _remove_preamble("Transcript: hello Transcript: world")
            "hello Transcript: world"
        """
        return text.split(self.preamble, maxsplit=1)[-1].strip()

    def __call__(self, eval_preds: transformers.EvalPrediction):
        targets = eval_preds.label_ids
        logits = eval_preds.predictions
        if logits.ndim > targets.ndim:
            logits = logits.argmax(axis=-1)

        if logits.shape[1] == targets.shape[1] + 1:
            # This is because we set max_new_tokens to 1 so that we can use
            # predict_with_generate which is faster.
            # I can make this cover more cases, but I actually want it to fail
            # since I don't know whether that would be a bug or not.
            logits = logits[:, : targets.shape[1]]

        ref_strs: t.List[str] = []
        pred_strs: t.List[str] = []
        accuracies: t.List[float] = []
        # for logit, target in zip(logits[..., :-1], targets[..., 1:]):
        # No need to shift logits left? Is that because of Seq2Seq?
        for logit, target in zip(logits, targets):
            idx = target > 0
            # Remove special tokens <s>: 1, <unk>: 0, and </s>: 2
            ref_ids = [x for x in target[idx] if x > 2]
            pred_ids = [x for x in logit[idx] if x > 2]
            # CER complains if the string is empty
            ref_strs.append(self.tokenizer.decode(ref_ids).strip() or "<empty>")
            pred_strs.append(self.tokenizer.decode(pred_ids).strip() or "<empty>")

            # TODO: is text sometimes empty?
            accuracies.append(
                np.mean(target[idx] == logit[idx]) if idx.sum() > 0 else np.nan
            )

        ref_strs = [self._remove_preamble(x).lower() for x in ref_strs]
        pred_strs = [self._remove_preamble(x).lower() for x in pred_strs]

        # for ref, pred in zip(ref_strs, pred_strs):
        #     print(f"REF: {ref}\nPRED: {pred}\n")

        return {
            # "perplexity" TODO?
            "cer_assisted": self.cer_metric.compute(
                predictions=pred_strs, references=ref_strs
            ),
            "wer_assisted": self.wer_metric.compute(
                predictions=pred_strs, references=ref_strs
            ),
            # "cheat_ratio": cheat_ratio,
            "token_accuracy": np.nanmean(accuracies),
        }


def main():
    logging.basicConfig(level=logging.DEBUG)
    pyrallis.decode.register(
        t.List[str], lambda x: x.split(",") if isinstance(x, str) else x
    )
    cfg = pyrallis.parse(config_class=configs_base.TrainConfig)
    logging.warning(f"\n\nRunning with config:\n{cfg}\n\n")

    training_function(cfg)

    wandb.finish()


if __name__ == "__main__":
    env.set_env_vars_azure()
    main()
