import abc
from typing import Callable

import transformers

from ultravox.data import datasets
from ultravox.model import ultravox_config
from ultravox.model import ultravox_data_proc
from ultravox.model import ultravox_model
from ultravox.model import ultravox_pipeline
from ultravox.model import ultravox_processing
from ultravox.training import config_base


class ModelPack(abc.ABC):
    model: transformers.PreTrainedModel
    processor: transformers.ProcessorMixin
    config: transformers.PretrainedConfig
    data_collator: Callable

    @abc.abstractmethod
    def wrap_with_data_proc(
        self, dataset: datasets.SizedIterableDataset
    ) -> datasets.Dataproc:
        pass

    @abc.abstractmethod
    def get_pipeline(self) -> transformers.Pipeline:
        pass

    @abc.abstractmethod
    def get_text_tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        pass

    @abc.abstractmethod
    def change_text_padding_side(self, padding_side: str):
        pass


class UltravoxModelPack(ModelPack):
    def __init__(self, args: config_base.TrainConfig):
        self.args = args
        self.text_tokenizer: transformers.PreTrainedTokenizerFast = (
            transformers.AutoTokenizer.from_pretrained(args.text_model)
        )
        self.text_tokenizer.padding_side = "right"
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        audio_processor = transformers.AutoProcessor.from_pretrained(args.audio_model)

        # Instantiate the model and processor
        self.config = ultravox_config.UltravoxConfig(
            audio_model_id=args.audio_model,
            text_model_id=args.text_model,
            text_model_lora_config=args.text_model_lora_config,
            audio_model_lora_config=args.audio_model_lora_config,
            torch_dtype=args.data_type,
            pad_token_id=self.text_tokenizer.eos_token_id,
            projector_ln_mid=args.projector_ln_mid,
        )

        # Instantiate the model
        self.model: ultravox_model.UltravoxModel = ultravox_model.UltravoxModel(
            self.config
        )

        self.processor = ultravox_processing.UltravoxProcessor(
            audio_processor,
            self.text_tokenizer,
            audio_context_size=self.model.audio_tower_context_length,
        )

        # loss_config needs to be passed separately just for model training
        if args.loss_config is not None:
            self.model.set_loss_config(args.loss_config)

        # Set up the data loader
        self.data_collator = ultravox_processing.DataCollatorForSeq2SeqWithAudio(
            tokenizer=self.text_tokenizer,
            include_alt_fields=self.model.loss_config.requires_alt_fields,
        )

        assert self.model.get_input_embeddings().num_embeddings == len(
            self.text_tokenizer
        ), f"Model and tokenizer mismatch: {self.model.get_input_embeddings().num_embeddings} != {len(self.text_tokenizer)}"

        self.model.language_model.config.use_cache = False
        if args.disable_layerdrop and hasattr(
            self.model.audio_tower.config, "layerdrop"
        ):
            # layerdrop causes issues when training with DDP
            # https://github.com/huggingface/transformers/issues/17116#issuecomment-1121340890
            self.model.audio_tower.config.layerdrop = 0.0

    def wrap_with_data_proc(self, dataset: datasets.SizedIterableDataset):
        return ultravox_data_proc.UltravoxDataproc(
            dataset,
            processor=self.processor,
            train_on_inputs=self.args.train_on_inputs,
            include_alt_fields=self.model.loss_config.requires_alt_fields,
            max_response_tokens=self.args.max_response_tokens,
        )

    def get_pipeline(self):
        return ultravox_pipeline.UltravoxPipeline(
            self.model, tokenizer=self.text_tokenizer, device=self.model.device
        )

    def get_text_tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        return self.text_tokenizer

    def change_text_padding_side(self, padding_side: str):
        self.text_tokenizer.padding_side = padding_side


def create_model_pack(args: config_base.TrainConfig) -> ModelPack:
    if args.model_type == "ultravox":
        return UltravoxModelPack(args)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
