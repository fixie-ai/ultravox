import abc
import fnmatch
import glob
import logging
import os
from typing import Any, Callable, Dict

import safetensors
import torch
import transformers

from ultravox.data import datasets
from ultravox.model import file_utils
from ultravox.model import ultravox_config
from ultravox.model import ultravox_data_proc
from ultravox.model import ultravox_model
from ultravox.model import ultravox_pipeline
from ultravox.model import ultravox_processing
from ultravox.model import ultravox_tokenizer
from ultravox.training import config_base
from ultravox.utils import device_helpers


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
            ultravox_tokenizer.from_pretrained_text_tokenizer(args.text_model)
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
            audio_token_index=ultravox_tokenizer.get_audio_token_id(
                self.text_tokenizer
            ),
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

        model_vocab_size = self.model.get_input_embeddings().num_embeddings
        tokenizer_vocab_size = len(self.text_tokenizer)

        if model_vocab_size != tokenizer_vocab_size + (
            1
            if ultravox_tokenizer.get_audio_token_id(self.text_tokenizer) is not None
            else 0
        ):
            logging.warning(
                "Vocabulary size mismatch:"
                f" Model has {model_vocab_size} tokens,"
                f" but the tokenizer has {tokenizer_vocab_size} tokens."
            )

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
            loss_mask_type=self.args.loss_mask_type,
            augmentation=self.args.get_train_augmentation(),
            include_alt_fields=self.model.loss_config.requires_alt_fields,
            max_response_tokens=self.args.max_response_tokens,
            chat_template=self.args.chat_template,
        )

    def get_pipeline(self):
        return ultravox_pipeline.UltravoxPipeline(
            self.model, tokenizer=self.text_tokenizer, device=self.model.device
        )

    def get_text_tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        return self.text_tokenizer

    def change_text_padding_side(self, padding_side: str):
        self.text_tokenizer.padding_side = padding_side


class LLMOnlyModelPack(ModelPack):
    def __init__(self, args: config_base.TrainConfig):
        self.args = args
        self.text_tokenizer: transformers.PreTrainedTokenizerFast = (
            transformers.AutoTokenizer.from_pretrained(args.text_model)
        )
        self.text_tokenizer.padding_side = "right"
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        # Instantiate the model and processor
        self.config = ultravox_config.UltravoxConfig(
            audio_model_id=None,
            text_model_id=args.text_model,
            text_model_lora_config=args.text_model_lora_config,
            audio_model_lora_config=None,
            torch_dtype=args.data_type,
            pad_token_id=self.text_tokenizer.eos_token_id,
            projector_ln_mid=args.projector_ln_mid,
            llm_only_training=True,
        )

        # Instantiate the model
        self.model: ultravox_model.UltravoxModel = ultravox_model.UltravoxModel(
            self.config
        )

        self.processor = ultravox_processing.UltravoxProcessor(
            None,
            self.text_tokenizer,
            audio_context_size=None,
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
            include_alt_fields=self.model.loss_config.requires_alt_fields,
            max_response_tokens=self.args.max_response_tokens,
            chat_template=self.args.chat_template,
            loss_mask_type=self.args.loss_mask_type,
        )

    def get_pipeline(self):
        return ultravox_pipeline.UltravoxPipeline(
            self.model, tokenizer=self.text_tokenizer, device=self.model.device
        )

    def get_text_tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        return self.text_tokenizer

    def change_text_padding_side(self, padding_side: str):
        self.text_tokenizer.padding_side = padding_side


def create_model_pack(config: config_base.TrainConfig) -> ModelPack:
    model_pack: ModelPack | None = None
    if config.model_type == "ultravox":
        model_pack = (
            LLMOnlyModelPack(config)
            if config.llm_only_training
            else UltravoxModelPack(config)
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    if config.model_load_dir and (
        not config.use_fsdp or device_helpers.is_global_master()
    ):
        logging.info(f"Loading model state dict from {config.model_load_dir}")
        load_path = file_utils.download_dir_if_needed(config.model_load_dir)
        if os.path.isdir(load_path):
            load_path = os.path.join(load_path, "model*.safetensors")
        paths = glob.glob(load_path)
        assert len(paths) > 0, f"No model files found at {load_path}"
        for path in paths:
            state_dict = safetensors.torch.load_file(path)
            # Handle parameter name mapping for LoRA compatibility
            if isinstance(model_pack.model, ultravox_model.UltravoxModel):
                # Handle audio tower parameters
                state_dict = rename_state_dict_for_lora(
                    state_dict,
                    "audio_tower",
                    model_pack.model.config.audio_model_lora_config,
                )
                # Handle language model parameters
                state_dict = rename_state_dict_for_lora(
                    state_dict,
                    "language_model",
                    model_pack.model.config.text_model_lora_config,
                )
            # Filter state_dict by parameter names if model_load_parameters is specified
            if config.model_load_parameters:
                filtered_state_dict = {}
                skipped_keys = []
                for key, value in state_dict.items():
                    # Check if the parameter name matches any of the patterns
                    if any(
                        fnmatch.fnmatch(key, pattern)
                        for pattern in config.model_load_parameters
                    ):
                        filtered_state_dict[key] = value
                    else:
                        skipped_keys.append(key)

                if device_helpers.is_global_master():
                    logging.info(
                        f"Selectively loading parameters: {len(filtered_state_dict)} matched, {len(skipped_keys)} skipped"
                    )
                    if config.verbose:
                        logging.info(
                            f"Matched parameters: {sorted(filtered_state_dict.keys())}"
                        )
                        logging.info(f"Skipped parameters: {sorted(skipped_keys)}")

                state_dict = filtered_state_dict

            mismatch = model_pack.model.load_state_dict(state_dict, strict=False)
            if mismatch.unexpected_keys:
                raise ValueError(
                    f"Unexpected keys in state dict: {mismatch.unexpected_keys}"
                )
    return model_pack


def rename_state_dict_for_lora(
    state_dict: Dict[str, torch.Tensor], param_prefix: str, lora_config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """Handle compatibility between state dict and model with LoRA enabled.

    This function handles loading pre-LoRA weights into a model with LoRA enabled.
    Only parameters in modules specified by target_modules in the LoRA config will be transformed.

    Args:
        state_dict: The state dict to load
        param_prefix: The prefix for parameters (e.g. "audio_tower" or "language_model")
        lora_config: The LoRA configuration for this component

    Returns:
        The state dict with parameter names mapped if needed
    """
    # Check if LoRA is enabled for this component
    lora_enabled = lora_config.get("r", 0) > 0
    if not lora_enabled:
        return state_dict

    # Get target modules from LoRA config
    target_modules = lora_config.get("target_modules", [])

    # Check if state dict has LoRA-style names
    has_lora_names = any(
        f"{param_prefix}.base_model" in key for key in state_dict.keys()
    )
    if has_lora_names:
        return state_dict

    # Need to rename parameters to LoRA format
    new_state_dict = state_dict.copy()
    for key in list(state_dict.keys()):
        if key.startswith(f"{param_prefix}.") and "base_model" not in key:
            # For LoRA, we need to add base_layer suffix to the weight
            new_key = key.replace(
                f"{param_prefix}.", f"{param_prefix}.base_model.model."
            )

            # Check if this parameter is in a target module and replace the module name
            for module in target_modules:
                if f".{module}." in new_key:
                    new_key = new_key.replace(f".{module}.", f".{module}.base_layer.")
                    break

            new_state_dict[new_key] = state_dict[key]
            del new_state_dict[key]
            logging.info(f"renamed state dict key {key} -> {new_key}")
    return new_state_dict
