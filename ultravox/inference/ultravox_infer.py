import logging
from typing import Optional

import transformers

from ultravox.inference import infer
from ultravox.model import file_utils
from ultravox.model import ultravox_model
from ultravox.model import ultravox_processing
from ultravox.training import ddp_utils
from ultravox.training.helpers import prefetch_weights
from ultravox.utils import device_helpers


class UltravoxInference(infer.LocalInference):
    def __init__(
        self,
        model_path: str,
        audio_processor_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        device: Optional[str] = None,
        use_fsdp: bool = False,
        use_tp: bool = False,
        data_type: Optional[str] = None,
        conversation_mode: bool = False,
        chat_template: Optional[str] = None,
        enable_thinking: bool = False,
        thinking_regex: Optional[str] = None,
    ):
        """
        Args:
            model_path: can refer to a HF hub model_id, a local path, or a W&B artifact
                Examples:
                    fixie-ai/ultravox
                    runs/llama2_asr_gigaspeech/checkpoint-1000/
                    wandb://fixie/ultravox/model-llama2_asr_gigaspeech:v0
            audio_processor_id: model_id for the audio processor to use. If not provided, it will be inferred
            tokenizer_id: model_id for the tokenizer to use. If not provided, it will be inferred
            device: where to put the model and data
            data_type: data type to use for the model
            conversation_mode: if true, keep track of past messages in a conversation
            chat_template: template for formatting chat messages
            enable_thinking: if true, enable thinking tokens
            thinking_regex: regex pattern for extracting thinking content from responses
        """
        assert not use_tp or not use_fsdp, "tp and fsdp cannot be used together"

        device = device or device_helpers.default_device()
        dtype = device_helpers.get_dtype(data_type)

        with ddp_utils.run_on_master_first():
            model_path = file_utils.download_dir_if_needed(model_path)
            prefetch_weights.download_sub_models(model_path)

        tp_plan = "auto" if use_tp else None
        logging.info(
            f"Loading model from {model_path} with dtype {dtype}, tp_plan {tp_plan}, use_fsdp {use_fsdp} on {device}"
        )
        model = ultravox_model.UltravoxModel.from_pretrained(
            model_path, torch_dtype=dtype, tp_plan=tp_plan
        )
        model.merge_and_unload()

        ddp_utils.model_to_device(model, device, use_fsdp=use_fsdp, use_tp=use_tp)
        model.to(dtype=dtype)

        tokenizer_id = tokenizer_id or model_path
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        # tincans-ai models don't set audio_model_id, instead audio_config._name_or_path has the
        # model name. A default value is added just as a precaution, but it shouldn't be needed.
        audio_processor = transformers.AutoProcessor.from_pretrained(
            audio_processor_id
            or model.config.audio_model_id
            or model.config.audio_config._name_or_path
            or "openai/whisper-tiny",
        )

        processor = ultravox_processing.UltravoxProcessor(
            audio_processor,
            tokenizer=tokenizer,
            stack_factor=model.config.stack_factor,
            audio_context_size=model.audio_tower_context_length,
        )

        super().__init__(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            dtype=dtype,
            conversation_mode=conversation_mode,
            chat_template=chat_template,
            enable_thinking=enable_thinking,
            thinking_regex=thinking_regex,
        )
