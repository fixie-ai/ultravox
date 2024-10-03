from typing import Optional, Tuple, Union

import torch
import transformers

from ultravox.data import datasets
from ultravox.inference import base
from ultravox.inference import utils
from ultravox.model import wandb_utils
from ultravox.ultravoxls import ultravoxls_processing
from ultravox.ultravoxls.ultravoxls_config import UltravoxLSConfig
from ultravox.ultravoxls.ultravoxls_model import UltravoxLSModel

MAX_NEW_TOKENS = 1024


class LocalLSInference(base.VoiceInference):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        processor: ultravoxls_processing.UltravoxLSProcessor,
        device: str,
        dtype: torch.dtype,
    ):
        self.model = model.to(device).to(dtype).eval()
        self.processor = processor
        self.dtype = dtype

    def infer(
        self,
        sample: datasets.VoiceSample,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> base.VoiceOutput:
        inputs = self.processor.dataproc(sample)
        input_len = inputs["input_ids"].shape[1]
        output = self._generate(inputs, max_tokens, temperature)
        output_tokens = output.sequences[0][input_len:]
        for i in range(3 - output_tokens.dim()):
            output_tokens = output_tokens.unsqueeze(0)
        output_audio = self.processor.decode(output_tokens, skip_special_tokens=True)
        output_len = len(output_tokens)
        return base.VoiceOutput(output_audio, input_len, output_len)

    @torch.inference_mode()
    def _generate(
        self,
        inputs: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        streamer: Optional[transformers.TextStreamer] = None,
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
        return_dict_in_generate: Optional[bool] = True,
    ):
        temperature = temperature or None
        do_sample = temperature is not None

        return self.model.generate(
            **inputs,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
            temperature=temperature,
            streamer=streamer,
            past_key_values=past_key_values,
            return_dict_in_generate=return_dict_in_generate,
        )


class UltravoxLSInference(LocalLSInference):
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        data_type: Optional[str] = None,
    ):
        """
        Args:
            model_path: can refer to a HF hub model_id, a local path, or a W&B artifact
                Examples:
                    fixie-ai/ultravox
                    runs/llama2_asr_gigaspeech/checkpoint-1000/
                    wandb://fixie/ultravox/model-llama2_asr_gigaspeech:v0
            device: where to put the model and data
            data_type: data type to use for the model
        """
        device = device or utils.default_device()
        dtype = utils.get_dtype(data_type) if data_type else utils.default_dtype()
        if wandb_utils.is_wandb_url(model_path):
            model_path = wandb_utils.download_model_from_wandb(model_path)

        config = UltravoxLSConfig(text_model_id=model_path)

        model = UltravoxLSModel(config)
        model.to(dtype=dtype, device=device)

        processor = ultravoxls_processing.UltravoxLSProcessor(model_device=model.device)

        super().__init__(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
        )

        self.data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer=processor.tokenizer,
        )
