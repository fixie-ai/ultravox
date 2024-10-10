from typing import Optional, Tuple, Union

import torch
import transformers

from ultravox.data import datasets
from ultravox.inference import base
from ultravox.inference import utils
from ultravox.model import wandb_utils
from ultravox.ultravoxls import ultravoxls_model
from ultravox.ultravoxls import ultravoxls_processing

MAX_NEW_TOKENS = 1024


class LocalLSInference(base.VoiceInference):
    def __init__(
        self,
        model: ultravoxls_model.UltravoxLSModel,
        processor: ultravoxls_processing.UltravoxLSProcessor,
        collate_fn: ultravoxls_processing.DataCollatorForLSM,
        device: str,
    ):
        self.model = model.to(device).eval()
        self.processor = processor
        self.collate_fn = collate_fn

    def infer(
        self,
        sample: datasets.VoiceSample,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> base.VoiceOutput:
        inputs = self.processor.dataproc(sample)
        inputs = self.collate_fn([inputs])
        input_len = inputs["num_tokens"][0].item()
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        output_audio, output_len = self._generate(inputs, max_tokens, temperature)
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
        if wandb_utils.is_wandb_url(model_path):
            model_path = wandb_utils.download_model_from_wandb(model_path)

        model = ultravoxls_model.UltravoxLSModel.from_pretrained(model_path)
        model.to(device=device)

        # TODO: fix left padding for this to work right
        model.config.pad_to_multiple_of = 1

        processor = ultravoxls_processing.UltravoxLSProcessor()
        collate_fn = ultravoxls_processing.DataCollatorForLSM()

        super().__init__(
            model=model,
            processor=processor,
            collate_fn=collate_fn,
            device=device,
        )
