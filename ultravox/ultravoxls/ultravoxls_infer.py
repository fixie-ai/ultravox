import os
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
import transformers
import ultravoxls_processing
from huggingface_hub import hf_hub_download

from ultravox.data import datasets
from ultravox.inference import base
from ultravox.inference import utils
from ultravox.model import wandb_utils
from ultravox.tokenizer.wav_tokenizer import CustomWavTokenizer
from ultravox.ultravoxls.ultravoxls_config import UltravoxLSConfig
from ultravox.ultravoxls.ultravoxls_model import UltravoxLSModel

SAMPLE_RATE_LS = 24000
MAX_NEW_TOKENS = 1024


class LocalLSInference(base.VoiceInference):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        processor: ultravoxls_processing.UltravoxLSProcessor,
        tokenizer: transformers.PreTrainedTokenizer,
        device: str,
        dtype: torch.dtype,
    ):
        self.model = model.to(device).to(dtype).eval()
        self.tokenizer = tokenizer
        self.processor = processor
        self.dtype = dtype

    def infer(
        self,
        sample: datasets.VoiceSample,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> base.VoiceOutput:
        inputs = self._dataproc(sample)
        input_len = inputs["input_ids"].shape[1]
        output = self._generate(inputs, max_tokens, temperature)
        output_tokens = output.sequences[0][input_len:]
        for i in range(3 - output_tokens.dim()):
            output_tokens = output_tokens.unsqueeze(0)
        output_audio = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        output_len = len(output_tokens)
        return base.VoiceOutput(output_audio, input_len, output_len)

    def _dataproc(self, sample: datasets.VoiceSample):
        if sample.audio is not None:
            audio = sample.audio
            sample_rate = sample.sample_rate
            # Normalize audio to float32.
            if audio.dtype == np.int16:
                audio = audio / np.float32(32768.0)
            if audio.dtype not in [np.float64, np.float32]:
                raise ValueError("Audio must be float64 or float32 or int16")

            # Convert to tensor, resampling to 24kHz if needed.
            if sample_rate != SAMPLE_RATE_LS:
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE_LS
                )
            audio_input = torch.from_numpy(audio)
            if audio_input.ndim == 1:
                audio_input = audio_input.unsqueeze(0)

        else:
            raise ValueError("Audio input is required for ultravoxls inference")

        inputs = self.processor(
            audio=audio_input, return_tensors="pt", sampling_rate=SAMPLE_RATE_LS
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

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

        # CustomWavTokenizer
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(
            root_dir,
            "tokenizer",
            "configs",
            "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        )
        hf_model_name = "novateur/WavTokenizer"
        checkpoint_file = "WavTokenizer_small_600_24k_4096.ckpt"
        model_path = hf_hub_download(repo_id=hf_model_name, filename=checkpoint_file)
        tokenizer = CustomWavTokenizer(config_path, model_path)

        processor = ultravoxls_processing.UltravoxLSProcessor(tokenizer=tokenizer)

        super().__init__(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )

        self.data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
        )
