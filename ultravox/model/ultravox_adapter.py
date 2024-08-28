import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import transformers
import torch.nn.functional as F
import numpy as np

import logging
from transformers import WhisperConfig

from transformers.models.whisper import modeling_whisper as whisper
from transformers.models.wav2vec2 import modeling_wav2vec2 as wav2vec2
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers.models.whisper.configuration_whisper import WhisperConfig

from .ultravox_config import UltravoxConfig, UltravoxStackingAdapterConfig, UltravoxCFormerAdapterConfig

logger = logging.getLogger(__name__)

class RMSNorm(transformers.models.llama.modeling_llama.LlamaRMSNorm):
    def __init__(self, hidden_size: int, init: float = 1, eps: float = 1e-6):
        super().__init__(hidden_size=hidden_size, eps=eps)
        self.weight.data.fill_(init)

# currently attention_mask is not yet implemented in the forward method 
class UltravoxAdapter(nn.Module):
    def __init__(self, config: UltravoxConfig):
        super().__init__()
        audio_config: Union[Wav2Vec2Config, WhisperConfig] = config.audio_config
        text_config: transformers.LlamaConfig = config.text_config

        self.input_size = audio_config.hidden_size
        # self.hidden_size always matches audio_config.hidden_size 
        self.hidden_size = audio_config.hidden_size
        self.output_size = text_config.hidden_size

        self.post_ln = RMSNorm(self.hidden_size, init=config.norm_init)
        self.text_proj = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, audio_features: torch.Tensor, num_tokens: Optional[torch.Tensor]=None)  -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError("Subclasses must implement this method")

    def project_to_text(self, hidden_states):
        hidden_states = self.post_ln(hidden_states)
        hidden_states = self.text_proj(hidden_states)
        return hidden_states
    
    def get_audio_token_len(self, audio_frame_len: int, token_len: int) -> int:
        raise NotImplementedError("Subclasses must implement this method")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, init: float = 1.0):
        super().__init__()
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(dim) * init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        rms = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) / d + self.eps)
        x = x / rms
        return x * self.weight

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class StackAudioFrames(nn.Module):
    def __init__(self, stack_factor: int):
        super().__init__()
        self.stack_factor = stack_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stack_factor == 1:
            return x
        b, t, d = x.shape
        pad = (self.stack_factor - (t % self.stack_factor)) % self.stack_factor
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        return x.reshape(b, -1, d * self.stack_factor)


class StackingAdapter(UltravoxAdapter):
    def __init__(self, config: UltravoxConfig):
        super().__init__(config)

        self.adapter_config = UltravoxStackingAdapterConfig(**config.adapter_config)

        self._pad_and_stack = StackAudioFrames(self.adapter_config.stack_factor)
        stacked_size = self.input_size * self.adapter_config.stack_factor
        self.ln_pre = RMSNorm(stacked_size, init=config.norm_init)
        # swiglu reduces dimension by 2, so we double it here before swigu to keep effective hidden size consistent.
        intermediate_size = self.hidden_size * 2 if self.adapter_config.activation == "swiglu" else self.hidden_size
        self.linear_1 = nn.Linear(stacked_size, intermediate_size, bias=False)
        self.act = transformers.activations.get_activation(self.adapter_config.activation)

    def get_audio_token_len(self, audio_frame_len: int, token_len: int) -> int:
         return int(np.ceil(audio_frame_len / self.adapter_config.stack_factor))
         
    def forward(self, audio_features: torch.Tensor, num_tokens: Optional[torch.Tensor]=None)  -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self._pad_and_stack(audio_features)
        hidden_states = self.ln_pre(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.project_to_text(hidden_states)
        return hidden_states, None


class CFormerAdapter(UltravoxAdapter):
    def __init__(self, config: UltravoxConfig):
        super().__init__(config)

        adapter_config = UltravoxCFormerAdapterConfig(**config.adapter_config)

        self.num_pre_cif_layers = adapter_config.num_pre_cif_layers
        self.num_post_cif_layers = adapter_config.num_post_cif_layers

        if self.num_pre_cif_layers or self.num_post_cif_layers:
            if config.audio_config.model_type == "whisper":
                transformer_layer_class = whisper.WhisperEncoderLayer
            elif config.audio_config.model_type == "wav2vec2":
                transformer_layer_class = wav2vec2.Wav2Vec2EncoderLayer
            else:
                raise ValueError(f"Unsupported audio model type: {config.audio_config.model_type}")

        if self.num_pre_cif_layers > 0:
            self.pre_cif_layers = nn.ModuleList(
                [transformer_layer_class(config.audio_config) for _ in range(self.num_pre_cif_layers)]
            )
        
        self.cif_proj = nn.Linear(self.hidden_size-1, self.hidden_size)

        if self.num_post_cif_layers > 0:
            self.post_cif_layers = nn.ModuleList(
                [transformer_layer_class(config.audio_config) for _ in range(self.num_post_cif_layers)]
            )

    def get_audio_token_len(self, audio_frame_len: int, token_len: int) -> int:
         return token_len
    
    # This implements the continuous integrate-and-fire mechanism adapted from this paper: https://arxiv.org/abs/1905.11235
    # TODO: add support for attention_mask
    def forward_cif(self, hidden_states: torch.Tensor, alphas: torch.Tensor, num_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = hidden_states.device
        B, T, _ = hidden_states.size()

        max_num_tokens = num_tokens.max()

        # loop vars
        integrate = torch.zeros([B], device=device)  # accumulated alpha value that hasn't benen fired yet
        remainds = torch.zeros([B], device=device)  # reamining alpha value from recent firing
        token_index = torch.zeros([B], dtype=torch.long, device=device)  # num of fires that has happened

        # weights: B x max_num_tokens x T, weights[i, j, k] is the contribution of the k-th speech feature to the j-th text/speech token for the i-th sample 
        weights = torch.zeros((B, max_num_tokens, T), device=device)
        for t in range(T):
            if t > 0:
                weights[:, :, t - 1].scatter_add_(dim=1, index=token_index.unsqueeze(1), src=remainds.unsqueeze(1))

            alpha = alphas[:, t]
            alpha_needed = 1 - integrate
            integrate += alpha
            ready_to_fire = integrate >= 1.0

            while True:  # allow repeated firing if integrate > threshold
                integrate = torch.where(ready_to_fire, integrate - 1, integrate)
                alpha_integrated = torch.where(ready_to_fire, alpha_needed, alpha)

                weights[:, :, t].scatter_(dim=1, index=token_index.unsqueeze(1), src=alpha_integrated.unsqueeze(1))
                remainds = alpha - alpha_integrated

                token_index = token_index + ready_to_fire.type_as(token_index)
                token_index = torch.minimum(token_index, num_tokens - 1)

                alpha = remainds
                alpha_needed = 1
                ready_to_fire = integrate >= 1.0
                if not ready_to_fire.any():
                    break

        # the resulting hidden_states contains the hidden states of speech tokens right after CIF mechanism
        hidden_states = weights.type_as(hidden_states).bmm(hidden_states)

        return hidden_states


    def forward(self, audio_features: torch.Tensor, num_tokens: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = audio_features
        T = hidden_states.size(1)

        for layer in self.pre_cif_layers:
            hidden_states = layer(hidden_states, None, None)[0]

        # alphas is computed from the last element of hidden_states using a sigmoid function, and used to assign speech features to text/speech tokens.
        alphas = torch.sigmoid(hidden_states[:, :, -1])  
        pred_num_tokens = alphas.sum(-1)

        if self.training:
            if num_tokens is None:
                raise ValueError("num_tokens must be provided in training mode")
        else:
            # num_tokens is determined by accumulated predicted alpha values in inference mode
            num_tokens = torch.round(pred_num_tokens).int()
            # force the number of predicted tokens to be at least 1 in non-streaming mode
            # this will break streaming mode and needs to be updated
            num_tokens[num_tokens < 1] = 1

        # scale alphas so that the sum of alphas is equal to num_tokens
        alphas = alphas * (num_tokens / pred_num_tokens)[:, None].repeat(1, T)

        # remove the last element of hidden_states and apply CIF mechanism
        hidden_states = self.forward_cif(hidden_states[:, :, :-1], alphas, num_tokens)
        # project back to self.hidden_size
        hidden_states = self.cif_proj(hidden_states)

        for layer in self.post_cif_layers:
            hidden_states = layer(hidden_states, None, None)[0]

        hidden_states = self.project_to_text(hidden_states)

        return hidden_states, pred_num_tokens
    
transformers.activations.ACT2FN["swiglu"] = SwiGLU
