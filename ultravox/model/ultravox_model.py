import logging
from typing import Any, Dict, Optional, Set, Tuple, Union

import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import transformers.activations
import transformers.modeling_outputs
import transformers.models

# We must use relative import in this directory to allow uploading to HF Hub
# Even "from . import X" pattern doesn't work (undocumented and unclear why)
from .ultravox_config import UltravoxConfig
from .whisper_model_modified import WhisperEncoder as ModifiedWhisperEncoder


class UltravoxModel(
    transformers.LlamaPreTrainedModel,
    transformers.GenerationMixin,
):
    """
    The Ultravox model which consists of an audio encoder and a language model.

    Audio input is processed by the audio encoder, then every `stack_factor` frames are stacked together and
    projected to the language model's embedding space using a few linear layers.
    The text is embedded by the language model as usual and then the audio and text embeddings are merged together.

    A special token `<|audio|>` is used to indicate the start of the audio embeddings in the merged embeddings.

    Parameters:
        config: Model configuration class with all the parameters of the model.
    """

    config_class = UltravoxConfig
    config: UltravoxConfig  # for type hinting
    _no_split_modules = ["Wav2Vec2Model", "WhisperEncoder", "LlamaDecoderLayer"]

    def __init__(self, config: UltravoxConfig):
        super().__init__(config)

        self.keep_params: Set[str] = set()
        self.vocab_size = config.vocab_size

        self.audio_tower = self._create_audio_tower(config)
        self.multi_modal_projector = UltravoxProjector(config)
        self.language_model = self._create_language_model(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _setup_cache(
        self, cache_cls, max_batch_size: int, max_cache_len: Optional[int] = None
    ):
        self.language_model._setup_cache(cache_cls, max_batch_size, max_cache_len)

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.language_model._reorder_cache(past_key_values, beam_idx)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ) -> Union[Tuple, transformers.modeling_outputs.CausalLMOutputWithPast]:
        """
        Forward pass for the Ultravox model.

        `input_ids` are the tokenized text input. They are embedded by the language model as usual.
        `audio_values` are processed by the audio encoder and then every `stack_factor` frames are stacked together and
        projected to the language model's embedding space using a few linear layers.
        The audio and text embeddings are merged together. A special token `<|audio|>` is used to indicate the start
        of the audio embeddings in the merged embeddings.

        Args:
            input_ids: The tokenized text input.
            audio_values: The processed audio values.
            inputs_embeds: The embeddings for the input tokens.
            labels: The tokenized text labels.
            attention_mask: The attention mask for the input.
            position_ids: The position ids for the input.
            past_key_values: The past key value cache for the language model attention layers.
            **kwargs: Additional keyword arguments. Passed directly to the language model.
        """
        if inputs_embeds is None:
            # B x T  ->  B x T x D
            inputs_embeds = self.get_input_embeddings().forward(input_ids)

        if audio_values is not None:
            assert (
                audio_token_start_idx is not None and audio_token_len is not None
            ), "audio_token_start_idx and audio_token_len must be provided if audio_values are provided."
            assert (
                len(audio_token_start_idx) == len(audio_token_len) == len(audio_values)
            ), "audio_token_start_idx, audio_token_len, and audio_values must have the same batch size."

            # B x A/3200 x D
            audio_tower_output = self.audio_tower.forward(
                audio_values
            ).last_hidden_state
            audio_tower_output = audio_tower_output.to(inputs_embeds.dtype)

            audio_embeds = self.multi_modal_projector.forward(audio_tower_output)

            # combine audio and text embeddings
            for i, (audio, start, length) in enumerate(
                zip(audio_embeds, audio_token_start_idx, audio_token_len)
            ):
                length = min(length, audio.shape[0])
                inputs_embeds[i, start : start + length] = audio[:length]

        lm_output = self.language_model.forward(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

        return lm_output

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        audio_values: Optional[torch.FloatTensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        model_input = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if past_key_values is None and audio_values is not None:
            # We only want to use audio features in the 1st generation step
            model_input["audio_values"] = audio_values
            model_input["audio_token_start_idx"] = audio_token_start_idx
            model_input["audio_token_len"] = audio_token_len

        return model_input

    @classmethod
    def _create_audio_tower(
        cls, config: UltravoxConfig
    ) -> Union[transformers.Wav2Vec2Model, ModifiedWhisperEncoder]:
        if config.audio_model_id is not None:
            if "whisper" in config.audio_model_id is not None:
                audio_tower = ModifiedWhisperEncoder.from_pretrained(
                    config.audio_model_id
                )
            else:
                audio_tower = transformers.AutoModel.from_pretrained(
                    config.audio_model_id
                )
        else:
            if "whisper" in config.audio_config._name_or_path:
                audio_tower = ModifiedWhisperEncoder(config.audio_config)
            else:
                audio_tower = transformers.AutoModel.from_config(config.audio_config)

        if isinstance(
            audio_tower,
            (transformers.Wav2Vec2BertModel, transformers.WhisperModel),
        ):
            # For these models we only need the encoder part
            # Wav2Vec2BertModel -> Wav2Vec2BertEncoder
            # WhisperModel -> WhisperEncoder
            audio_tower = audio_tower.encoder

        audio_tower = apply_lora(audio_tower, config.audio_model_lora_config)
        return audio_tower

    @classmethod
    def _create_language_model(
        cls, config: UltravoxConfig
    ) -> transformers.LlamaForCausalLM:
        if config.text_model_id is not None:
            language_model = transformers.AutoModelForCausalLM.from_pretrained(
                config.text_model_id, attn_implementation=config._attn_implementation
            )
        else:
            language_model = transformers.AutoModelForCausalLM.from_config(
                config.text_config, attn_implementation=config._attn_implementation
            )

        language_model = apply_lora(language_model, config.text_model_lora_config)
        return language_model

    def merge_and_unload(self):
        if isinstance(self.language_model, peft.PeftModel):
            self.language_model = self.language_model.merge_and_unload()
            # no need to download base language model weights anymore, so we can remove the id
            self.config.text_model_id = None
            self.keep_params.update(
                set(
                    [
                        f"language_model.{name}"
                        for name, _ in self.language_model.named_parameters()
                    ]
                )
            )

        if isinstance(self.audio_tower, peft.PeftModel):
            self.audio_tower = self.audio_tower.merge_and_unload()
            # no need to download base audio model weights anymore, so we can remove the id
            self.config.audio_model_id = None
            self.keep_params.update(
                set(
                    [
                        f"audio_tower.{name}"
                        for name, _ in self.audio_tower.named_parameters()
                    ]
                )
            )

        for param in ["text_model_lora_config", "audio_model_lora_config"]:
            if hasattr(self.config, param):
                delattr(self.config, param)

    def push_to_hub(self, *args, **kwargs):
        self.merge_and_unload()
        self.to(self.language_model.dtype)
        return super().push_to_hub(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        named_params = dict(self.named_parameters())
        state_dict = super().state_dict(*args, **kwargs)

        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in self.keep_params
            or (k in named_params and named_params[k].requires_grad)
        }
        return state_dict

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        *args,
        **kwargs,
    ):
        self.keep_params.update(set(state_dict.keys()))
        return super().load_state_dict(state_dict, *args, **kwargs)

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model (reuses Peft model's method)
        """
        count_params = peft.peft_model.PeftModel.get_nb_trainable_parameters

        trainable_params, all_param = count_params(self)

        logging.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d}"
            f" || trainable%: {100 * trainable_params / all_param:.1f}%"
        )

        lm_trainable_params, lm_all_params = count_params(self.language_model)
        audio_trainable_params, audio_all_params = count_params(self.audio_tower)

        projector_trainable_params = (
            trainable_params - lm_trainable_params - audio_trainable_params
        )
        projector_all_params = all_param - lm_all_params - audio_all_params

        logging.info(
            f"Trainable%:   "
            f" LLM: {100 * lm_trainable_params / lm_all_params:.1f}%"
            f" || Audio Encoder: {100 * audio_trainable_params / audio_all_params:.1f}%"
            f" || Projector: {100 * projector_trainable_params / projector_all_params:.1f}%"
        )


def apply_lora(model: torch.nn.Module, lora_config: dict) -> torch.nn.Module:
    """
    Applies LoRA finetuning to the model. If the `r` parameter is set to 0, the model is frozen instead.
    """
    lora_config = peft.LoraConfig(**lora_config or {})

    if lora_config.r == 0:
        # freeze the model entirely
        for param in model.parameters():
            param.requires_grad = False
    else:
        model = peft.get_peft_model(model, lora_config)

    return model


class StackAudioFrames(nn.Module):
    """
    Stack the audio embedding frames to reduce the sequence length by a factor of `stack_factor`.

    The number of output frames will be `ceil(T / stack_factor) + 1` where `T` is the number of input frames.
    NOTE: the extra +1 is intentional: in case the number of audio tokens are over-estimated by the processor,
    we want to make sure `processor.audio_token_replacement` (i.e. EOS) doesn't get leaked into the middle of embeddings.
    In most cases this extra padding will get removed in the model's forward function so it has no effect.
    """

    def __init__(self, stack_factor: int = 8):
        super().__init__()
        self.stack_factor = stack_factor

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        B, T, C = audio_embeds.shape
        T_pad = (T + self.stack_factor - 1) // self.stack_factor * self.stack_factor
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, T_pad - T + self.stack_factor))
        B, T, C = audio_embeds.shape
        audio_embeds = audio_embeds.view(
            B, T // self.stack_factor, C * self.stack_factor
        )
        return audio_embeds


class RMSNorm(transformers.models.llama.modeling_llama.LlamaRMSNorm):
    def __init__(self, hidden_size: int, init: float = 1, eps: float = 1e-6):
        super().__init__(hidden_size=hidden_size, eps=eps)
        self.weight.data.fill_(init)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class UltravoxProjector(nn.Sequential):
    def __init__(self, config: UltravoxConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self._pad_and_stack = StackAudioFrames(config.stack_factor)
        dim = config.audio_config.hidden_size * config.stack_factor
        self.ln_pre = RMSNorm(dim, init=config.norm_init)
        self.linear_1 = nn.Linear(dim, self.hidden_dim, bias=False)
        dim = self.hidden_dim
        self.act = transformers.activations.get_activation(config.projector_act)
        dim = dim // 2 if config.projector_act == "swiglu" else dim
        self.linear_2 = nn.Linear(dim, config.text_config.hidden_size, bias=False)
        self.ln_post = RMSNorm(config.text_config.hidden_size, init=config.norm_init)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        audio_features = self._pad_and_stack(audio_features)
        audio_features = self.ln_pre(audio_features)
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln_post(hidden_states)
        return hidden_states


UltravoxConfig.register_for_auto_class()
UltravoxModel.register_for_auto_class()

transformers.AutoConfig.register("ultravox", UltravoxConfig)
transformers.AutoModel.register(UltravoxConfig, UltravoxModel)
# transformers.AutoProcessor.register(UltravoxConfig, UltravoxProcessor)  # TODO: make processor work standalone

transformers.activations.ACT2FN["swiglu"] = SwiGLU
