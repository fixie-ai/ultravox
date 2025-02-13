import logging
import re
from typing import Any, Dict, Generator, Optional, Set, Tuple, Union

import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import transformers.activations
import transformers.modeling_outputs
import transformers.models
from transformers.models.whisper import modeling_whisper as whisper

# We must use relative import in this directory to allow uploading to HF Hub
# Even "from . import X" pattern doesn't work (undocumented and unclear why)
from .ultravox_config import LossConfig
from .ultravox_config import LossFunction
from .ultravox_config import UltravoxConfig


class UltravoxModel(transformers.LlamaPreTrainedModel):
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
    # Usually we load encoder and LLM weights from a pretrained model separately, so they are allowed to be missing
    _keys_to_ignore_on_load_missing = ["audio_tower.*", "language_model.*"]
    # Since we have kwargs in forward, we need to set this to False, otherwise grad_accum_steps will cause incorrect train loss to be reported
    # see https://github.com/huggingface/transformers/issues/35856 and https://github.com/huggingface/trl/pull/2615/files
    accepts_loss_kwargs = False

    def __init__(self, config: UltravoxConfig):
        super().__init__(config)
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

        self.keep_params: Set[str] = set()
        self.vocab_size = config.vocab_size

        self.audio_tower = self._create_audio_tower(config)
        self.audio_tower_context_length: Optional[int] = None
        self.audio_tower_context_length = self.audio_tower.max_context_length

        self.multi_modal_projector = self._create_multi_modal_projector(config)
        self.language_model = self._create_language_model(config)

        # Determine no_split_modules dynamically to use with FSDP auto_wrap policy.
        # FSDP throws an error if some of the layer types are not found in the model.
        # This would be something like ["LlamaDecoderLayer"] as we don't split audio encoder layers.
        self._no_split_modules = self.language_model._no_split_modules

        self.loss_config = LossConfig()
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

    def set_loss_config(self, loss_config: LossConfig):
        self.loss_config = loss_config

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

    def _compute_kl_loss(
        self,
        lm_output: transformers.modeling_outputs.CausalLMOutputWithPast,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
        alt_input_ids: Optional[torch.Tensor] = None,
        alt_attention_mask: Optional[torch.Tensor] = None,
        alt_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # disable gradient computation for the teacher model
        with torch.no_grad():
            # compute the teacher (text-only) model's distribution
            alt_inputs_embeds = self.get_input_embeddings().forward(alt_input_ids)
            alt_lm_output = self.language_model.forward(
                inputs_embeds=alt_inputs_embeds,
                labels=alt_labels,
                attention_mask=alt_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        # compute the KL divergence loss between the two models
        kl_loss = F.kl_div(
            F.log_softmax(
                lm_output.logits[labels != -100] / self.loss_config.kl_temperature,
                dim=-1,
            ),
            F.softmax(
                alt_lm_output.logits[alt_labels != -100]
                / self.loss_config.kl_temperature,
                dim=-1,
            ),
            reduction="batchmean",
        )
        return {"loss": kl_loss}

    def _audio_iter(
        self, audio_batch_size: torch.Tensor
    ) -> Generator[Tuple[int, int], None, None]:
        """
        Iterate over the audio batch size and yield the batch index and audio index of each audio item.

        Args:
            audio_batch_size: A tensor of shape (B,) where B is the batch size.

        Returns:
            A generator that yields a tuple of (start index, length) for each audio item.
        """
        audio_index = 0
        for i_b, batch_count in enumerate(audio_batch_size):
            for _ in range(batch_count):
                yield i_b, audio_index
                audio_index += 1

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_lens: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
        audio_batch_size: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
        # the alt_* fields are needed for KL divergence loss
        alt_input_ids: Optional[torch.Tensor] = None,
        alt_attention_mask: Optional[torch.Tensor] = None,
        alt_labels: Optional[torch.Tensor] = None,
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

        if audio_values is not None and len(audio_values) > 0:
            assert (
                audio_token_start_idx is not None
                and audio_token_len is not None
                and audio_lens is not None
                and audio_batch_size is not None
            ), "audio_token_start_idx/audio_token_len/audio_lens must be provided if audio_values are provided."
            assert (
                len(audio_token_start_idx)
                == len(audio_token_len)
                == len(audio_lens)
                == len(audio_values)
            ), "audio_token_start_idx/audio_token_len/audio_lens/audio_values must have the same batch size."
            assert len(audio_batch_size) == len(
                inputs_embeds
            ), "audio_batch_size and inputs_embeds must have the same batch size."

            # B x A/3200 x (D=max-audio-length-in-batch)
            audio_tower_output = self.audio_tower.forward(
                audio_values.to(self.audio_tower.dtype),
                audio_len=audio_lens,
            ).last_hidden_state
            audio_tower_output = audio_tower_output.to(inputs_embeds.dtype)
            audio_embeds = self.multi_modal_projector.forward(audio_tower_output)

            # combine audio and text embeddings
            for i_b, i_a in self._audio_iter(audio_batch_size):
                start_idx = audio_token_start_idx[i_a]
                token_len = audio_token_len[i_a]
                item_embedding = audio_embeds[i_a][:token_len]
                inputs_embeds[i_b][start_idx : start_idx + token_len] = item_embedding

        lm_output = self.language_model.forward(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        if self.training:
            if self.loss_config.loss_function == LossFunction.CrossEntropy:
                return lm_output
            elif self.loss_config.loss_function == LossFunction.KL_Divergence:
                return self._compute_kl_loss(
                    lm_output=lm_output,
                    labels=labels,
                    past_key_values=past_key_values,
                    alt_input_ids=alt_input_ids,
                    alt_attention_mask=alt_attention_mask,
                    alt_labels=alt_labels,
                    **kwargs,
                )
            else:
                raise ValueError(
                    f"Unsupported loss function: {self.loss_config.loss_function}"
                )
        else:
            return lm_output

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        audio_values: Optional[torch.FloatTensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
        audio_lens: Optional[torch.Tensor] = None,
        audio_batch_size: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        model_input = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        # include audio information in model_input only when it is needed during prefilling
        # audio_token_start_idx should always be relative to the current cache position
        prefill_start_idx: torch.Tensor = (
            torch.tensor(0) if cache_position is None else cache_position[0]
        )
        if (
            audio_values is not None
            and audio_token_start_idx is not None
            and prefill_start_idx <= torch.max(audio_token_start_idx)
        ):
            model_input["audio_values"] = audio_values
            model_input["audio_token_start_idx"] = (
                audio_token_start_idx - prefill_start_idx
            )
            model_input["audio_token_len"] = audio_token_len
            model_input["audio_batch_size"] = audio_batch_size
            model_input["audio_lens"] = audio_lens

        return model_input

    @classmethod
    def _create_multi_modal_projector(
        cls, config: UltravoxConfig
    ) -> "UltravoxProjector":
        projector = UltravoxProjector(config)
        projector.to(config.torch_dtype)
        return projector

    @classmethod
    def _create_audio_tower(
        cls, config: UltravoxConfig
    ) -> Union[transformers.Wav2Vec2Model, "ModifiedWhisperEncoder"]:
        if config.audio_model_id is not None:
            if "whisper" in config.audio_model_id.lower():
                audio_tower = ModifiedWhisperEncoder.from_pretrained(
                    config.audio_model_id, torch_dtype=config.torch_dtype
                )
                audio_tower.init_latency_mask(
                    config.audio_latency_block_size, dtype=config.torch_dtype
                )
            else:
                assert config.audio_latency_block_size in (
                    None,
                    0,
                ), "only whisper audio tower supports audio latency masking, got non-zero value for 'audio_latency_block_size'"
                audio_tower = transformers.AutoModel.from_pretrained(
                    config.audio_model_id, torch_dtype=config.torch_dtype
                )
        else:
            if "whisper" in config.audio_config._name_or_path.lower():
                audio_tower = ModifiedWhisperEncoder(config.audio_config)
                audio_tower.init_latency_mask(
                    config.audio_latency_block_size, dtype=config.torch_dtype
                )
            else:
                assert config.audio_latency_block_size in (
                    None,
                    0,
                ), "only whisper audio tower supports audio latency masking, got non-zero value for 'audio_latency_block_size'"
                with transformers.modeling_utils.no_init_weights():
                    # we only ever use from_config if the weights are retrained, hence initializing is not
                    # required. This makes the model quite creation faster since init on CPU is quite slow.
                    audio_tower = transformers.AutoModel.from_config(
                        config.audio_config
                    )

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
                config.text_model_id,
                attn_implementation=config._attn_implementation,
                torch_dtype=config.torch_dtype,
            )
        else:
            with transformers.modeling_utils.no_init_weights():
                # we only ever use from_config if the weights are retrained, hence initializing is not
                # required. This makes the model quite creation faster since init on CPU is quite slow.
                language_model = transformers.AutoModelForCausalLM.from_config(
                    config.text_config,
                    attn_implementation=config._attn_implementation,
                    torch_dtype=config.torch_dtype,
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
        return super().push_to_hub(*args, **kwargs)

    def diff_state_dict(
        self, state_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if state_dict is None:
            state_dict = super().state_dict()

        trainable_params = {k for k, v in self.named_parameters() if v.requires_grad}
        # normalize the keys to match the original model
        # Example: audio_tower.base_model.model.layers.0._fsdp_wrapped_module.self_attn.k_proj.lora_B.default.weight
        trainable_params = {
            k.replace("_fsdp_wrapped_module.", "") for k in trainable_params
        }

        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in self.keep_params or k in trainable_params
        }

        return state_dict

    def save_pretrained(
        self, *args, state_dict: Optional[Dict[str, Any]] = None, **kwargs
    ):
        state_dict = self.diff_state_dict(state_dict)

        super().save_pretrained(*args, state_dict=state_dict, **kwargs)

    def _pre_load_state_dict_hook(self, state_dict: Dict[str, Any], *args, **kwargs):
        self.keep_params.update(set(state_dict.keys()))

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


# TODO: refactor common parts to a shared module
def is_cache_empty(
    past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]],
) -> bool:
    """
    Check if the cache is empty.
    """
    if past_key_values is None:
        return True
    if isinstance(past_key_values, tuple):
        return all(len(c) == 0 for c in past_key_values)
    return past_key_values.get_seq_length() == 0


def apply_lora(model: torch.nn.Module, lora_config: dict) -> torch.nn.Module:
    """
    Applies LoRA finetuning to the model. If the `r` parameter is set to 0, the model is frozen instead.
    """
    unfreeze_layers = lora_config.pop("unfreeze_layers", None)
    lora_config = peft.LoraConfig(**lora_config or {})

    if lora_config.r == 0:
        # freeze the model entirely, except for the specified layers
        for name, param in model.named_parameters():
            if not unfreeze_layers or not any(
                re.match(layer, name) for layer in unfreeze_layers
            ):
                param.requires_grad = False
            else:
                logging.info(f"Unfreezing layer: {name} with #{param.numel()} params")
    else:
        model = peft.get_peft_model(model, lora_config)

    return model


class StackAudioFrames(nn.Module):
    """
    Stack the audio embedding frames to reduce the sequence length by a factor
    of `stack_factor`.
    """

    def __init__(self, stack_factor: int = 8):
        super().__init__()
        self.stack_factor = stack_factor

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        B, T, C = audio_embeds.shape
        T_pad = (T + self.stack_factor - 1) // self.stack_factor * self.stack_factor
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, T_pad - T))
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


class UltravoxProjector(nn.Module):
    def __init__(self, config: UltravoxConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self._pad_and_stack = StackAudioFrames(config.stack_factor)
        dim_in = config.audio_config.hidden_size * config.stack_factor
        self.ln_pre = RMSNorm(dim_in, init=config.norm_init)
        self.linear_1 = nn.Linear(dim_in, self.hidden_dim, bias=False)
        dim_mid = self.hidden_dim
        self.act = transformers.activations.get_activation(config.projector_act)
        dim_mid = dim_mid // 2 if config.projector_act == "swiglu" else dim_mid
        dim_out = config.text_config.hidden_size
        self.linear_2 = nn.Linear(dim_mid, dim_out, bias=False)

        # Ultravox v0.4.1 and below uses layer_norm after the second linear layer,
        # while v0.5.0 and above uses layer_norm after the first linear layer.
        if config.projector_ln_mid:
            self.ln_mid: nn.Module = RMSNorm(dim_mid, init=config.norm_init)
            self.ln_post: nn.Module = nn.Identity()
        else:
            self.ln_mid = nn.Identity()
            self.ln_post = RMSNorm(dim_out, init=config.norm_init)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        audio_features = self._pad_and_stack(audio_features)
        audio_features = self.ln_pre(audio_features)
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.ln_mid(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln_post(hidden_states)
        return hidden_states


class ModifiedWhisperEncoder(
    whisper.WhisperEncoder, transformers.modeling_utils.ModuleUtilsMixin
):
    """
    Encoder portion of OpenAI's Whisper model.

    This implementation is a slightly modified version of HF Transformers' Whisper Encoder, with only a few fixes:
    1. base_model_prefix updated to allow for doing `.from_pretrained` directly on the encoder
    2. allow less than 30 second of audio padding to be passed in:
        - relaxed ValueError check for `input_features` length to be less than or equal to `expected_seq_length` instead of strictly equal
        - embed_pos is now sliced to match the length of `inputs_embeds`

    Original: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
    """

    base_model_prefix = "model.encoder"
    _no_split_modules = ["WhisperEncoderLayer"]

    def __init__(self, config: transformers.WhisperConfig):
        super().__init__(config)
        self.config.is_decoder = False

    @property
    def max_context_length(self):
        return (
            self.config.max_source_positions
            * self.conv1.stride[0]
            * self.conv2.stride[0]
        )

    def init_latency_mask(self, audio_latency_block_size: int, dtype: torch.dtype):
        if audio_latency_block_size is None:
            self.audio_streaming_mask = None
            return

        # Use max_context_length directly in the calculation
        max_seqlen = self.max_context_length
        assert (
            max_seqlen > 0
        ), f"maximum sequence length must be positive, got {max_seqlen}"
        assert (
            max_seqlen % audio_latency_block_size == 0
        ), f"audio_latency_block_size {audio_latency_block_size} must divide {max_seqlen} evenly."
        # Given the block size, we calculate number of blocks.
        audio_latency_nblocks = max_seqlen // audio_latency_block_size
        audio_streaming_mask = (
            torch.tril(
                torch.ones(audio_latency_nblocks, audio_latency_nblocks),
                diagonal=0,
            )
            .repeat_interleave(audio_latency_block_size, dim=0)
            .repeat_interleave(audio_latency_block_size, dim=1)
        )
        audio_streaming_mask = (1.0 - audio_streaming_mask) * torch.finfo(dtype).min
        audio_streaming_mask = audio_streaming_mask[None, None, :, :]
        self.register_buffer(
            "audio_streaming_mask", audio_streaming_mask, persistent=False
        )

    def forward(
        self,
        input_features,
        audio_len=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        expected_seq_length = self.max_context_length
        if input_features.shape[-1] > expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length} or less, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight[: inputs_embeds.size(-2)]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Create attention mask based on audio lengths to mask out padding tokens
        # For each sample in batch:
        # - Convert raw audio length to feature length after convolutions
        # - Create boolean mask that is True for valid positions and False for padding
        # - Convert to extended attention mask format expected by transformer layers
        #   (1.0 for positions to attend to, large negative for positions to ignore)
        # This masking ensures consistent behavior between training and inference
        # by preventing the model from attending to padding tokens in both cases
        attention_mask = None
        if audio_len != None:
            audio_feature_len = self._get_feat_extract_output_lengths(audio_len)
            max_seq_len = hidden_states.shape[1]
            attention_mask = torch.arange(max_seq_len, device=hidden_states.device)[
                None, :
            ].lt(audio_feature_len.view(-1, 1))
            attention_mask = self.get_extended_attention_mask(
                attention_mask,
                None,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        if self.audio_streaming_mask is not None:
            seqlen = hidden_states.size(-2)
            if attention_mask is not None:
                attention_mask = torch.minimum(
                    self.audio_streaming_mask[:, :, :seqlen, :seqlen], attention_mask
                )  # merge
            else:
                attention_mask = self.audio_streaming_mask[:, :, :seqlen, :seqlen]
            attention_mask = attention_mask.to(hidden_states.dtype)

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


UltravoxConfig.register_for_auto_class()
UltravoxModel.register_for_auto_class()

transformers.AutoConfig.register("ultravox", UltravoxConfig)
transformers.AutoModel.register(UltravoxConfig, UltravoxModel)

transformers.activations.ACT2FN["swiglu"] = SwiGLU
