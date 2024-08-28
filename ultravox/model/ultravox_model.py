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
from transformers.models.whisper import modeling_whisper as whisper

# We must use relative import in this directory to allow uploading to HF Hub
# Even "from . import X" pattern doesn't work (undocumented and unclear why)
from .ultravox_config import LossConfig, LossFunction, UltravoxConfig, AdapterType
from .ultravox_adapter import UltravoxAdapter, StackingAdapter, CFormerAdapter

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
    _no_split_modules = ["Wav2Vec2Model", "WhisperEncoder", "LlamaDecoderLayer"]
    # We minimize the weights in state_dict in order to reduce the size of the checkpoint
    # The issue is that load_pretrained() uses state_dict() keys to know what keys are expected
    # As such we have to tell is to ignore some keys that are not always in the model
    _keys_to_ignore_on_load_unexpected = ["audio_tower.*", "language_model.*"]
    # Usually we load encoder weights from a pretrained model, so we don't want to load the decoder weights
    # Technically we never hit this issue because these keys are already removed from state_dict() however,
    # but there's no harm in keeping it here for when we change that behavior.
    _keys_to_ignore_on_load_missing = ["audio_tower.*"]

    def __init__(self, config: UltravoxConfig):
        super().__init__(config)

        self.keep_params: Set[str] = set()
        self.vocab_size = config.vocab_size

        self.audio_tower = self._create_audio_tower(config)
        self.adapter = self._create_adapter(config)
        self.language_model = self._create_language_model(config)

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
        if LossFunction.Input_KL in loss_config.loss_weights and self.config.adapter_type != AdapterType.CFORMER:
            raise ValueError(
                f"Input KL loss is only supported for CFormer adapter, not {self.config.adapter_type}."
            )
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
    
    def _insert_speech_placeholders(self, inputs_embeds, attention_mask, audio_start_idx, num_speech_tokens, position_ids, cache_position):
        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        
        if position_ids is None or position_ids.shape != (batch_size, seq_len):
            raise ValueError(f"position_ids must be provided and have shape (batch_size, seq_len). Got: {position_ids.shape if position_ids is not None else None}")

        # Calculate the actual length of each sample and the new length after insertion
        sample_lengths = attention_mask.sum(dim=1)
        new_sample_lengths = sample_lengths + num_speech_tokens
        new_seq_len = new_sample_lengths.max().item()

        # Create new tensors with expanded size
        new_inputs_embeds = torch.zeros((batch_size, new_seq_len, hidden_dim), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        new_attention_mask = torch.zeros((batch_size, new_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
        new_position_ids = torch.zeros((batch_size, new_seq_len), device=position_ids.device, dtype=position_ids.dtype)

        for i in range(batch_size):
            insert_pos = audio_start_idx[i]
            insert_len = num_speech_tokens[i].item()
            original_len = sample_lengths[i].item()
            new_len = new_sample_lengths[i].item()

            # Update inputs_embeds
            new_inputs_embeds[i, :insert_pos] = inputs_embeds[i, :insert_pos]
            new_inputs_embeds[i, insert_pos + insert_len:new_len] = inputs_embeds[i, insert_pos:original_len]

            # Update attention_mask
            new_attention_mask[i, :insert_pos] = attention_mask[i, :insert_pos]
            new_attention_mask[i, insert_pos:insert_pos + insert_len] = 1  # Set attention mask for audio tokens
            new_attention_mask[i, insert_pos + insert_len:new_len] = attention_mask[i, insert_pos:original_len]

            # Update position_ids
            new_position_ids[i, :insert_pos] = position_ids[i, :insert_pos]
            new_position_ids[i, insert_pos:insert_pos+insert_len] = position_ids[i, insert_pos] + torch.arange(insert_len, device=position_ids.device, dtype=position_ids.dtype)
            new_position_ids[i, insert_pos + insert_len:new_len] = position_ids[i, insert_pos:original_len] + insert_len

        # Update cache_position if provided
        new_cache_position = cache_position
        if cache_position is not None:
            if not isinstance(cache_position, torch.Tensor) or cache_position.dim() != 1:
                raise ValueError(f"cache_position must be a 1-dimensional torch.Tensor. Got: {type(cache_position)}")
            
            # Validation check for the last position in cache_position
            if cache_position[-1] != seq_len - 1:
                raise ValueError(f"The last position in cache_position should be seq_len-1 ({seq_len-1}), but got {cache_position[-1]}")
            
            max_inserted = new_seq_len - seq_len
            new_cache_position = torch.cat([
                cache_position,
                torch.arange(
                    cache_position[-1] + 1,
                    cache_position[-1] + 1 + max_inserted,
                    device=cache_position.device,
                    dtype=cache_position.dtype
                )
            ])

        return new_inputs_embeds, new_attention_mask, new_position_ids, new_cache_position

    def _compute_kl_loss(
        self,
        lm_output: transformers.modeling_outputs.CausalLMOutputWithPast,
        labels: Optional[torch.Tensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
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
        losses: Dict[LossFunction, torch.FloatTensor] = {}
        # compute the KL divergence loss between the two models
        if LossFunction.Response_KL in self.loss_config.loss_weights:
            loss = F.kl_div(
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
            losses[LossFunction.Response_KL] = loss
        if LossFunction.Input_KL in self.loss_config.loss_weights and \
            audio_token_start_idx is not None and \
            audio_token_len is not None:

            # compute the KL divergence loss for audio tokens
            audio_mask = ((audio_token_start_idx.unsqueeze(1) <= torch.arange(labels.size(1), device=labels.device)) & 
                        (audio_token_start_idx.unsqueeze(1) + audio_token_len.unsqueeze(1) > torch.arange(labels.size(1), device=labels.device)))
            
            loss = F.kl_div(
                F.log_softmax(
                    lm_output.logits[audio_mask] / self.loss_config.kl_temperature,
                    dim=-1,
                ),
                F.softmax(
                    alt_lm_output.logits[audio_mask] / self.loss_config.kl_temperature,
                    dim=-1,
                ),
                reduction="batchmean",
            )
            losses[LossFunction.Input_KL] = loss
        return losses

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
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

            audio_embeds, pred_num_tokens = self.adapter.forward(audio_tower_output, audio_token_len)
            if pred_num_tokens is not None:
                inputs_embeds, attention_mask, position_ids, cache_position = self._insert_speech_placeholders(
                    inputs_embeds,
                    attention_mask,
                    audio_token_start_idx,
                    pred_num_tokens,
                    kwargs.get("position_ids"),
                    kwargs.get("cache_position", None)
                )
                audio_token_len = pred_num_tokens
                kwargs["position_ids"] = position_ids
                kwargs["cache_position"] = cache_position

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
        if self.training:
            total_loss = 0
            if self.loss_config.contains_kl_loss:
                kl_loss = self._compute_kl_loss(
                    lm_output=lm_output,
                    labels=labels,
                    audio_token_start_idx=audio_token_start_idx,
                    audio_token_len=audio_token_len,
                    past_key_values=past_key_values,
                    alt_input_ids=alt_input_ids,
                    alt_attention_mask=alt_attention_mask,
                    alt_labels=alt_labels,
                    **kwargs
                )
            for loss, weight in self.loss_config.loss_weights.items():
                if loss == LossFunction.Response_CE:
                    total_loss += weight * lm_output.loss
                elif loss == LossFunction.Response_KL:
                    total_loss += weight * kl_loss[LossFunction.Response_KL]
                elif loss == LossFunction.Input_KL:
                    total_loss += weight * kl_loss[LossFunction.Input_KL]
                elif loss == LossFunction.CIF_L1:
                    total_loss += weight * F.l1_loss(pred_num_tokens/audio_token_len, torch.ones_like(audio_token_len), reduction="mean")
                else:
                    raise ValueError(f"Unsupported loss function: {loss}")
            
            lm_output.loss = total_loss
        return lm_output

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        audio_values: Optional[torch.FloatTensor] = None,
        audio_token_start_idx: Optional[torch.Tensor] = None,
        audio_token_len: Optional[torch.Tensor] = None,
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
        prefill_start_idx = 0 if cache_position is None else cache_position[0]
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

        return model_input

    @classmethod
    def _create_adapter(
        cls, config: UltravoxConfig
    ) -> UltravoxAdapter:
        if config.adapter_type is AdapterType.STACKING:
            adapter = StackingAdapter(config)
        elif config.adapter_type is AdapterType.CFORMER:
            adapter = CFormerAdapter(config)
        else:
            raise ValueError(f"Unsupported adapter type: {config.adapter_type}")
        return adapter


    @classmethod
    def _create_audio_tower(
        cls, config: UltravoxConfig
    ) -> Union[transformers.Wav2Vec2Model, "ModifiedWhisperEncoder"]:
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
                config.text_model_id, attn_implementation=config._attn_implementation
            )
        else:
            with transformers.modeling_utils.no_init_weights():
                # we only ever use from_config if the weights are retrained, hence initializing is not
                # required. This makes the model quite creation faster since init on CPU is quite slow.
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

        adapter_trainable_params = (
            trainable_params - lm_trainable_params - audio_trainable_params
        )
        projector_all_params = all_param - lm_all_params - audio_all_params

        logging.info(
            f"Trainable%:   "
            f" LLM: {100 * lm_trainable_params / lm_all_params:.1f}%"
            f" || Audio Encoder: {100 * audio_trainable_params / audio_all_params:.1f}%"
            f" || Adapter: {100 * adapter_trainable_params / projector_all_params:.1f}%"
        )


def is_cache_empty(
    past_key_values: Optional[Union[Tuple, transformers.cache_utils.Cache]]
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
    lora_config = peft.LoraConfig(**lora_config or {})

    if lora_config.r == 0:
        # freeze the model entirely
        for param in model.parameters():
            param.requires_grad = False
    else:
        model = peft.get_peft_model(model, lora_config)

    return model


class ModifiedWhisperEncoder(whisper.WhisperEncoder):
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

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        expected_seq_length = (
            self.config.max_source_positions
            * self.conv1.stride[0]
            * self.conv2.stride[0]
        )
        if input_features.shape[-1] > expected_seq_length:
            import warnings

            warnings.warn(
                f"Input features length ({input_features.shape[-1]}) exceeds the expected sequence length ({expected_seq_length}). The input will be chunked to {expected_seq_length}."
            )
            input_features = input_features[..., :expected_seq_length]

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
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
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

UltravoxModel.register_for_auto_class()
transformers.AutoModel.register(UltravoxConfig, UltravoxModel)

