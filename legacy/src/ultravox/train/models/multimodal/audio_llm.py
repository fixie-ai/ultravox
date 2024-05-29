import os
import typing as t

import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import transformers.models
from train.models import audio as audio_models
from train.models import text as text_models

from .config import FreezingConfig
from .config import SpeechLMConfig


class SpeechLM(transformers.LlamaPreTrainedModel, transformers.GenerationMixin):
    config_class = SpeechLMConfig
    base_model_prefix = "llm.model"
    _tied_weights_keys = ["llm.lm_head.weight", "llm.embed_tokens.weight"]
    # TODO: other generation kwargs?

    def __init__(self, config: SpeechLMConfig):
        self.keep_params = set()
        self.config = config  # Do not move this line after the init. It's point is just for the type hints
        super().__init__(config)

        self.llm: transformers.LlamaForCausalLM = (
            transformers.LlamaForCausalLM.from_pretrained(
                config.llm_name,
                torch_dtype=config.torch_dtype,
                token=os.environ.get("HF_ACCESS_TOKEN", None),
                low_cpu_mem_usage=False,
                # device_map=config.device_map,
            )
        )
        self.generation_config = self.llm.generation_config

        self.audio_enc = audio_models.AudioEncLoader(
            config.audio_enc_name, dtype=config.torch_dtype
        ).get_model()

        # Taking out embed_tokens out of LLM so we can apply it to text only
        self.embed_tokens: nn.Embedding = self.llm.get_input_embeddings()
        self.llm.set_input_embeddings(nn.Identity())

        # self.token_embed_dim: int = self.llm.config.hidden_size
        self.token_embed_dim: int = self.embed_tokens.embedding_dim

        if config.audio_squeeze_type in ["stride", "mean", "random"]:
            audio_embed_in_dim = self.audio_enc.config.hidden_size
        elif config.audio_squeeze_type == "stack":
            audio_embed_in_dim = self.audio_enc.config.hidden_size * config.audio_stride
        else:
            raise ValueError(f"Unknown audio_squeeze_type: {config.audio_squeeze_type}")

        self.audio_to_embed = nn.Sequential(
            nn.LayerNorm(audio_embed_in_dim),
            nn.Linear(
                audio_embed_in_dim, self.token_embed_dim, dtype=config.torch_dtype
            ),
            nn.ReLU(),  # SiLU?
            nn.LayerNorm(self.token_embed_dim),
            nn.Linear(
                self.token_embed_dim, self.token_embed_dim, dtype=config.torch_dtype
            ),
            # nn.LayerNorm(self.token_embed_dim, elementwise_affine=False),
            # nn.MultiheadAttention(
            #     embed_dim=self.token_embed_dim, num_heads=8, dtype=config.torch_dtype
            # ),
        )

        if config.init_type == "small":
            for layer in self.audio_to_embed:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.div_(10)
                    layer.bias.data.div_(10)

        # Fuck me ...
        transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[
            "audio-llm"
        ] = self.__class__.__name__
        if isinstance(self.audio_enc.config, transformers.HubertConfig):
            self.audio_enc.config.mask_time_length = 6

    def state_dict(self, *args, **kwargs):
        named_params = dict(self.named_parameters())
        state_dict: dict = super().state_dict(*args, **kwargs)

        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in self.keep_params
            or (k in named_params and named_params[k].requires_grad)
        }
        return state_dict

    def load_state_dict(
        self,
        state_dict: t.Dict[str, t.Any],
        *args,
        **kwargs,
    ):
        self.keep_params.update(set(state_dict.keys()))
        return super().load_state_dict(state_dict, *args, **kwargs)

    def apply_lora_configs(self, config: FreezingConfig):
        # Freeze parameters or apply LoRA as needed
        if config.llm_lora_config:
            self.llm = text_models.apply_lora(
                self.llm, lora_config=config.llm_lora_config
            )
        if config.audio_enc_lora_config:
            self.audio_enc = text_models.apply_lora(
                self.audio_enc, lora_config=config.audio_enc_lora_config
            )
        if config.freeze_text_embeds:
            text_models.freeze_parameters(self.embed_tokens)
        if config.freeze_audio_embeds:
            text_models.freeze_parameters(self.audio_to_embed)

    def forward_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        # B: batch size, A: audio length, T: text length
        # 768 is the hidden dim of Wave2Vec2 (audio_dim)
        # BxA  ->  B x A/320 x 768
        audio_features = audio_features.to(dtype=self.audio_enc.dtype)

        audio_hidden: torch.Tensor = self.audio_enc.forward(
            audio_features
        ).last_hidden_state

        audio_hidden = audio_hidden.to(self.dtype)

        stride: int = self.config.audio_stride

        # B x A/3200 x 768
        audio_hidden = self._downsample_audio_features(audio_hidden, stride)

        # B x A/3200 x 768  ->  B x A/3200 x 2048
        audio_embed = self.audio_to_embed(audio_hidden.contiguous())

        # Lingo:
        # * audio_features are the raw audio features
        # * audio_hidden are the output of the audio encoder
        # * audio_embed are the embeddings and by-pass the LLM embedding layer

        return audio_embed

    def _downsample_audio_features(self, audio_hidden: torch.Tensor, stride: int):
        """Downsamples audio frames."""
        # throw away frames to make sure audio_seq_len is divisible by audio_stride
        hidden_seq_len = audio_hidden.shape[-2]
        hidden_seq_len = hidden_seq_len - hidden_seq_len % stride
        audio_hidden = audio_hidden[..., :hidden_seq_len, :]

        # combine frames by applying: stack, drop, or average operation
        if self.config.audio_squeeze_type == "stride":
            # Throw away some frames to get to increase stride of audio
            # Given a ~20ms initial stride and a 10x here we get ~200ms audio-tokens
            # B x A/320 x 768  ->  B x A/3200 x 768
            audio_hidden = audio_hidden[..., ::stride, :]
        elif self.config.audio_squeeze_type == "mean":
            # reshape to B x A/3200 x 10 x 768 then average over the 10 frames
            audio_hidden = audio_hidden.view(
                *audio_hidden.shape[:-2],
                -1,
                stride,
                audio_hidden.shape[-1],
            ).mean(dim=-2)
        elif self.config.audio_squeeze_type == "stack":
            # reshape to B x A/3200 x 10 x 768 then stack the 10 frames
            audio_hidden = audio_hidden.view(
                *audio_hidden.shape[:-2],
                -1,
                stride * audio_hidden.shape[-1],
            )
        else:
            # TODO: random drop type (Google SLM)
            raise ValueError(
                f"Unknown audio_squeeze_type: {self.config.audio_squeeze_type}"
            )

        return audio_hidden

    # def inference(self, audio, tokenizer: transformers.LlamaTokenizer):
    #     with torch.no_grad():
    #         audio_embed = self.forward_audio(audio.unsqueeze(0))
    #         tokens = tokenizer.apply_chat_template(
    #             ['Repeat the following"'],
    #             tokenize=False,
    #             add_generation_prompt=True,
    #             return_tensors="pt",
    #         )

    # prompt: please repeat the following: "[audio]"
    # also todo: compute metrics ...

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.llm._reorder_cache(past_key_values, beam_idx)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        audio_features=None,
        audio_token_start_idx=None,
        audio_token_len=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
        # input_ids: torch.Tensor,
        # audio_features: torch.Tensor,
        # audio_token_mask: t.Optional[torch.Tensor] = None,
        # *args,
        # **kwargs
    ) -> t.Dict[str, t.Any]:
        model_input = self.llm.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if past_key_values is None:
            # We only want to use audio features in the 1st generation step
            model_input["audio_features"] = audio_features
            model_input["audio_token_start_idx"] = audio_token_start_idx
            model_input["audio_token_len"] = audio_token_len
            # TODO: but why does LTU do this if embeds is not present?
            # What are embeds?
        return model_input

    def forward(
        self,
        input_ids: torch.Tensor,  # this is the text
        audio_features: t.Optional[torch.Tensor] = None,
        # TODO: this should be optional to allow for inference
        inputs_embeds: t.Optional[torch.FloatTensor] = None,
        labels: t.Optional[torch.Tensor] = None,
        attention_mask: t.Optional[torch.Tensor] = None,
        audio_token_start_idx: t.Optional[torch.Tensor] = None,
        audio_token_len: t.Optional[torch.Tensor] = None,
        past_key_values: t.Optional[t.Tuple] = None,
        **kwargs,
    ):
        # V (vocab size for TinyLlama): 32000
        # D (embedding dim): 2048
        # B x T  ->  B x T x D
        if inputs_embeds is None:
            input_embeds = self.embed_tokens.forward(input_ids)

        if audio_features is not None:
            # TODO: there is a bug here if you use fp16. I couldn't figure it out, but it
            # has to do with unscaling the gradients in automatic mixed precision. Not a priority to fix.

            # B x A/3200 x D
            audio_embed = self.forward_audio(audio_features)

            input_embeds = self._combine_embeds(
                audio_embed, input_embeds, audio_token_start_idx, audio_token_len
            )
        else:
            input_embeds = input_embeds
            audio_embed = None

        input_embeds = input_embeds.contiguous()

        if labels is not None and labels.shape[1] != input_embeds.shape[1]:
            # This is a hack to allow full labels to be passed in even
            # if we want to generate the response.
            labels = labels[:, : input_embeds.shape[1]]
            # labels = None

        llm_output = self.llm.forward(
            inputs_embeds=input_embeds,
            labels=labels,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

        # print(f"{input_ids[0, :7]} ... {input_ids[0, -7:]}")
        # max_logit = llm_output.logits.argmax(-1)
        # print(f"{max_logit[0, :7]} ... {max_logit[0, -7:]}")
        return llm_output

    def _combine_embeds(
        self,
        audio_embed: torch.Tensor,
        text_embeds: torch.Tensor,
        audio_token_start_idx: t.Optional[torch.Tensor] = None,
        audio_token_len: t.Optional[torch.Tensor] = None,
    ):
        """
        Combining text and audio embeddings into the same tensor.

        The tokens should roughly be: `concat(preample, audio, text)`
        The above format doesn't account for some minor templating, but it's close enough.

        The preamble is the same for all samples, so we can just take the first one
        """

        embeds = text_embeds

        for i, (audio, start, length) in enumerate(
            zip(audio_embed, audio_token_start_idx, audio_token_len)
        ):
            length = min(length, audio.shape[0])
            embeds[i, start : start + length] = audio[:length]

        return embeds

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model (reuses Peft model's method)
        """
        count_params = peft.peft_model.PeftModel.get_nb_trainable_parameters

        trainable_params, all_param = count_params(self)

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d}"
            f" || trainable%: {100 * trainable_params / all_param:.1f}%"
        )

        llm_trainable_params, llm_all_params = count_params(self.llm)
        audio_enc_trainable_params, audio_enc_all_params = count_params(self.audio_enc)

        glue_trainable_params = (
            trainable_params - llm_trainable_params - audio_enc_trainable_params
        )
        glue_all_params = all_param - llm_all_params - audio_enc_all_params

        print(
            f"LLM trainable%: {100 * llm_trainable_params / llm_all_params:.1f}%"
            f" || Audio Encoder trainable%: {100 * audio_enc_trainable_params / audio_enc_all_params:.1f}%"
            f" || Glue trainable%: {100 * glue_trainable_params / glue_all_params:.1f}%"
        )

    # def to(
    #     self,
    #     device: t.Optional[t.Union[int, torch.device]] = None,
    #     dtype: t.Optional[t.Union[torch.dtype, str]] = None,
    #     **kwargs,
    # ):
    #     for name, child in self.named_children():
    #         if (
    #             name != "audio_enc"
    #             or dtype != torch.bfloat16
    #             or not self.config.is_auido_enc_w2vbert()
    #         ):
    #             child.to(device=device, dtype=dtype, **kwargs)
    #         else:
    #             logging.warning(
    #                 "Skipping conversion of audio_enc to bfloat16 since it's not supported."
    #             )
    #             child.to(device=device, **kwargs)


def audio_text_matching_loss(
    audio_embed: torch.Tensor,
    text_embeds: torch.Tensor,
    text_mask: torch.LongTensor,
):
    audio_embed = audio_embed.float()
    text_embeds = text_embeds.float()
    text_mask = text_mask.float()

    audio_token_count = audio_embed.shape[-2]
    # text_token_count = text_embeds.shape[-2]

    # We want to expand text embddings to match audio embeddings
    # Values in channel are interpolated independently, hence the transpose.
    # B x T x 2048  ->  B x A/3200 x 2048
    expanded_text_embeds = F.interpolate(
        text_embeds.transpose(-1, -2), size=audio_token_count, mode="linear"
    ).transpose(-1, -2)

    # B x T -> B x 1 x T -> B x T x 1
    expanded_mask = F.interpolate(
        text_mask.unsqueeze(1), size=audio_token_count, mode="linear"
    ).transpose(-1, -2)

    # B x T x 2048
    # element-wise loss: no reduction to allow masking padded tokens
    # loss_ew = F.mse_loss(audio_embed, expanded_text_embeds, reduction="none") + F.l1_loss(audio_embed, expanded_text_embeds, reduction="none")
    loss_ew = F.l1_loss(audio_embed, expanded_text_embeds, reduction="none")
    # BxTx2048 * BxTx1  ->  BxTx2048  ->  1  (scalar loss)
    masked_loss = (loss_ew * expanded_mask).mean() / expanded_mask.mean()

    return masked_loss
