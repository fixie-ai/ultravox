
from typing import Optional, Union, Iterable, NoReturn, Dict, Tuple, ClassVar
import os, hashlib, urllib
import tqdm

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np
from transformers import WhisperModel


def sinusoids(
    length: int,
    channels: int,
    max_timescale: int = 10000,
) -> Tensor:

    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class MultiHeadAttention(nn.Module):
    def __init__(self,
        n_state: int,
        n_head: int,
        causality: str,
        nblocks: int = None,
        bsize: int = None,
    ) -> NoReturn:
        super().__init__()

        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.causality = causality
        if causality == 'grouped-causal':
            self.nblocks = nblocks
            self.bsize = bsize

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> NoReturn:

        q = self.query(x)
        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)

        x = self.qkv_attention(q, k, v, mask=mask)
        return self.out(x)

    def qkv_attention(self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> NoReturn:

        B, T, C = q.shape
        scale = (C // self.n_head) ** -0.25
        if self.causality != 'grouped-causal':
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            qk = (q @ k)
            if mask is not None and self.causality in ('causal', 'bw-semi-causal'):
                qk = qk + mask[:T, :T]
            w = F.softmax(qk.float(), dim=-1).to(q.dtype)
            return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        else:
            q = q.view(B, self.nblocks, self.bsize, self.n_head, -1).permute(0, 3, 1, 2, 4) * scale
            k = k.view(B, self.nblocks, self.bsize, self.n_head, -1).permute(0, 3, 1, 4, 2) * scale
            v = v.view(B, self.nblocks, self.bsize, self.n_head, -1).permute(0, 3, 1, 2, 4)
            w = (q @ k).float().softmax(dim=-1).to(q.dtype)
            return (w @ v).permute(0, 2, 3, 1, 4).flatten(start_dim=3).view(B, T, C)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
        n_state: int, n_head: int, cross_attention: bool = False,
        causality: str = 'causal', nblocks: int = None, bsize: int = None,
    ) -> NoReturn:
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head, causality=causality, nblocks=nblocks, bsize=bsize)
        self.attn_ln = LayerNorm(n_state, eps=1e-8)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head, causality='non-causal') if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )

        self.mlp_ln = LayerNorm(n_state)

    def forward(self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        x = x + self.attn(self.attn_ln(x), mask=mask)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    base_model_prefix = 'model.encoder'
    _no_split_modules = ['WhisperEncoderLayer']

    def __init__(self, conf: ClassVar) -> NoReturn:
        super().__init__()

        n_state, n_head, n_layers = conf.d_model, conf.encoder_attention_heads, conf.encoder_layers
        dropout, n_mels, n_ctx = conf.dropout, conf.num_mel_bins, conf.max_source_positions

        self.dtype = torch.bfloat16 # bad
        # self.autocast = torch.autocast(device_type='cuda', dtype=self.dtype) # bad

        causality = 'non-causal'
        self.n_layers = n_layers
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer('positional_embedding', sinusoids(n_ctx, n_state))

        nblocks = None
        bsize = None

        if causality == 'causal':
            mask = torch.empty(n_ctx, n_ctx).fill_(float('-inf')).triu_(1)
            self.register_buffer('mask', mask, persistent=False)
        elif causality == 'bw-semi-causal':
            nblocks = 12 # 30 for one, 15 for two, 12 for two and half, 5 for six
            bsize = 125 # 50 for one, 100 for two, 125 for two and half, 300 for six
            mask = torch.tril(torch.ones(nblocks, nblocks), diagonal=0).repeat_interleave(bsize, dim=0).repeat_interleave(bsize, dim=1)
            mask[mask == 0] = float('-inf')
            mask[mask == 1] = 0
            self.register_buffer('mask', mask, persistent=False)
        elif causality == 'grouped-causal':
            nblocks = 30 # 30 for one, 15 for two, 12 for two and half, 5 for six
            bsize = 50 # 50 for one, 100 for two, 125 for two and half, 300 for six
            self.mask = None
            print('nblocks:', nblocks, ', ', 'bsize:', bsize)
        else:
            self.mask = None


        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state,
                n_head,
                causality=causality,
                nblocks=nblocks,
                bsize=bsize,
                ) for idx in range(n_layers)]
        )
        self.dropout = dropout
        self.ln_post = LayerNorm(n_state)
        self.expected_seq_length = 3000


    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] > self.expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length} or less, but found {x.shape[-1]}." \
                f"Make sure to pad the input mel features to {expected_seq_length}." \
            )
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        B, T, C = x.shape

        x = (x + self.positional_embedding[:T]).to(x.dtype)

        for i, block in enumerate(self.blocks):
            x = block(x, mask=self.mask)
        x = self.ln_post(x)
        return x

whisper_models = {
    'tiny.en': 'https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt',
    'small.en': 'https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt',
    'small': 'https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt',
}

def download_whisper(url: str, root: str) -> str:
    os.makedirs(root, exist_ok=True)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return download_target
        else:
            print(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm.tqdm(
            total=int(source.info().get('Content-Length')),
            ncols=80,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )
    return download_target

def audio_encoder_init(model_id: str = None, config: ClassVar = None) -> ClassVar:
    path = download_whisper(whisper_models['small'], 'whisper_models')
    checkpoint = torch.load(path, weights_only=True)['model_state_dict']
    checkpoint = {x.replace('encoder.', ''):checkpoint[x] for x in checkpoint if x.startswith('encoder')}
    audio_encoder = AudioEncoder(config)
    if model_id is not None:
        audio_encoder.load_state_dict(checkpoint)

    return audio_encoder
