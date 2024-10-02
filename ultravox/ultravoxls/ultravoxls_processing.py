from typing import Optional, Union

import numpy as np
import torch
import transformers


class UltravoxLSProcessor:
    tokenizer: transformers.PreTrainedTokenizerBase

    def __init__(
        self,
        tokenizer=None,
    ):
        self.tokenizer = tokenizer

    def __call__(
        self,
        audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_tensors: Optional[
            Union[str, transformers.TensorType]
        ] = transformers.TensorType.PYTORCH,
        **kwargs,
    ) -> transformers.BatchFeature:
        tokenized_audio = self.tokenizer.encode(audio)
        if len(tokenized_audio.shape) == 3:
            tokenized_audio = tokenized_audio.squeeze(1)

        attention_mask = (tokenized_audio != 0).astype(np.int64)
        data = {"input_ids": tokenized_audio, "attention_mask": attention_mask}
        return transformers.BatchFeature(data=data, tensor_type=return_tensors)
