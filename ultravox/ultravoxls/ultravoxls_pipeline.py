from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import transformers

# We must use relative import in this directory to allow uploading to HF Hub
# Even "from . import X" pattern doesn't work (undocumented and unclear why)
from .ultravoxls_model import UltravoxLSModel
from .ultravoxls_processing import DataCollatorForLSM
from .ultravoxls_processing import UltravoxLSProcessor

SAMPLE_RATE = 24000


class UltravoxLSPipeline(transformers.Pipeline):
    def __init__(self, model: UltravoxLSModel, **kwargs):
        self.processor = UltravoxLSProcessor()
        self.collate_fn = DataCollatorForLSM()
        super().__init__(model=model, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        generation_keys = ["temperature", "max_new_tokens", "repetition_penalty"]
        generation_kwargs = {k: kwargs[k] for k in kwargs if k in generation_keys}
        return {}, generation_kwargs, {}

    def preprocess(self, audio: np.ndarray, sampling_rate: int = SAMPLE_RATE):
        # Convert to float32 if needed.
        if isinstance(audio, np.ndarray):
            if audio.dtype == np.float64:
                audio = audio.astype(np.float32)
            elif audio.dtype == np.int16:
                audio = audio.astype(np.float32) / np.float32(32768.0)
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / np.float32(2147483648.0)

        if sampling_rate != SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=sampling_rate, target_sr=SAMPLE_RATE
            )

        inputs = self.processor(audio)
        inputs = self.collate_fn([inputs])
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        return inputs

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = 32,
        repetition_penalty: float = 1.1,
    ) -> List[int]:
        temperature = temperature or None
        do_sample = temperature is not None

        return self.model.generate(
            **model_inputs,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )

    def postprocess(self, model_outputs) -> str:
        return model_outputs[0]


transformers.pipelines.PIPELINE_REGISTRY.register_pipeline(
    "ultravoxls-pipeline",
    pipeline_class=UltravoxLSPipeline,
    pt_model=transformers.AutoModel,
    type="multimodal",
)
