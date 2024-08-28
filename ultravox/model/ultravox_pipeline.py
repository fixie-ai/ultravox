import logging
from typing import Any, Dict, List, Optional

import numpy as np
import transformers

# We must use relative import in this directory to allow uploading to HF Hub
# Even "from . import X" pattern doesn't work (undocumented and unclear why)
from .ultravox_model import UltravoxModel
from .ultravox_processing import UltravoxProcessor


class UltravoxPipeline(transformers.Pipeline):
    def __init__(
        self,
        model: UltravoxModel,
        tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
        audio_processor: Optional[transformers.ProcessorMixin] = None,
        **kwargs
    ):
        if tokenizer is None:
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model.config._name_or_path
                )
            except:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model.config.text_model_id or model.config.text_config._name_or_path
                )

        if audio_processor is None:
            audio_processor = transformers.AutoProcessor.from_pretrained(
                model.config.audio_model_id or model.config.audio_config._name_or_path
            )

        self.processor = UltravoxProcessor(
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            adapter=model.adapter,
        )

        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        generation_keys = ["temperature", "max_new_tokens", "repetition_penalty"]
        generation_kwargs = {k: kwargs[k] for k in kwargs if k in generation_keys}
        return {}, generation_kwargs, {}

    def preprocess(self, inputs: Dict[str, Any]):
        turns: list = inputs.get("turns", [])

        audio = inputs.get("audio", None)
        # Convert to float32 if needed.
        if isinstance(audio, np.ndarray):
            if audio.dtype == np.float64:
                audio = audio.astype(np.float32)
            elif audio.dtype == np.int16:
                audio = audio.astype(np.float32) / np.float32(32768.0)
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / np.float32(2147483648.0)

        if audio is not None and (len(turns) == 0 or turns[-1]["role"] != "user"):
            prompt = inputs.get("prompt", "<|audio|>")
            if "<|audio|>" not in prompt:
                logging.warning(
                    "Prompt does not contain '<|audio|>', appending '<|audio|>' to the end of the prompt."
                )

                prompt += " <|audio|>"
            turns.append({"role": "user", "content": prompt})

        text = self.processor.tokenizer.apply_chat_template(
            turns, add_generation_prompt=True, tokenize=False
        )

        if "sampling_rate" not in inputs and audio is not None:
            logging.warning(
                "No sampling rate provided, using default of 16kHz. We highly recommend providing the correct sampling rate."
            )

        output = self.processor(
            text=text,
            audio=audio,
            sampling_rate=inputs.get("sampling_rate", 16000),
        )
        if "audio_values" in output:
            output["audio_values"] = output["audio_values"].to(self.model.dtype)

        return output

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: float = 1.1,
    ) -> List[int]:
        temperature = temperature or None
        do_sample = temperature is not None

        terminators = [self.tokenizer.eos_token_id]
        if "<|eot_id|>" in self.tokenizer.added_tokens_encoder:
            terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

        input_len = model_inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **model_inputs,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            eos_token_id=terminators
        )
        return outputs[0][input_len:]

    def postprocess(self, model_outputs) -> str:
        output_text = self.tokenizer.decode(model_outputs, skip_special_tokens=True)
        return output_text


transformers.pipelines.PIPELINE_REGISTRY.register_pipeline(
    "ultravox-pipeline",
    pipeline_class=UltravoxPipeline,
    pt_model=transformers.AutoModel,
    type="multimodal",
)
